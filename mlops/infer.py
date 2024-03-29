import os
from typing import Optional, Union

import spacy
import torch
from torchtext.data import Field
from tqdm.auto import tqdm

from mlops.models.loading import init_model
from mlops.models.transformer import Transformer
from mlops.preprocess import init_fields, read_test_data
from mlops.utils import init_ml_constants


def translate_sentence(
    model: Transformer,
    sentence: Union[str, list],
    src: Field,
    trg: Field,
    device: str,
    max_length: int = 50,
) -> list:
    """Translate given sentence

    Function translates the given sentence in the `source` language
    into the its equivalent in the `target` language. In our case
    from Russian into English.

    Args:

        model: instance of torch.nn.Module (Transformer in our case).
        sentence: string or list of token in
            the source language to be translated.
        src: torchtext.data.Field instance with source language.
        trg: torchtext.data.Field instance with target language.
        device: string of available device to use for evaluation (cuda | cpu).
        max_length: integer with the max length of the translated sentence.

    Return:

        list: list of translated sentences.

    """

    # Load src tokenizer
    spacy_ger = spacy.load("en_core_web_sm")

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, src.init_token)  # type: ignore
    tokens.append(src.eos_token)  # type: ignore

    # Go through each src token and convert to an index
    text_to_indices = [src.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [trg.vocab.stoi["<sos>"]]

    model.eval()
    with torch.inference_mode():
        for _ in range(max_length):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

            output = model(sentence_tensor, trg_tensor)
            best_guess = output.argmax(2)[-1, :].item()
            outputs.append(best_guess)

            if best_guess == trg.vocab.stoi["<eos>"]:
                break

        transl_sent = [trg.vocab.itos[idx] for idx in outputs]  # type: ignore

        # remove start token
    return transl_sent[1:]


def infer_main(
    dataset: Optional[list[str]],
    model: Transformer,
    src: Field,
    trg: Field,
    device: str,
    max_length: int,
) -> None:
    """Inference step of the model

    Execute the inference of the model (same as `infer` function)
    but with more parameters, so it is called consequently after the
    `infer` function. `infer` is some kind of wrapper over `infer_main`.

    Args:

        dataset: list of strings to be translated.
        model: instance of torch.nn.Module (Transformer in our case)
        src: instance of torchtext.data.Field with source language.
        trg: instance of torchtext.data.Field with target language.
        device: string of available device to use for evaluation (cuda | cpu).
        max_length: integer with the max length of the translated sentence.

    Return:

        None: nothing to return. The `output.txt` file with translated
            sentences will be save to the current directory.

    """
    ans = list()
    pbar = tqdm(dataset, desc="Inference", ncols=80)
    for sentence in pbar:
        translated = translate_sentence(
            model=model,
            sentence=sentence,
            src=src,
            trg=trg,
            device=device,
            max_length=max_length,
        )
        translated = " ".join(translated[:-2]) + translated[-2]
        ans.append(translated)

    output_file = "output.txt"
    with open("./output/" + output_file, "w") as file:
        print("\n".join(ans), end="", file=file)
    print("File {} added to directory output/.".format(output_file))


def infer(path: str) -> None:
    """Inference phase for pretrained model.

    Load tuned weights of the model.
    Evaluate it on the test dataset.
    Saves the `output.txt` in the current dir.

    Args:

        path: str - path to test dataset `test.txt`

    Return:

        None: nothing to return.

    """

    # create folder to store output.txt
    if "output" not in os.listdir():
        os.mkdir("./output")

    dataset = read_test_data(path=path)
    params_ml = init_ml_constants(
        need="model",
    )
    params_preprocess = init_ml_constants(
        need="preprocess",
    )

    params = params_ml | params_preprocess
    src, trg = init_fields(
        SRC_LANG=params["src_lang"],
        TRG_LANG=params["trg_lang"],
        MIN_FREQ=params["min_freq"],
        MAX_SIZE=params["max_size"],
    )

    SRC_VOC_SIZE = len(src.vocab)
    TRG_VOC_SIZE = len(trg.vocab)
    SRC_PAD_IDX = trg.vocab.stoi["<pad>"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_FLAG = False

    model = init_model(
        train_flag=TRAIN_FLAG,
        src_voc_size=SRC_VOC_SIZE,
        trg_voc_size=TRG_VOC_SIZE,
        src_pad_idx=SRC_PAD_IDX,  # type: ignore
        device=DEVICE,
        **params_ml,
        seed=params_preprocess["seed"]
    )

    infer_main(
        dataset=dataset,
        model=model,
        src=src,
        trg=trg,
        device=DEVICE,
        max_length=params["max_len"],
    )
