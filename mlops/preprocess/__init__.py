import sys
from typing import Optional

import spacy
from torchtext.data import BucketIterator, Field, TabularDataset


def get_path_to_train_data() -> str:
    """Give path to training data

    Returns path to the training dataset `data.tsv`.
    If unavailable - through OSError.

    Args:

        None: nothing to take in.

    Return:

        str: path to the training dataset `data.tsv`.

    """

    try:
        file = open("./output/tmp.txt", "r")
    except OSError:
        print("Model was not trained!")
        sys.exit()
    else:
        path = next(file)
        file.close()
    return path


def init_fields(
    SRC_LANG: str,
    TRG_LANG: str,
    MIN_FREQ: int,
    MAX_SIZE: int,
    path: str = "",
) -> tuple[Field, Field]:
    """Initialize fields

    Load the `tmp.txt` file with path to the train data `data.tsv`.
    Construct source and target vocabularies.

    Args:

        path: string with path to the train data `data.tsv`
            If passed: use it.
            Else look for `tmp.txt` file.

    Return:

        Field: source language torchtext.data.Field.
        Field: target language torchtext.data.Field.

    """

    path = get_path_to_train_data() if path == "" else path

    spacy_ru = spacy.load(SRC_LANG)
    spacy_en = spacy.load(TRG_LANG)

    def tokenize_en(text: str) -> list:
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_ru(text: str) -> list:
        return [tok.text for tok in spacy_ru.tokenizer(text)]

    src = Field(
        tokenize=tokenize_ru,
        init_token="<sos>",
        eos_token="<eos>",
    )

    trg = Field(
        tokenize=tokenize_en,
        init_token="<sos>",
        eos_token="<eos>",
    )

    fields = {
        "Target": ("trg", trg),
        "Source": ("src", src),
    }

    try:
        train_data = TabularDataset(
            path=path,
            format="tsv",
            fields=fields,
        )
    except OSError:
        print("Error in constructing train data: {}.".format(path))
        sys.exit()

    src.build_vocab(
        train_data,
        max_size=MAX_SIZE,
        min_freq=MIN_FREQ,
    )

    trg.build_vocab(
        train_data,
        max_size=MAX_SIZE,
        min_freq=MIN_FREQ,
    )
    return (src, trg)


def read_test_data(path: str) -> Optional[list[str]]:
    """Reads the test dataset

    Take path to the test data and tries to open it.
    If: file exists, then it returns a list of sentences to be translated.
    Else: (files does not exist) through an OSError.

    Args:

        path: path to the test dataset `test.txt`.

    Return:

        Optional[list] - if `path` file exists returns the list
            of sentences to be translated. Else - through OSError.

    """
    try:
        file = open(path, "r")
    except OSError:
        print("Could not find file: {}.".format(path))
        sys.exit()
    else:
        dataset = [line.strip() for line in file]
        file.close()
    return dataset


def read_train_data(
    path: str,
    src: Field,
    trg: Field,
    batch_size: int,
    device: str,
) -> BucketIterator:
    """Give train data in train cycle

    Get path to the train data, Fields for source
    and target, batch_size and device.
    Forms train_iterator (Bucket Iterator) and returns it.

    Args:

        path: str - path to train data.
        src: Field - pretrained field for train data (source language)
        trg: Field - pretrained field for train data (target language)
        batch_size: int - how many observations should be there is a batch.
        device: str - what to use 'cuda' or 'cpu' devices.

    Return:

        BucketIterator: train data ready for model training cycle.

    """

    fields = {
        "Source": ("src", src),
        "Target": ("trg", trg),
    }

    try:
        train_data = TabularDataset(
            path=path,
            format="tsv",
            fields=fields,
        )
    except OSError:
        print("Unable to form train dataset!")
        sys.exit()

    train_iter = BucketIterator(
        dataset=train_data,
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    return train_iter
