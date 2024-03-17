import os
from datetime import datetime

import torch
from torchtext.data import BucketIterator
from tqdm.auto import tqdm

from mlops.models.loading import init_model
from mlops.models.transformer import Transformer
from mlops.preprocess import init_fields, read_train_data
from mlops.utils import init_ml_constants, save_checkpoint


def train_main(
    train_iter: BucketIterator,
    model: Transformer,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: str,
    params: dict,
) -> None:
    """Train and save weights of the model

    Trains the model and saves its weights to `output` directory.

    Args:
        train_iter: BucketIterator - train dataset iterator.
        model: Transformer - model instance of Transformer.
        criterion: torch.nn.Module - cost (loss) function.
        optimizer: torch.optim.Optimizer - optimizer for training model.
        lr_scheduler: torch.optim.lr_scheduler.MultiStepLR - learning rate.
        device: str - use 'cuda' or 'cpu' to train model.
        params: dict - other hyperparameters for training process.

    Return:
        None: nothing to return.

    """

    pbar_scheme = "Epoch: {}/{} | "
    pbar_scheme += "Train cost: {:.5f} | "
    pbar_scheme += "Learning Rate: {:.6f} | "
    pbar_scheme += "Time: {}"

    pbar_batch_scheme = "Epoch: {}/{} | "
    pbar_batch_scheme += "Batch: {}/{} | "
    pbar_batch_scheme += "Train cost: {:.5f} | "
    pbar_batch_scheme += "Batch train cost: {:.5f} | "
    pbar_batch_scheme += "L.R.: {:.6f}"

    epochs = params["epochs"]
    pbar = tqdm(range(epochs), ncols=150)
    for epoch in pbar:

        save_checkpoint(model.state_dict())

        losses = []
        start = datetime.now()

        model.train()
        for batch_idx, batch in enumerate(train_iter):
            inp_data = batch.src.to(device)  # type: ignore
            target = batch.trg.to(device)  # type: ignore

            output = model(inp_data, target[:-1, :])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1,
            )
            optimizer.step()

            pbar.set_description(
                pbar_batch_scheme.format(
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    len(train_iter),
                    sum(losses) / len(losses),
                    losses[-1],
                    optimizer.param_groups[0]["lr"],
                )
            )

        end = datetime.now()

        mean_loss = sum(losses) / len(losses)
        lr_scheduler.step()

        pbar.set_description(
            pbar_scheme.format(
                epoch + 1,
                epochs,
                mean_loss,
                optimizer.param_groups[0]["lr"],
                (end - start).total_seconds(),
            )
        )


def train(path: str) -> None:
    """Train the model.

    Train and save the tuned weights of the model.

    Args:
        path: path to train dataset `train.tsv`

    Return:
        No return.

    Side effect:
        Creates `output` dir in the current folder to store
            1. checkpoint.ptr.tar - pretrained weights.
            2. tmp.txt - path to train data (data.tsv).
    """
    if "output" not in os.listdir():
        os.mkdir("output")
    with open("./output/tmp.txt", "w") as file:
        print(path, file=file, end="")

    params_ml = init_ml_constants(
        need="model",
    )
    params_preprocess = init_ml_constants(
        need="preprocess",
    )
    params_train = init_ml_constants(
        need="train",
    )
    params = params_ml | params_preprocess | params_train

    src, trg = init_fields(
        SRC_LANG=params["src_lang"],
        TRG_LANG=params["trg_lang"],
        MIN_FREQ=params["min_freq"],
        MAX_SIZE=params["max_size"],
        path=path,
    )

    SRC_VOC_SIZE = len(src.vocab)
    TRG_VOC_SIZE = len(trg.vocab)
    SRC_PAD_IDX = trg.vocab.stoi["<pad>"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_FLAG = True

    dataset = read_train_data(
        path=path,
        src=src,
        trg=trg,
        batch_size=params["batch_size"],
        device=DEVICE,
    )

    model = init_model(
        train_flag=TRAIN_FLAG,
        src_voc_size=SRC_VOC_SIZE,
        trg_voc_size=TRG_VOC_SIZE,
        src_pad_idx=SRC_PAD_IDX,  # type: ignore
        device=DEVICE,
        **params_ml,
        seed=params_preprocess["seed"],
    )

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=SRC_PAD_IDX,  # type: ignore
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        amsgrad=True,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[0, 4, 8, 16],
        gamma=0.1,
    )

    train_main(
        train_iter=dataset,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        params=params,
    )
