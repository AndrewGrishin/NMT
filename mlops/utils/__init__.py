import os
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml
from yaml.loader import SafeLoader

from mlops.models.transformer import Transformer


def count_parameters(model: Transformer) -> int:
    """Counts parameters

    Count model's tunable parameters.

    Args:

        model: torch.nn.Module instance.

    Return:

        int: number of tunable parameters.

    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    state: Any,
) -> None:
    """Save model

    Saves model's and optimizer's weights to the file.

    Args:

        state: model.state_dict()

    """
    if "output" not in os.listdir():
        os.mkdir("output")
    filename: str = "./output/checkpoint.ptr.tar"
    torch.save(state, filename)


def load_checkpoint(
    checkpoint: Any,
    model: Transformer,
) -> None:
    """Load model

    Loads model's checkpoint.

    Args:

        checkpoint: torch.load(<path>) return object.
        model: torch.nn.Module instance to load weights in.

    Return:

        None: nothing to return. Weights are loaded in the OOP format.

    """

    model.load_state_dict(checkpoint)


def reproducibility(SEED: int) -> None:
    """Make research reproducible.

    Fixes seeds to make generation of pseudo random numbers predictable.

    Args:

        SEED: integer to fix the starting seed.

    Return:

        None: nothing to return.

    """

    torch.random.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True


def init_ml_constants(
    need: str,
    path: str = "config/ml.yaml",
) -> dict:
    """Initialize hyperparameters

    Reads the yaml file and returns the dict.
    of available hyperparameters.

    Args:

        path: str - address of ml.yaml configuration.

    Return:

        dict: dictionary structure of all hyperparameters.

    """
    try:
        file = open(path, "r")
    except OSError:
        print("Cannot find file: {}.".format(path))
        sys.exit()
    else:
        conf_vault = yaml.load(file, SafeLoader)
        file.close()

    if need not in set(list(conf_vault.keys())):
        print(
            "Expected: {}. Received: {}.".format(
                list(conf_vault.keys()),
                need,
            )
        )
        sys.exit()

    return conf_vault[need]
