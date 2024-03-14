import random
from typing import Any

import numpy as np
import torch

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

    filename: str = "./checkpoint.ptr.tar"
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
