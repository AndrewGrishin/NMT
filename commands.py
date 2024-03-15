import warnings

import fire

from mlops.infer import infer
from mlops.train import train

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    fire.Fire(
        {
            "infer": infer,
            "train": train,
        }
    )
