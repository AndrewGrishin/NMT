import fire

from mlops.infer import infer
from mlops.train import train

# train and infer are called
# from the cli interface

if __name__ == "__main__":
    fire.Fire()
