import torch

from mlops.constants import SEED
from mlops.models.transformer import Transformer
from mlops.utils import load_checkpoint, reproducibility


def init_model(
    train: bool,
    emb_size: int,
    num_heads: int,
    src_voc_size: int,
    trg_voc_size: int,
    num_enc_layers: int,
    num_dec_layers: int,
    src_pad_idx: int,
    forward_expansion: int,
    dropout: float,
    max_len: int,
    device: str,
) -> Transformer:
    """Initialize model instance

    Creates instance of Transformer class, but it on device
    and give it back for any purposes.

    Args:

        train: bool - if we are training the model or not.
        emb_size: int - size of the embedding in model's block.
        num_heads: int - number of attention heads in the model.
        src_voc_size: int - size of the source vocabulary.
        trg_voc_size: int - size of the target vocabulary.
        num_enc_layers: int - number of encoder layers.
        num_dec_layers: int - number of decoder layers.
        src_pad_idx: int - source padding index.
        forward_expansion: int - forward expansion for the model.
        dropout: float - probability of neuron to switch off.
        max_len: int - max length of the translated sentence.
        device: str - cuda or cpu device.

    Return:

        Transformer: torch.nn.Module instance; the model.

    """

    if train:
        reproducibility(SEED)

    model = Transformer(
        embedding_size=emb_size,
        src_vocab_size=src_voc_size,
        trg_vocab_size=trg_voc_size,
        src_pad_idx=src_pad_idx,
        num_heads=num_heads,
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        forward_expansion=forward_expansion,
        max_len=max_len,
        dropout=dropout,
        device=device,
    ).to(device)

    if not train:
        load_checkpoint(
            checkpoint=torch.load("./checkpoint.ptr.tar"),
            model=model,
        )

    return model
