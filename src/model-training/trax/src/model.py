"""
Model definition for the Trax model.
"""

from trax import layers as tl

def transformer_model(vocab_size: int, d_model: int = 128, d_ff: int = 512,
                        n_heads: int = 4, n_layers: int = 2, mode: str = 'train') -> tl.Serial:
    """
    Defines the Transformer model.

    Args:
        vocab_size: Vocabulary size.
        d_model:  Depth of embedding (n_units in the attention layer).
        d_ff:  Depth of feed-forward layer.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        mode: 'train' or 'eval'.

    Returns:
        A Trax Serial model.
    """
    return tl.Serial(
        tl.Embedding(vocab_size, d_model),
        [tl.EncoderBlock(d_model, d_ff, n_heads, dropout=0.1) for _ in range(n_layers)],
        tl.Dense(vocab_size),
        tl.LogSoftmax()
    )


def get_model(vocab_size: int, mode: str = 'train') -> tl.Serial:
    """Instantiates the Transformer model."""
    return transformer_model(vocab_size, mode=mode)