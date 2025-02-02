import os
import numpy as np
import trax
from trax import layers as tl
from trax.data import inputs

def quantum_circuit_data_generator(input_file: str, output_file: str):
    """
    Generator that reads input and output files and yields circuit pairs.
    
    Each line in the input and output file is assumed to be a space-separated
    sequence of integers.
    """
    with open(input_file, 'r') as in_file, open(output_file, 'r') as out_file:
        for in_line, out_line in zip(in_file, out_file):
            yield (in_line.strip(), out_line.strip())

def get_data_pipelines(input_filepath: str, output_filepath: str, batch_size: int):
    """
    Creates a data pipeline for training or evaluation.
    
    Args:
        input_filepath: Path to the file containing input circuits.
        output_filepath: Path to the file containing target circuits.
        batch_size: Number of samples per batch.
        
    Returns:
        A Trax data pipeline that yields batches.
    """
    def preprocess(data_pair):
        inp, outp = data_pair
        return {
            'inputs': np.array(list(map(int, inp.split()))),
            'targets': np.array(list(map(int, outp.split())))
        }

    data_pipeline = trax.data.Serial(
        lambda _: quantum_circuit_data_generator(input_filepath, output_filepath),
        preprocess,
        trax.data.Batch(batch_size=batch_size)
    )
    return data_pipeline

def transformer_model(vocab_size: int, d_model: int = 128, d_ff: int = 512,
                        n_heads: int = 4, n_layers: int = 2, mode: str = 'train'):
    """
    Defines a Transformer model for quantum circuit optimization.
    
    Args:
        vocab_size: Size of the vocabulary (number of unique tokens).
        d_model: Dimensionality of the model embeddings.
        d_ff: Dimensionality of the feed-forward layers.
        n_heads: Number of attention heads.
        n_layers: Number of encoder blocks.
        mode: The mode in which the model operates ('train' or 'predict').
        
    Returns:
        A Trax serial model.
    """
    return tl.Serial(
        tl.Embedding(vocab_size, d_model),
        [tl.EncoderBlock(d_model, d_ff, n_heads, dropout=0.6) for _ in range(n_layers)],
        tl.Mean(axis=1),  # Mean pooling over sequence length
        tl.Dense(vocab_size),
        tl.LogSoftmax()
    )

def get_model(mode: str = 'train'):
    """
    Instantiates the Transformer model with a fixed vocabulary size.
    
    Args:
        mode: Mode for the model ('train' or 'predict').
        
    Returns:
        A Trax model instance.
    """
    VOCAB_SIZE = 2**16
    return transformer_model(VOCAB_SIZE, mode=mode)
