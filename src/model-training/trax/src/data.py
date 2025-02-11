"""
Data loading and preprocessing functions for the Trax model.
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Dict
import trax

def quantum_circuit_data_generator(input_file: str, output_file: str) -> set:
    """
    Generates pairs of input and output quantum circuits.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.

    Yields:
        Pairs of input and output quantum circuits.
    """
    vocab: set = set()
    try:
        with open(input_file, 'r') as in_file, open(output_file, 'r') as out_file:
            for in_line, out_line in zip(in_file, out_file):
                in_line = in_line.strip()
                out_line = out_line.strip()
                vocab.update(in_line.split())
                vocab.update(out_line.split())
                yield (in_line, out_line)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    return vocab


def preprocess(data_pair: Tuple[str, str]) -> Dict[str, np.ndarray]:
    """
    Converts input and output strings to numpy arrays.

    Args:
        data_pair: A tuple containing the input and output strings.

    Returns:
        A dictionary containing the input and output numpy arrays.
    """
    inp: str
    outp: str
    inp, outp = data_pair
    return {
        'inputs': np.array(list(map(int, inp.split()))),
        'targets': np.array(list(map(int, outp.split())))
    }


def create_data_pipeline(input_filepath: str, output_filepath: str, batch_size: int) -> Tuple[trax.data.Serial, int]:
    """
    Creates a data pipeline for training and evaluation.

    Args:
        input_filepath: Path to the processed input file.
        output_filepath: Path to the processed output file.
        batch_size: Batch size for training.

    Returns:
        A tuple containing the data pipeline and the vocabulary size.
    """
    data_generator = lambda _: quantum_circuit_data_generator(input_filepath, output_filepath)
    vocab: set = data_generator(None)
    vocab_size: int = len(vocab)

    data_pipeline: trax.data.Serial = trax.data.Serial(
        data_generator,
        preprocess,
        trax.data.Batch(batch_size=batch_size)
    )
    return data_pipeline, vocab_size