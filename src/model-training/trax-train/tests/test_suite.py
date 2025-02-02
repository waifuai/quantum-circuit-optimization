import os
import tempfile
import numpy as np
import pytest
import trax
from trax.fastmath import numpy as jnp

# Import functions from your modules.
from src.prep import preprocess_data
from src.utils import load_data, data_generator
from src.predict import (
    get_tokenizers,
    tokenize_sentence,
    greedy_decode,
    beam_search,
)
from src.train import create_transformer_model

# --- Dummy model for testing decoding functions ---
def dummy_model_greedy(inputs):
    """
    Dummy model for testing decoding.
    Returns predictions with high probability for token index 2 in the first 3 positions,
    then a high probability for token index 0 (EOS/padding) afterwards.
    """
    tokenized_input, current_output = inputs
    batch, length = current_output.shape
    vocab_size = 5  # arbitrary small vocabulary
    # Initialize predictions with very low log probabilities.
    dummy_pred = jnp.full((batch, length, vocab_size), -1000.0, dtype=jnp.float32)
    for i in range(length):
        if i < 3:
            # Force token index 2 to be the maximum.
            dummy_pred = dummy_pred.at[0, i, 2].set(1000.0)
        else:
            # After 3 tokens, return EOS (assumed token 0).
            dummy_pred = dummy_pred.at[0, i, 0].set(1000.0)
    return dummy_pred

# --- Tests for helper functions in predict.py ---
def test_get_tokenizers():
    vocab = ['a', 'b', 'c']
    char_to_index, index_to_char = get_tokenizers(vocab)
    assert char_to_index == {'a': 0, 'b': 1, 'c': 2}
    assert index_to_char == {0: 'a', 1: 'b', 2: 'c'}

def test_tokenize_sentence():
    vocab = ['a', 'b', 'c']
    char_to_index, _ = get_tokenizers(vocab)
    sentence = "abcab"
    max_length = 7
    tokens = tokenize_sentence(sentence, char_to_index, max_length)
    # The tokenized sentence should have shape (1, max_length)
    assert tokens.shape == (1, max_length)
    # Since 'a'->0, 'b'->1, 'c'->2, and unknown tokens default to 0:
    expected = [0, 1, 2, 0, 1, 0, 0]
    np.testing.assert_array_equal(tokens, np.array([expected]))

def test_greedy_decode():
    # Define a dummy vocabulary.
    vocab = ['<pad>', 'x', 'y', 'z']
    # Our dummy model (dummy_model_greedy) is set so that the first three decoded tokens are index 2.
    # Since index 2 corresponds to 'y', we expect the output to be "yyy".
    output = greedy_decode(dummy_model_greedy, "test", vocab, max_length=10)
    assert output == "yyy"

def test_beam_search():
    # For beam search, using the same dummy model should yield a similar outcome.
    vocab = ['<pad>', 'x', 'y', 'z']
    output = beam_search(dummy_model_greedy, "test", vocab, beam_width=2, max_length=10)
    # Our dummy model will pick token index 2 ("y") for the first few positions.
    # Check that the resulting string contains the expected character.
    assert "y" in output

# --- Tests for utils.py functions ---
def test_load_data_and_generator(tmp_path):
    # Create temporary input and output processed files.
    input_data = "abc\ndef\n"
    output_data = "bcd\nef\n"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input_processed.txt"
    output_file = data_dir / "output_processed.txt"
    input_file.write_text(input_data)
    output_file.write_text(output_data)
    
    inputs, targets, vocab = load_data(str(data_dir), "input_processed.txt", "output_processed.txt")
    
    # Verify that the files are read correctly.
    assert inputs == ["abc", "def"]
    assert targets == ["bcd", "ef"]
    # Check that the vocabulary contains all characters from the files.
    for char in "abcdef":
        assert char in vocab
    
    # Test that the data generator yields batches with the correct shape.
    gen = data_generator(inputs, targets, vocab, batch_size=1, max_length=5)
    input_batch, target_batch, mask_batch = next(gen)
    assert input_batch.shape == (1, 5)
    assert target_batch.shape == (1, 5)
    assert mask_batch.shape == (1, 5)

# --- Test for train.py model creation ---
def test_create_transformer_model():
    vocab_size = 10
    max_length = 16
    model = create_transformer_model(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        d_model=256,
        d_ff=512,
        n_layers=2,
        n_heads=4,
        max_len=max_length,
        mode='train'
    )
    # Verify that the returned model is a Trax Serial model.
    assert isinstance(model, trax.layers.combinators.Serial)

# --- Test for prep.py data preprocessing ---
def test_preprocess_data(tmp_path):
    # Create a temporary input file with leading whitespace.
    input_lines = ["  line1\n", "  line2\n", "  line3\n"]
    input_file = tmp_path / "input.txt"
    input_file.write_text("".join(input_lines))
    # Define paths for processed files.
    input_processed_file = tmp_path / "input_processed.txt"
    output_processed_file = tmp_path / "output_processed.txt"
    # Run the preprocessing function.
    preprocess_data(str(input_file), str(input_processed_file), str(output_processed_file))
    
    processed_input = input_processed_file.read_text().splitlines()
    processed_output = output_processed_file.read_text().splitlines()
    # The function removes the last line for input and the first line for output.
    # Also, leading whitespace should be stripped.
    assert processed_input == ["line1", "line2"]
    assert processed_output == ["line2", "line3"]
