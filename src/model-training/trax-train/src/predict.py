import os
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
from utils import load_data
from train import create_transformer_model

# Constants
BATCH_SIZE = 16
MAX_LENGTH = 256
N_LAYERS = 2
D_MODEL = 256
D_FF = 512
N_HEADS = 4
MODE = 'predict'  # Important for positional encoding

DATA_DIR = "data"
MODEL_DIR = "model"
PREDICT_INPUT_FILE = "phrases_input.txt"

def get_tokenizers(vocab: list):
    """Returns dictionaries for converting between characters and token indices."""
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {index: char for index, char in enumerate(vocab)}
    return char_to_index, index_to_char

def tokenize_sentence(sentence: str, char_to_index: dict, max_length: int) -> np.ndarray:
    """Tokenizes and pads a sentence to a fixed length."""
    tokenized = [char_to_index.get(char, 0) for char in sentence][:max_length]
    tokenized += [0] * (max_length - len(tokenized))
    return np.array(tokenized)[None, :]  # Add batch dimension

def greedy_decode(model, input_sentence: str, vocab: list, max_length: int = MAX_LENGTH) -> str:
    """
    Greedy decoding for the Transformer model.
    """
    char_to_index, index_to_char = get_tokenizers(vocab)
    tokenized_input = tokenize_sentence(input_sentence, char_to_index, max_length)
    current_output = jnp.zeros((1, max_length), dtype=jnp.int32)
    output = []

    for i in range(max_length):
        predictions = model((tokenized_input, current_output))
        next_token = int(jnp.argmax(predictions[0, i, :]))
        output.append(index_to_char.get(next_token, '<UNK>'))
        current_output = current_output.at[:, i].set(next_token)
        if next_token == 0:  # Stop if padding/EOS token reached
            break

    return "".join(output)

def beam_search(model, input_sentence: str, vocab: list, beam_width: int, max_length: int = MAX_LENGTH) -> str:
    """
    Beam search decoding for the Transformer model.
    """
    char_to_index, index_to_char = get_tokenizers(vocab)
    tokenized_input = tokenize_sentence(input_sentence, char_to_index, max_length)
    initial_beam = {"sequence": jnp.zeros((1, max_length), dtype=jnp.int32), "score": 0.0}
    beams = [initial_beam]

    for i in range(max_length):
        new_beams = []
        for beam in beams:
            # If EOS has been generated, keep the beam unchanged.
            if beam["sequence"][0, i] == 0 and i != 0:
                new_beams.append(beam)
                continue

            predictions = model((tokenized_input, beam["sequence"]))
            log_probs = jnp.log(predictions[0, i, :])
            top_k_indices = jnp.argpartition(log_probs, -beam_width)[-beam_width:]
            top_k_log_probs = log_probs[top_k_indices]

            for j in range(beam_width):
                next_token = top_k_indices[j]
                new_sequence = beam["sequence"].at[:, i].set(next_token)
                new_score = beam["score"] + float(top_k_log_probs[j])
                new_beams.append({"sequence": new_sequence, "score": new_score})

        beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]

    best_beam = beams[0]
    decoded = [index_to_char.get(int(token), '<UNK>') for token in best_beam["sequence"][0]]
    return "".join(decoded).replace("<pad>", "").strip()

def decode_prediction(model, input_sentence: str, vocab: list, beam_width: int, max_length: int) -> str:
    """Decodes the prediction using greedy decoding or beam search based on beam_width."""
    if beam_width == 1:
        return greedy_decode(model, input_sentence, vocab, max_length)
    return beam_search(model, input_sentence, vocab, beam_width, max_length)

def predict_from_file(model, vocab, input_file: str, beam_width: int = 1, max_length: int = MAX_LENGTH) -> None:
    """Reads circuits from a file, decodes them, and prints the optimized circuits."""
    with open(input_file, "r") as f:
        for line in f:
            input_circuit = line.strip()
            optimized_circuit = decode_prediction(model, input_circuit, vocab, beam_width, max_length)
            print(f"Input Circuit:    {input_circuit}")
            print(f"Optimized Circuit:{optimized_circuit}\n")

def interactive_prediction(model, vocab, beam_width: int = 1, max_length: int = MAX_LENGTH) -> None:
    """Allows interactive prediction of quantum circuits."""
    while True:
        input_circuit = input("Enter a quantum circuit (or 'q' to quit): ")
        if input_circuit.lower() == 'q':
            break
        optimized_circuit = decode_prediction(model, input_circuit, vocab, beam_width, max_length)
        print(f"Input Circuit:    {input_circuit}")
        print(f"Optimized Circuit:{optimized_circuit}\n")

def main():
    _, _, vocab = load_data(DATA_DIR, "input_processed.txt", "output_processed.txt")
    model = create_transformer_model(
        len(vocab),
        len(vocab),
        D_MODEL,
        D_FF,
        N_LAYERS,
        N_HEADS,
        MAX_LENGTH,
        MODE
    )
    # Initialize model with proper input shapes.
    shape = (1, MAX_LENGTH)
    model.init(trax.shapes.ShapeDtype((shape, shape), dtype=jnp.int32))
    model.init_from_file(os.path.join(MODEL_DIR, "model.pkl.gz"), weights_only=True)

    choice = input("Predict from file (f) or interactively (i)? ")
    beam_width = int(input("Enter beam width (1 for greedy decoding): "))
    if choice.lower() == "f":
        predict_from_file(model, vocab, PREDICT_INPUT_FILE, beam_width, MAX_LENGTH)
    elif choice.lower() == "i":
        interactive_prediction(model, vocab, beam_width, MAX_LENGTH)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
