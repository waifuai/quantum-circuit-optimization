import os
import sys
import numpy as np
import trax
from trax import layers as tl
from src.trainer.problem import get_model

def tokenize_circuit(circuit_str: str) -> np.ndarray:
    """Tokenizes the input circuit string into a numpy array of integers."""
    return np.array([list(map(int, circuit_str.split()))])

def detokenize_output(prediction: np.ndarray) -> str:
    """Converts the prediction array to a space‐separated string."""
    return " ".join(map(str, prediction))

def predict(model_dir: str, input_circuit: str) -> str:
    """
    Predicts the optimized circuit for a given input circuit.
    
    Args:
        model_dir: Path to the directory containing the trained model.
        input_circuit: The input circuit as a space‐separated string.
        
    Returns:
        The optimized circuit as a space‐separated string.
    """
    # Get the model in prediction mode
    model = get_model(mode='predict')

    # Load the trained weights using an OS-independent path
    model_file = os.path.join(model_dir, "model.pkl.gz")
    model.init_from_file(model_file, weights_only=True)

    # Tokenize the input circuit
    tokenized_input = tokenize_circuit(input_circuit)

    # Create a dummy batch (second element is a placeholder for targets)
    batch = (tokenized_input, np.zeros_like(tokenized_input))
    
    # Run the model and obtain log probabilities
    _, log_probs = model(batch)

    # Get the predicted output by taking the argmax over the last axis
    predicted_tokens = np.argmax(log_probs, axis=-1)[0]

    # Convert tokens back to string format
    return detokenize_output(predicted_tokens)

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_dir> <input_circuit>")
        sys.exit(1)

    model_dir = sys.argv[1]
    input_circuit = sys.argv[2]
    optimized_circuit = predict(model_dir, input_circuit)
    print("Optimized circuit:", optimized_circuit)

if __name__ == "__main__":
    main()
