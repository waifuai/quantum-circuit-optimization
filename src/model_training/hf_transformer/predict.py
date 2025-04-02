import argparse
from pathlib import Path
import sys
import torch # Often needed for device placement

from transformers import (
    PreTrainedTokenizerFast,
    EncoderDecoderModel
)

# Import local utilities
try:
    from data_utils import MAX_LENGTH # Import default max length
except ImportError:
    print("Error: Could not import local data_utils. Ensure it's in the Python path or same directory.", file=sys.stderr)
    # Define a fallback if import fails, though it's better if the import works
    MAX_LENGTH = 256
    print(f"Warning: Using fallback MAX_LENGTH={MAX_LENGTH}", file=sys.stderr)


def load_model_and_tokenizer(model_dir: Path, tokenizer_dir: Path) -> tuple[EncoderDecoderModel, PreTrainedTokenizerFast]:
    """Loads the trained model and tokenizer."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    print(f"Loading tokenizer from: {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    print(f"Loading model from: {model_dir}")
    model = EncoderDecoderModel.from_pretrained(str(model_dir))

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded onto device: {device}")

    model.eval() # Set model to evaluation mode
    return model, tokenizer

def predict_sequence(
    model: EncoderDecoderModel,
    tokenizer: PreTrainedTokenizerFast,
    input_text: str,
    max_length: int = MAX_LENGTH,
    num_beams: int = 4,
    early_stopping: bool = True
) -> str:
    """
    Generates an output sequence for the given input text using the model.

    Args:
        model: The loaded EncoderDecoderModel.
        tokenizer: The loaded PreTrainedTokenizerFast.
        input_text: The input sequence string (space-separated tokens).
        max_length: Maximum length for the generated sequence.
        num_beams: Number of beams for beam search.
        early_stopping: Whether to stop generation early in beam search.

    Returns:
        The generated output sequence string.
    """
    device = model.device # Get the device the model is on

    print(f"\nInput sequence: '{input_text}'")
    # Tokenize the input text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True, # Pad to longest in batch (only 1 here)
        truncation=True,
        max_length=max_length # Use same max_length as training? Or specific inference length?
    ).to(device) # Move input tensors to the same device as the model

    print("Generating output sequence...")
    # Generate output sequence using beam search
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length, # Max length of the *generated* sequence
        decoder_start_token_id=model.config.decoder_start_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_beams=num_beams,
        early_stopping=early_stopping,
        # Add other generation parameters if needed (e.g., temperature, top_k, top_p)
    )

    # Decode the generated sequence
    # output_sequences is on the model's device, decode expects list/numpy on CPU
    generated_ids = output_sequences[0].cpu().numpy()
    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Generated sequence: '{decoded_output}'")
    return decoded_output.strip() # Remove leading/trailing whitespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using HF Transformer for Quantum Circuit Optimization")
    parser.add_argument(
        "--model_dir", type=Path, required=True,
        help="Directory containing the saved trained model (output from train.py)."
    )
    parser.add_argument(
        "--tokenizer_dir", type=Path, default=Path("./tokenizer"),
        help="Directory containing the saved custom tokenizer."
    )
    parser.add_argument(
        "--input_circuit", type=str, required=True,
        help="Input circuit sequence as a space-separated string (e.g., '1 2 3 4')."
    )
    parser.add_argument(
        "--max_length", type=int, default=MAX_LENGTH,
        help="Maximum sequence length for generation."
    )
    parser.add_argument(
        "--num_beams", type=int, default=4,
        help="Number of beams for beam search."
    )

    args = parser.parse_args()

    try:
        # 1. Load Model and Tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_dir, args.tokenizer_dir)

        # 2. Predict
        optimized_circuit = predict_sequence(
            model,
            tokenizer,
            args.input_circuit,
            max_length=args.max_length,
            num_beams=args.num_beams
        )

        # 3. Print result
        print("\n--- Prediction Result ---")
        print(f"Input:  '{args.input_circuit}'")
        print(f"Output: '{optimized_circuit}'")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure model and tokenizer directories exist and contain the necessary files.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}", file=sys.stderr)
        sys.exit(1)