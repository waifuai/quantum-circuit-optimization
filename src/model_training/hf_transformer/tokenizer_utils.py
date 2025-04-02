import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
import argparse
from pathlib import Path

# --- Configuration ---
# Adjust VOCAB_SIZE based on expected vocabulary complexity.
# If the actual unique tokens are fewer, the tokenizer will use that smaller size.
VOCAB_SIZE = 5000 # Example: Set a reasonable upper limit
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

def build_tokenizer(data_dir: Path, tokenizer_save_dir: Path):
    """
    Builds and saves a custom WordLevel tokenizer from text files.

    Args:
        data_dir: Path to the directory containing input.txt and output.txt.
        tokenizer_save_dir: Path to the directory where the tokenizer will be saved.
    """
    input_file = data_dir / "input.txt"
    output_file = data_dir / "output.txt"
    vocab_file_path = tokenizer_save_dir / "vocab.json" # Internal state file

    if not input_file.exists() or not output_file.exists():
        raise FileNotFoundError(
            f"Ensure 'input.txt' and 'output.txt' exist in '{data_dir}'"
        )

    data_files = [str(input_file), str(output_file)]

    # --- Build Vocabulary from Files ---

    # 1. Initialize a tokenizer model (WordLevel splits on whitespace)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    # 2. Set up a pre-tokenizer (Whitespace splits by spaces)
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Initialize a trainer
    trainer = WordLevelTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    # 4. Train the tokenizer on your data files
    print(f"Training tokenizer on: {data_files}")
    tokenizer.train(files=data_files, trainer=trainer)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer training complete. Actual vocabulary size: {actual_vocab_size}")

    # 5. Save the trained tokenizer's state (vocabulary, rules etc.)
    tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(vocab_file_path))
    print(f"Tokenizer vocabulary and rules saved to {vocab_file_path}")

    # --- Create Hugging Face Compatible Tokenizer ---

    # 6. Load the trained tokenizer state into PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(vocab_file_path),
        bos_token="[SOS]", # Start of Sequence
        eos_token="[EOS]", # End of Sequence
        unk_token="[UNK]", # Unknown token
        pad_token="[PAD]", # Padding token
    )

    # 7. Save the Hugging Face tokenizer (config + vocab)
    hf_tokenizer.save_pretrained(str(tokenizer_save_dir))
    print(f"Hugging Face compatible tokenizer saved to {tokenizer_save_dir}")

    # --- Test the Tokenizer ---
    print("\n--- Tokenizer Test ---")
    text = "1 2 3 4 5" # Example text similar to circuit data
    encoded = hf_tokenizer(text)
    print(f"Encoding for '{text}': {encoded}")
    decoded = hf_tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")

    # Check special token IDs
    print("\nSpecial Token IDs:")
    print(f"PAD: {hf_tokenizer.pad_token_id}")
    print(f"UNK: {hf_tokenizer.unk_token_id}")
    print(f"SOS: {hf_tokenizer.bos_token_id}")
    print(f"EOS: {hf_tokenizer.eos_token_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Custom Tokenizer")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing input.txt and output.txt",
    )
    parser.add_argument(
        "--tokenizer_save_dir",
        type=Path,
        default=Path("./tokenizer"),
        help="Directory to save the trained tokenizer",
    )
    args = parser.parse_args()

    build_tokenizer(args.data_dir, args.tokenizer_save_dir)