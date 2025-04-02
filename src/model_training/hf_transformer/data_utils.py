from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import argparse

# Default configuration (can be overridden)
MAX_LENGTH = 256 # Maximum sequence length for padding/truncation

def load_custom_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    """Loads the custom tokenizer saved by tokenizer_utils.py."""
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    print(f"Loaded tokenizer from {tokenizer_dir} with vocab size {tokenizer.vocab_size}")
    return tokenizer

def load_processed_text_files(data_dir: Path, input_filename: str = "input_processed.txt", output_filename: str = "output_processed.txt") -> DatasetDict:
    """
    Loads the processed input and output text files into a Hugging Face DatasetDict.

    Args:
        data_dir: The directory containing the processed text files.
        input_filename: The name of the processed input file.
        output_filename: The name of the processed output file.

    Returns:
        A DatasetDict containing a 'train' split (or potentially others later).
    """
    input_file = data_dir / input_filename
    output_file = data_dir / output_filename

    if not input_file.exists() or not output_file.exists():
        raise FileNotFoundError(
            f"Ensure '{input_filename}' and '{output_filename}' exist in '{data_dir}'"
        )

    # Load using the 'text' loading script, treating each line as a sample
    # We need to manually pair them afterwards.
    try:
        with open(input_file, "r") as f_in, open(output_file, "r") as f_out:
            inputs = [line.strip() for line in f_in.readlines() if line.strip()]
            outputs = [line.strip() for line in f_out.readlines() if line.strip()]

        if len(inputs) != len(outputs):
            print(f"Warning: Mismatched number of lines in {input_filename} ({len(inputs)}) and {output_filename} ({len(outputs)}). Check data preparation.")
            # Optionally truncate to the shorter length or raise an error
            min_len = min(len(inputs), len(outputs))
            inputs = inputs[:min_len]
            outputs = outputs[:min_len]

        # Create a list of dictionaries, which Dataset.from_list expects
        data_list = [{"input_text": inp, "target_text": out} for inp, out in zip(inputs, outputs)]

        # Create the Dataset object
        dataset = Dataset.from_list(data_list)

        # Wrap it in a DatasetDict (useful for train/validation splits later)
        dataset_dict = DatasetDict({"train": dataset})
        print(f"Loaded dataset with {len(dataset)} examples.")
        return dataset_dict

    except Exception as e:
        print(f"Error loading processed text files: {e}")
        raise

def tokenize_and_prepare_data(dataset_dict: DatasetDict, tokenizer: PreTrainedTokenizerFast, max_length: int = MAX_LENGTH) -> DatasetDict:
    """
    Tokenizes the input and target text in the dataset.

    Args:
        dataset_dict: The DatasetDict loaded by load_processed_text_files.
        tokenizer: The loaded custom tokenizer.
        max_length: The maximum sequence length for padding/truncation.

    Returns:
        A DatasetDict with tokenized 'input_ids', 'attention_mask', and 'labels'.
    """

    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_length,
            truncation=True,
            padding="max_length" # Pad to max_length for consistent tensor shapes initially
                                 # DataCollator will handle dynamic padding per batch later if used
        )

        # Tokenize targets (labels)
        labels = tokenizer(
            examples["target_text"],
            max_length=max_length,
            truncation=True,
            padding="max_length" # Pad to max_length
        )

        model_inputs["labels"] = labels["input_ids"]

        # Replace padding token id in labels with -100 so it's ignored in loss calculation
        # Important: Do this *after* assigning to model_inputs['labels']
        processed_labels = []
        for label_ids in model_inputs["labels"]:
            processed_label_ids = [
                label_id if label_id != tokenizer.pad_token_id else -100
                for label_id in label_ids
            ]
            processed_labels.append(processed_label_ids)
        model_inputs["labels"] = processed_labels

        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "target_text"] # Remove original text columns
    )
    print("Tokenization complete.")
    print("Tokenized dataset features:", tokenized_datasets["train"].features)
    return tokenized_datasets


if __name__ == "__main__":
    # Example usage when run directly
    parser = argparse.ArgumentParser(description="Load and Tokenize Processed Data")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing processed input/output files.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=Path("./tokenizer"),
        help="Directory containing the saved tokenizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum sequence length for tokenization.",
    )
    args = parser.parse_args()

    try:
        # 1. Load tokenizer
        tokenizer = load_custom_tokenizer(args.tokenizer_dir)

        # 2. Load processed text data
        raw_datasets = load_processed_text_files(args.data_dir)
        print("\nRaw Dataset sample:")
        print(raw_datasets["train"][0])

        # 3. Tokenize data
        tokenized_datasets = tokenize_and_prepare_data(raw_datasets, tokenizer, args.max_length)
        print("\nTokenized Dataset sample:")
        # Print details of the first example's tokenization
        sample = tokenized_datasets["train"][0]
        print("{")
        for key, value in sample.items():
            print(f"  '{key}': {value[:20]}... (length: {len(value)})") # Print first 20 tokens
        print("}")

        # Example of decoding back
        print("\nDecoded Input Sample (first 20 tokens):")
        print(tokenizer.decode(sample['input_ids'][:20], skip_special_tokens=False))
        print("\nDecoded Label Sample (first 20 tokens, ignoring -100):")
        label_ids_no_ignore = [lid for lid in sample['labels'][:20] if lid != -100]
        print(tokenizer.decode(label_ids_no_ignore, skip_special_tokens=False))


    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)