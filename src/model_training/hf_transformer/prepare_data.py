import argparse
from pathlib import Path
from typing import List
import sys

def prepare_seq2seq_data(input_file: Path, processed_input_file: Path, processed_output_file: Path):
    """
    Prepares data for sequence-to-sequence training by creating shifted input/output pairs.

    Reads the input file, writes all lines except the last to the processed_input_file,
    and writes all lines except the first to the processed_output_file.

    Args:
        input_file: Path to the original input file (e.g., containing all sequences).
        processed_input_file: Path to save the processed input sequences (source).
        processed_output_file: Path to save the processed output sequences (target).
    """
    try:
        print(f"Reading original data from: {input_file}")
        with open(input_file, 'r') as f:
            # Read lines and strip leading/trailing whitespace
            lines: List[str] = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 2:
            print(f"Warning: Input file '{input_file}' has less than 2 lines. Cannot create shifted pairs.", file=sys.stderr)
            # Create empty files to avoid downstream errors
            processed_input_file.parent.mkdir(parents=True, exist_ok=True)
            processed_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(processed_input_file, 'w') as f_in_proc, open(processed_output_file, 'w') as f_out_proc:
                pass # Write empty files
            return

        # Input sequences: all lines except the last one
        input_sequences = lines[:-1]
        # Output sequences: all lines except the first one
        output_sequences = lines[1:]

        # Ensure parent directories exist
        processed_input_file.parent.mkdir(parents=True, exist_ok=True)
        processed_output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing processed input sequences to: {processed_input_file}")
        with open(processed_input_file, 'w') as f_in_proc:
            f_in_proc.write("\n".join(input_sequences) + "\n") # Add trailing newline

        print(f"Writing processed output sequences to: {processed_output_file}")
        with open(processed_output_file, 'w') as f_out_proc:
            f_out_proc.write("\n".join(output_sequences) + "\n") # Add trailing newline

        print(f"Data preparation complete. Processed {len(input_sequences)} input and {len(output_sequences)} output sequences.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data preparation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Sequence-to-Sequence Data")
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the original input file containing sequences line by line.",
    )
    parser.add_argument(
        "--processed_input_file",
        type=Path,
        required=True,
        help="Path to save the processed input sequences (source).",
    )
    parser.add_argument(
        "--processed_output_file",
        type=Path,
        required=True,
        help="Path to save the processed output sequences (target).",
    )
    args = parser.parse_args()

    prepare_seq2seq_data(args.input_file, args.processed_input_file, args.processed_output_file)