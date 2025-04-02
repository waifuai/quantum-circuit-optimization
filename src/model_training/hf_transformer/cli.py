import argparse
from pathlib import Path
import subprocess
import sys
import os

# --- Helper to get the script's directory ---
SCRIPT_DIR = Path(__file__).parent.resolve()

def run_script(script_name: str, args: list):
    """Runs a Python script located in the same directory as this CLI script."""
    script_path = SCRIPT_DIR / script_name
    command = [sys.executable, str(script_path)] + args
    print(f"\n--- Running: {' '.join(command)} ---")
    try:
        # Use check=True to raise CalledProcessError if the script fails
        subprocess.run(command, check=True, text=True)
        print(f"--- Finished: {script_name} ---")
    except FileNotFoundError:
        print(f"Error: Script '{script_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Script '{script_name}' failed with exit code {e.returncode}.", file=sys.stderr)
        # Optionally print stdout/stderr from the failed script
        # print(f"Stdout:\n{e.stdout}")
        # print(f"Stderr:\n{e.stderr}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An unexpected error occurred while running {script_name}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="CLI for HF Transformer Quantum Circuit Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Tokenize Command ---
    parser_tokenize = subparsers.add_parser("tokenize", help="Build the custom tokenizer.")
    parser_tokenize.add_argument(
        "--data_dir", type=Path, default=SCRIPT_DIR / "data",
        help="Directory containing input.txt and output.txt for tokenizer training."
    )
    parser_tokenize.add_argument(
        "--tokenizer_save_dir", type=Path, default=SCRIPT_DIR / "tokenizer",
        help="Directory to save the trained tokenizer."
    )

    # --- Prepare Data Command ---
    parser_prep = subparsers.add_parser("prep", help="Prepare data by creating shifted input/output files.")
    parser_prep.add_argument(
        "--input_file", type=Path, default=SCRIPT_DIR / "data/input.txt",
        help="Path to the original input file containing all sequences."
    )
    parser_prep.add_argument(
        "--processed_input_file", type=Path, default=SCRIPT_DIR / "data/input_processed.txt",
        help="Path to save the processed input sequences (source)."
    )
    parser_prep.add_argument(
        "--processed_output_file", type=Path, default=SCRIPT_DIR / "data/output_processed.txt",
        help="Path to save the processed output sequences (target)."
    )

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train the sequence-to-sequence model.")
    parser_train.add_argument(
        "--data_dir", type=Path, default=SCRIPT_DIR / "data",
        help="Directory containing processed input/output files."
    )
    parser_train.add_argument(
        "--tokenizer_dir", type=Path, default=SCRIPT_DIR / "tokenizer",
        help="Directory containing the saved custom tokenizer."
    )
    parser_train.add_argument(
        "--output_dir", type=Path, default=SCRIPT_DIR / "hf_transformer_results",
        help="Directory to save training results."
    )
    parser_train.add_argument("--max_length", type=int, default=256, help="Max sequence length.")
    parser_train.add_argument("--num_train_epochs", type=int, default=5, help="Training epochs.")
    parser_train.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size.")
    parser_train.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser_train.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser_train.add_argument("--logging_steps", type=int, default=100, help="Log every X steps.")
    parser_train.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser_train.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep.")
    parser_train.add_argument("--fp16", action='store_true', help="Enable mixed precision training.")
    parser_train.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    # Add other relevant training args here

    # --- Predict Command ---
    parser_predict = subparsers.add_parser("predict", help="Predict using the trained model.")
    parser_predict.add_argument(
        "--model_dir", type=Path, required=True,
        help="Directory containing the saved trained model."
    )
    parser_predict.add_argument(
        "--tokenizer_dir", type=Path, default=SCRIPT_DIR / "tokenizer",
        help="Directory containing the saved custom tokenizer."
    )
    parser_predict.add_argument(
        "--input_circuit", type=str, required=True,
        help="Input circuit sequence as a space-separated string."
    )
    parser_predict.add_argument("--max_length", type=int, default=256, help="Max generation length.")
    parser_predict.add_argument("--num_beams", type=int, default=4, help="Number of beams for search.")

    args = parser.parse_args()

    # --- Execute Commands ---
    if args.command == "tokenize":
        run_script("tokenizer_utils.py", [
            "--data_dir", str(args.data_dir),
            "--tokenizer_save_dir", str(args.tokenizer_save_dir)
        ])
    elif args.command == "prep":
        run_script("prepare_data.py", [
            "--input_file", str(args.input_file),
            "--processed_input_file", str(args.processed_input_file),
            "--processed_output_file", str(args.processed_output_file)
        ])
    elif args.command == "train":
        train_args_list = [
            "--data_dir", str(args.data_dir),
            "--tokenizer_dir", str(args.tokenizer_dir),
            "--output_dir", str(args.output_dir),
            "--max_length", str(args.max_length),
            "--num_train_epochs", str(args.num_train_epochs),
            "--per_device_train_batch_size", str(args.per_device_train_batch_size),
            "--learning_rate", str(args.learning_rate),
            "--weight_decay", str(args.weight_decay),
            "--logging_steps", str(args.logging_steps),
            "--save_steps", str(args.save_steps),
            "--save_total_limit", str(args.save_total_limit),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        ]
        if args.fp16:
            train_args_list.append("--fp16")
        run_script("train.py", train_args_list)
    elif args.command == "predict":
        run_script("predict.py", [
            "--model_dir", str(args.model_dir),
            "--tokenizer_dir", str(args.tokenizer_dir),
            "--input_circuit", args.input_circuit, # Pass as single string
            "--max_length", str(args.max_length),
            "--num_beams", str(args.num_beams)
        ])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()