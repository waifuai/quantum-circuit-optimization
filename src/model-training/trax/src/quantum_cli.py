import os
import sys
import argparse
import numpy as np
import trax
from trax import layers as tl
from trax.supervised import training
from typing import Tuple, List, Dict, Any


def preprocess_data(input_file: str, input_processed_file: str, output_processed_file: str) -> None:
    """
    Preprocesses data by removing the first and last lines from the input file.

    Args:
        input_file: Path to the original input file.
        input_processed_file: Path to save the processed input (all but last line).
        output_processed_file: Path to save the processed output (all but first line).
    """
    try:
        with open(input_file, 'r') as f:
            lines: List[str] = [line.lstrip() for line in f.readlines()]

        with open(input_processed_file, 'w') as f_in_proc:
            f_in_proc.writelines(lines[:-1])

        with open(output_processed_file, 'w') as f_out_proc:
            f_out_proc.writelines(lines[1:])
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)


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

    def quantum_circuit_data_generator(input_file: str, output_file: str):
        """Generates pairs of input and output quantum circuits."""
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

    vocab: set = quantum_circuit_data_generator(input_filepath, output_filepath)
    vocab_size: int = len(vocab)

    def preprocess(data_pair: Tuple[str, str]) -> Dict[str, np.ndarray]:
        """Converts input and output strings to numpy arrays."""
        inp: str
        outp: str
        inp, outp = data_pair
        return {
            'inputs': np.array(list(map(int, inp.split()))),
            'targets': np.array(list(map(int, outp.split())))
        }

    data_pipeline: trax.data.Serial = trax.data.Serial(
        lambda _: quantum_circuit_data_generator(input_filepath, output_filepath),
        preprocess,
        trax.data.Batch(batch_size=batch_size)
    )
    return data_pipeline, vocab_size


def transformer_model(vocab_size: int, d_model: int = 128, d_ff: int = 512,
                        n_heads: int = 4, n_layers: int = 2, mode: str = 'train') -> tl.Serial:
    """
    Defines the Transformer model.

    Args:
        vocab_size: Vocabulary size.
        d_model:  Depth of embedding (n_units in the attention layer).
        d_ff:  Depth of feed-forward layer.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        mode: 'train' or 'eval'.

    Returns:
        A Trax Serial model.
    """
    return tl.Serial(
        tl.Embedding(vocab_size, d_model),
        [tl.EncoderBlock(d_model, d_ff, n_heads, dropout=0.1) for _ in range(n_layers)],
        tl.Dense(vocab_size),
        tl.LogSoftmax()
    )


def get_model(vocab_size: int, mode: str = 'train') -> tl.Serial:
    """Instantiates the Transformer model."""
    return transformer_model(vocab_size, mode=mode)


def tokenize_input(circuit_str: str) -> np.ndarray:
    """Tokenizes the input circuit string."""
    return np.array([list(map(int, circuit_str.split()))])


def detokenize_prediction(prediction: np.ndarray) -> str:
    """Converts the prediction array to a string."""
    return " ".join(map(str, prediction))


def predict(model_dir: str, input_circuit: str, vocab_size: int) -> str:
    """
    Predicts the optimized circuit using a trained Transformer model.

    Args:
        model_dir: Directory containing the trained model.
        input_circuit: Input circuit as a space-separated string.
        vocab_size: Vocabulary size.

    Returns:
        The optimized circuit as a space-separated string.
    """
    model: tl.Serial = get_model(vocab_size, mode='predict')
    model_file: str = os.path.join(model_dir, "model.pkl.gz")
    try:
        model.init_from_file(model_file, weights_only=True)
    except FileNotFoundError:
        print(f"Error: Model file '{model_file}' not found.", file=sys.stderr)
        sys.exit(1)

    tokenized_input: np.ndarray = tokenize_input(input_circuit)
    predictions: np.ndarray = model(tokenized_input)
    predicted_tokens: np.ndarray = np.argmax(predictions[0], axis=-1)
    return detokenize_prediction(predicted_tokens)


def train_model(input_file: str, output_file: str, model_dir: str,
                batch_size: int, n_steps: int, vocab_size: int) -> None:
    """
    Trains the Transformer model.

    Args:
        input_file: Path to the processed input file.
        output_file: Path to the processed output file.
        model_dir: Directory to save the trained model.
        batch_size: Batch size for training.
        n_steps: Number of training steps.
        vocab_size: Vocabulary size.
    """
    os.makedirs(model_dir, exist_ok=True)
    train_pipeline: trax.data.Serial
    eval_pipeline: trax.data.Serial
    train_pipeline, _ = create_data_pipeline(input_file, output_file, batch_size)
    eval_pipeline, _ = create_data_pipeline(input_file, output_file, batch_size)
    model: tl.Serial = get_model(vocab_size, mode='train')

    train_task: training.TrainTask = training.TrainTask(
        labeled_data=train_pipeline,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=100,
    )

    loop: training.Loop = training.Loop(
        model,
        train_task,
        eval_tasks=[training.EvalTask(
            labeled_data=eval_pipeline,
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        )],
        output_dir=model_dir,
    )
    loop.run(n_steps=n_steps)


def main() -> None:
    """Main CLI entry point."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Quantum Circuit Optimization CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Prep Subcommand
    prep_parser = subparsers.add_parser("prep", help="Preprocess data")
    prep_parser.add_argument("--input_file", required=True, help="Path to the original input file")
    prep_parser.add_argument("--input_processed_file", required=True, help="Path to save processed input")
    prep_parser.add_argument("--output_processed_file", required=True, help="Path to save processed output")

    # Train Subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--input_file", required=True, help="Path to the processed input file")
    train_parser.add_argument("--output_file", required=True, help="Path to the processed output file")
    train_parser.add_argument("--model_dir", required=True, help="Directory to save the trained model")
    train_parser.add_argument("--data_dir", default="data", help="Data directory")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--n_steps", type=int, default=1000, help="Number of training steps")

    # Predict Subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict using the trained model")
    predict_parser.add_argument("--model_dir", required=True, help="Directory containing the trained model")
    predict_parser.add_argument("--input_circuit", required=True, help="Input circuit as a space-separated string")
    predict_parser.add_argument("--data_dir", default="data", help="Data directory")
    args: argparse.Namespace = parser.parse_args()

    if args.command == "prep":
        if not os.path.exists(args.input_file):
            print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
            sys.exit(1)
        preprocess_data(args.input_file, args.input_processed_file, args.output_processed_file)
        print("Data preprocessing complete.")

    elif args.command == "train":
        input_file: str = os.path.join(args.data_dir, args.input_file)
        output_file: str = os.path.join(args.data_dir, args.output_file)

        if not os.path.exists(input_file) or not os.path.exists(output_file):
            print("Error: Input or output file not found.", file=sys.stderr)
            sys.exit(1)
        _, vocab_size = create_data_pipeline(input_file, output_file, args.batch_size)

        train_model(input_file, output_file, args.model_dir, args.batch_size, args.n_steps, vocab_size)
        print("Training complete.")

    elif args.command == "predict":
        input_file: str = os.path.join(args.data_dir, "input.txt")  # dummy file needed for vocab
        output_file: str = os.path.join(args.data_dir, "output.txt")  # dummy file needed for vocab
        _, vocab_size = create_data_pipeline(input_file, output_file, batch_size=64)  # dummy batch size

        if not os.path.exists(args.model_dir):
            print("Error: Model directory not found.", file=sys.stderr)
            sys.exit(1)
        optimized_circuit: str = predict(args.model_dir, args.input_circuit, vocab_size)
        print("Optimized circuit:", optimized_circuit)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()