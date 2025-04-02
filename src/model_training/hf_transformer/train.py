import argparse
from pathlib import Path
import sys

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import DatasetDict

# Import local utilities
try:
    from data_utils import load_custom_tokenizer, load_processed_text_files, tokenize_and_prepare_data, MAX_LENGTH
    from model import create_hf_transformer_model
except ImportError:
    print("Error: Could not import local modules (data_utils, model). Ensure they are in the Python path or same directory.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train HF Transformer for Quantum Circuit Optimization")
    parser.add_argument(
        "--data_dir", type=Path, default=Path("./data"),
        help="Directory containing processed input/output files (input_processed.txt, output_processed.txt)."
    )
    parser.add_argument(
        "--tokenizer_dir", type=Path, default=Path("./tokenizer"),
        help="Directory containing the saved custom tokenizer."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./hf_transformer_results"),
        help="Directory to save training results (checkpoints, logs, final model)."
    )
    parser.add_argument(
        "--max_length", type=int, default=MAX_LENGTH,
        help="Maximum sequence length for tokenization."
    )
    # --- Training Arguments ---
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints.")
    parser.add_argument("--fp16", action='store_true', help="Enable mixed precision training (requires compatible GPU).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    # Add more training args as needed (e.g., warmup_steps, eval_strategy)

    args = parser.parse_args()

    try:
        # 1. Load Tokenizer
        print("Loading tokenizer...")
        tokenizer = load_custom_tokenizer(args.tokenizer_dir)

        # 2. Load and Tokenize Data
        print("Loading and tokenizing data...")
        raw_datasets: DatasetDict = load_processed_text_files(args.data_dir)
        # TODO: Add validation split if needed: raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        tokenized_datasets: DatasetDict = tokenize_and_prepare_data(raw_datasets, tokenizer, args.max_length)

        # Ensure train dataset exists
        if "train" not in tokenized_datasets or len(tokenized_datasets["train"]) == 0:
             print("Error: No training data found after tokenization.", file=sys.stderr)
             sys.exit(1)

        # 3. Initialize Model
        print("Initializing model...")
        model = create_hf_transformer_model(
            vocab_size=tokenizer.vocab_size,
            decoder_start_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_position_embeddings=args.max_length + 2 # Account for potential special tokens
            # Pass other model hyperparameters if needed
        )

        # 4. Define Training Arguments
        print("Defining training arguments...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=str(args.output_dir / "logs"),
            logging_steps=args.logging_steps,
            save_strategy="steps", # Save based on steps
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            evaluation_strategy="no", # Change to "epoch" or "steps" if eval_dataset is provided
            predict_with_generate=True, # Important for Seq2Seq tasks
            fp16=args.fp16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # Add other arguments like warmup_steps if desired
            report_to="tensorboard", # Log to TensorBoard
        )

        # 5. Initialize Data Collator
        print("Initializing data collator...")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding='longest' # Pad batches dynamically to the longest sequence in the batch
        )

        # 6. Initialize Trainer
        print("Initializing trainer...")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            # eval_dataset=tokenized_datasets["test"], # Add if you have validation data
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics, # Add if you want custom evaluation metrics (e.g., BLEU, ROUGE)
        )

        # 7. Train
        print("Starting training...")
        train_result = trainer.train()
        print("Training finished.")

        # 8. Save Final Model & Stats
        print("Saving final model...")
        trainer.save_model() # Saves the final model weights and config
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # Saves optimizer, scheduler, etc.
        print(f"Model and training state saved to {args.output_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure data and tokenizer directories exist and contain the necessary files.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()