from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, PreTrainedTokenizerFast
from pathlib import Path
import argparse

# Default Hyperparameters (matching Trax where applicable)
DEFAULT_HIDDEN_SIZE = 128       # d_model in Trax
DEFAULT_INTERMEDIATE_SIZE = 512 # d_ff in Trax
DEFAULT_NUM_ATTENTION_HEADS = 4 # n_heads in Trax
DEFAULT_NUM_HIDDEN_LAYERS = 2   # n_layers in Trax (applied to both encoder/decoder)
DEFAULT_MAX_POSITION_EMBEDDINGS = 258 # Corresponds to MAX_LENGTH + 2 for special tokens

def create_hf_transformer_model(
    vocab_size: int,
    decoder_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS
) -> EncoderDecoderModel:
    """
    Creates and initializes a Hugging Face EncoderDecoderModel from scratch
    with specified configurations.

    Args:
        vocab_size: The size of the vocabulary (must match the tokenizer).
        decoder_start_token_id: The ID for the decoder's start token (e.g., tokenizer.bos_token_id).
        eos_token_id: The ID for the end-of-sequence token (e.g., tokenizer.eos_token_id).
        pad_token_id: The ID for the padding token (e.g., tokenizer.pad_token_id).
        hidden_size: Dimensionality of the encoder/decoder layers and embeddings.
        intermediate_size: Dimensionality of the "intermediate" (feed-forward) layer.
        num_attention_heads: Number of attention heads for each attention layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder/decoder.
        max_position_embeddings: The maximum sequence length that this model might ever be used with.

    Returns:
        An initialized EncoderDecoderModel (with random weights).
    """
    print("Creating model configuration...")
    # --- Configure Encoder ---
    encoder_config = BertConfig(
        vocab_size=vocab_size, # Needs vocab size for embeddings
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1, # Only one sentence type needed
        pad_token_id=pad_token_id, # Important for attention mask
    )
    print(f"Encoder Config: {encoder_config}")

    # --- Configure Decoder ---
    # Must be cross-attention compatible
    decoder_config = BertConfig(
        vocab_size=vocab_size, # Needs vocab size for embeddings and output layer
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
        is_decoder=True,
        add_cross_attention=True, # Crucial for seq2seq
        pad_token_id=pad_token_id, # Important for attention mask
    )
    print(f"Decoder Config: {decoder_config}")

    # --- Create EncoderDecoder Config ---
    # Tie encoder/decoder configs together
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    # --- Set Generation Parameters ---
    config.decoder_start_token_id = decoder_start_token_id
    config.eos_token_id = eos_token_id
    config.pad_token_id = pad_token_id
    config.vocab_size = vocab_size # Ensure model's vocab size matches tokenizer

    print(f"Combined EncoderDecoder Config: {config}")

    # --- Initialize Model from Config (Random Weights) ---
    print("Initializing EncoderDecoderModel from configuration...")
    model = EncoderDecoderModel(config=config)
    print(f"Model created successfully with {model.num_parameters():,} parameters.")

    return model

if __name__ == "__main__":
    # Example usage when run directly
    # Requires a tokenizer to be built first (e.g., using tokenizer_utils.py)
    parser = argparse.ArgumentParser(description="Create HF Transformer Model")
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=Path("./tokenizer"),
        help="Directory containing the saved tokenizer.",
    )
    args = parser.parse_args()

    try:
        # Load tokenizer to get necessary IDs
        from data_utils import load_custom_tokenizer # Assuming data_utils is in the same dir or python path
        tokenizer = load_custom_tokenizer(args.tokenizer_dir)

        # Create the model
        model = create_hf_transformer_model(
            vocab_size=tokenizer.vocab_size,
            decoder_start_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Print model summary (optional)
        # Note: model.summary() requires building the model first, e.g., by passing dummy input
        # print("\nModel Summary:")
        # model.summary() # This might require providing input shapes

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure you have built and saved the tokenizer first using tokenizer_utils.py")
    except ImportError:
        print("Error: Could not import data_utils. Ensure it's in the Python path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)