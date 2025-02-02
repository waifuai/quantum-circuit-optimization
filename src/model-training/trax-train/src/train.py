import os
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
from utils import load_data, data_generator

# Hyperparameters
BATCH_SIZE = 16
MAX_LENGTH = 256
N_LAYERS = 2
D_MODEL = 256
D_FF = 512
N_HEADS = 4
MODE = 'train'
N_TRAIN_STEPS = 1000
EVAL_STEPS = 10
LEARNING_RATE = 0.0005

DATA_DIR = "data"
MODEL_DIR = "model"
TRAIN_INPUT_FILE = "input_processed.txt"
TRAIN_OUTPUT_FILE = "output_processed.txt"

def create_transformer_model(input_vocab_size: int, output_vocab_size: int,
                             d_model: int, d_ff: int, n_layers: int,
                             n_heads: int, max_len: int, mode: str,
                             ff_activation=tl.Relu) -> tl.Serial:
    """
    Creates and returns a Transformer model.
    """
    # Encoder
    encoder_embedding = tl.Embedding(input_vocab_size, d_model)
    encoder_positional_encoder = tl.PositionalEncoding(max_len=max_len, mode=mode)
    encoder_layers = [
        tl.Serial(
            tl.LayerNorm(),
            tl.SelfAttention(d_model, n_heads=n_heads, mode=mode),
            tl.Dense(d_ff),
            ff_activation(),
            tl.LayerNorm(),
        ) for _ in range(n_layers)
    ]
    encoder = tl.Serial(
        encoder_embedding,
        encoder_positional_encoder,
        encoder_layers,
        tl.LayerNorm(),
    )

    # Decoder
    decoder_embedding = tl.Embedding(output_vocab_size, d_model)
    decoder_positional_encoder = tl.PositionalEncoding(max_len=max_len, mode=mode)
    decoder_layers = [
        tl.Serial(
            tl.LayerNorm(),
            tl.SelfAttention(d_model, n_heads=n_heads, mode=mode, causal=True),
            tl.Dense(d_ff),
            ff_activation(),
            tl.LayerNorm(),
            tl.Dense(d_model),
            tl.CrossAttention(d_model, n_heads=n_heads, mode=mode),
            tl.Dense(d_ff),
            ff_activation(),
            tl.LayerNorm(),
        ) for _ in range(n_layers)
    ]
    decoder = tl.Serial(
        decoder_embedding,
        decoder_positional_encoder,
        decoder_layers,
        tl.LayerNorm(),
    )

    # Full Transformer (encoder + decoder)
    transformer = tl.Serial(
        tl.Select([0, 0]),  # Duplicate input for encoder and decoder
        tl.Parallel(encoder, decoder),
        tl.Concatenate(),
        tl.Dense(output_vocab_size),
        tl.LogSoftmax()
    )
    return transformer

def training_loop(model, train_gen, eval_gen, steps: int, eval_steps: int, output_dir: str):
    """
    Custom training loop for the Transformer model.
    """
    optimizer = trax.optimizers.Adam(LEARNING_RATE)
    train_task = trax.supervised.training.TrainTask(
        labeled_data=train_gen,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=optimizer,
        n_steps_per_checkpoint=eval_steps
    )
    eval_task = trax.supervised.training.EvalTask(
        labeled_data=eval_gen,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )
    loop = trax.supervised.training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )
    loop.run(n_steps=steps)
    return loop

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_inputs, train_targets, vocab = load_data(DATA_DIR, TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE)
    train_size = int(0.8 * len(train_inputs))
    train_inputs_split, eval_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets_split, eval_targets = train_targets[:train_size], train_targets[train_size:]
    train_gen = data_generator(train_inputs_split, train_targets_split, vocab, BATCH_SIZE, MAX_LENGTH)
    eval_gen = data_generator(eval_inputs, eval_targets, vocab, BATCH_SIZE, MAX_LENGTH)

    input_vocab_size = len(vocab)
    output_vocab_size = len(vocab)
    model = create_transformer_model(
        input_vocab_size,
        output_vocab_size,
        D_MODEL,
        D_FF,
        N_LAYERS,
        N_HEADS,
        MAX_LENGTH,
        MODE
    )
    training_loop(model, train_gen, eval_gen, N_TRAIN_STEPS, EVAL_STEPS, MODEL_DIR)
    print(f"Training complete. Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
