#!/usr/bin/env python3
import os
import glob
import datetime
import tensorflow as tf

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 2**15
EPOCHS = 10
VALIDATION_SPLIT = 0.1

DATA_DIR = "data"
CSV_PATTERN = "*.csv"
LOG_DIR = "logs/fit"
MODEL_DIR = "models"
MODEL_LOAD_PATH = os.path.join(MODEL_DIR, "qc_dnn_8m_32s")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_file_paths() -> list:
    """Retrieve and sort CSV file paths from the data directory."""
    pattern = os.path.join(DATA_DIR, CSV_PATTERN)
    return sorted(glob.glob(pattern))


def split_file_paths(file_paths: list) -> tuple:
    """Split file paths into training and validation sets."""
    n_train = int((1 - VALIDATION_SPLIT) * len(file_paths))
    return file_paths[:n_train], file_paths[n_train:]


def setup_logging_and_model_dirs() -> tuple:
    """
    Create directories for TensorBoard logs and model checkpoints.

    Returns:
        A tuple containing the log path and model save path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_subdir = os.path.join(timestamp, "cpu_norm_ds_dnn")
    log_path = os.path.join(LOG_DIR, save_subdir)
    model_path = os.path.join(MODEL_DIR, save_subdir)
    for path in (log_path, model_path):
        os.makedirs(path, exist_ok=True)
    return log_path, model_path


def create_callbacks(log_path: str, model_path: str) -> list:
    """Initialize TensorBoard and ModelCheckpoint callbacks."""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_path, histogram_freq=0
    )
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=False,
        mode="auto"
    )
    return [tensorboard_callback, model_callback]


# -----------------------------------------------------------------------------
# Data Preparation Functions
# -----------------------------------------------------------------------------
def labeler(example: tf.Tensor) -> tuple:
    """
    Parse a CSV line into features and labels.

    Assumes that:
      - The features are located starting at index 32 with a length of 287.
      - The labels are the first 32 columns.

    Args:
        example: A line of CSV text.

    Returns:
        A tuple (features, label) as tensors.
    """
    values = tf.strings.to_number(tf.strings.split(example, ","))
    features = tf.slice(values, [32], [287])
    label = tf.slice(values, [0], [32])
    return features, label


def create_dataset(file_paths: list) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from a list of CSV file paths.

    The dataset:
      - Reads from the CSV files (skipping headers).
      - Parses each line using the labeler function.
      - Shuffles and batches the data.

    Args:
        file_paths: List of CSV file paths.

    Returns:
        A batched tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.skip(1)  # Skip header line
    dataset = dataset.map(labeler, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


# -----------------------------------------------------------------------------
# Main Training Routine
# -----------------------------------------------------------------------------
def main():
    # Retrieve and split file paths
    file_paths = get_file_paths()
    train_files, val_files = split_file_paths(file_paths)

    # Set up logging and model saving directories
    log_path, model_path = setup_logging_and_model_dirs()
    callbacks = create_callbacks(log_path, model_path)

    # Prepare the training and validation datasets
    train_dataset = create_dataset(train_files)
    val_dataset = create_dataset(val_files)

    # Load the pre-existing model
    model = tf.keras.models.load_model(MODEL_LOAD_PATH)

    # Train the model
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_dataset,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
