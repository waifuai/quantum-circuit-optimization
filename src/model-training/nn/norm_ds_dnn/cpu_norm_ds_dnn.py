#!/usr/bin/env python3
import os
import glob
import datetime
import tensorflow as tf

import config # Corrected import
from model-training.nn.norm_ds_dnn.utils import get_file_paths, split_file_paths, create_callbacks # Corrected path
from model-training.nn.utils.path_utils import setup_logging_and_model_dirs # Corrected path

# -----------------------------------------------------------------------------
# Configuration Constants (using centralized config)
# -----------------------------------------------------------------------------
# These seem to be general data/training settings used by this script
BUFFER_SIZE = config.BUFFER_SIZE
BATCH_SIZE = config.NORM_DS_DNN_BATCH_SIZE # Use specific batch size
EPOCHS = config.NORM_DS_DNN_EPOCHS # Use specific epochs
VALIDATION_SPLIT = config.VALIDATION_SPLIT

# Data location settings
DATA_DIR = config.DATA_DIR
CSV_PATTERN = config.CSV_PATTERN

# Logging/Model settings
LOG_DIR = config.LOG_DIR
MODEL_DIR = config.MODEL_DIR
# MODEL_LOAD_PATH = config.NORM_DS_DNN_MODEL_LOAD_PATH # Uncomment and define in config.py if needed
# For now, comment out the direct assignment as it's not defined in the central config yet
# MODEL_LOAD_PATH = config.MODEL_LOAD_PATH # This line will cause an error if MODEL_LOAD_PATH is not in config.py


# -----------------------------------------------------------------------------
# Data Preparation Functions
# -----------------------------------------------------------------------------
def labeler(example: tf.Tensor) -> tuple:
    """Parse a CSV line into features and labels.

    Assumes that:
      - The features are located starting at index 32 with a length of 287.
      - The labels are the first 32 columns.

    Args:
        example: A line of CSV text.

    Returns:
        A tuple (features, label) as tensors.
    """
    values = tf.strings.to_number(tf.strings.split(example, ","))
    # Extract features from the tensor
    features = tf.slice(values, [32], [287])
    # Extract labels from the tensor
    label = tf.slice(values, [0], [32])
    return features, label


def create_dataset(file_paths: list) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a list of CSV file paths.

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
    # Interleave the lines of the files into the dataset
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Skip the header line
    dataset = dataset.skip(1)
    # Map the labeler function to each element of the dataset
    dataset = dataset.map(labeler, num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle the dataset
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    # Batch the dataset
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
