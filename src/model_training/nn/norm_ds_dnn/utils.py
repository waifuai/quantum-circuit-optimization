import os
import glob
import datetime
import tensorflow as tf

import config

def get_file_paths() -> list:
    """Retrieve and sort CSV file paths from the data directory."""
    pattern = os.path.join(config.DATA_DIR, config.CSV_PATTERN)
    return sorted(glob.glob(pattern))


def split_file_paths(file_paths: list) -> tuple:
    """Split file paths into training and validation sets."""
    n_train = int((1 - config.VALIDATION_SPLIT) * len(file_paths))
    return file_paths[:n_train], file_paths[n_train:]


def setup_logging_and_model_dirs() -> tuple:
    """
    Create directories for TensorBoard logs and model checkpoints.

    Returns:
        A tuple containing the log path and model save path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_subdir = os.path.join(timestamp, "cpu_norm_ds_dnn")
    log_path = os.path.join(config.LOG_DIR, save_subdir)
    model_path = os.path.join(config.MODEL_DIR, save_subdir)
    for path in (log_path, model_path):
        os.makedirs(path, exist_ok=True)
    return log_path, model_path


def create_callbacks(log_path: str, model_path: str) -> list:
    """Initialize TensorBoard and ModelCheckpoint callbacks."""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_path, histogram_freq=0
    )
    # Construct the full filepath for the model checkpoint
    model_filepath = os.path.join(model_path, "model.keras")
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        model_filepath, # Use the full filepath
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=False,
        mode="auto"
    )
    return [tensorboard_callback, model_callback]