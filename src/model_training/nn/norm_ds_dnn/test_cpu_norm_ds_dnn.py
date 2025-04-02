import os
import tempfile
import shutil
import datetime
import tensorflow as tf
import numpy as np
import pytest
from unittest import mock # Keep mock for MagicMock if needed, or replace with pytest mocks

# Import the functions and constants from your module
from model_training.nn.norm_ds_dnn import cpu_norm_ds_dnn as dnn
from model_training.nn.norm_ds_dnn import utils
import config # Import config to patch it

@pytest.fixture
def temp_env(monkeypatch):
    """Pytest fixture to set up and tear down a temporary environment."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Store original config values
        orig_data_dir = config.DATA_DIR
        orig_log_dir = config.LOG_DIR
        orig_model_dir = config.MODEL_DIR

        # Define temporary paths
        temp_data_dir = os.path.join(test_dir, "data")
        temp_log_dir = os.path.join(test_dir, "logs", "fit")
        temp_model_dir = os.path.join(test_dir, "models")

        # Create temp data dir
        os.makedirs(temp_data_dir, exist_ok=True)

        # Patch config module attributes
        monkeypatch.setattr(config, 'DATA_DIR', temp_data_dir)
        monkeypatch.setattr(config, 'LOG_DIR', temp_log_dir)
        monkeypatch.setattr(config, 'MODEL_DIR', temp_model_dir)

        # Yield the temporary directory path (contains data, logs, models subdirs)
        yield test_dir

        # Teardown: Restore original config values (monkeypatch handles this automatically for setattr)
        # No need to manually restore if using monkeypatch.setattr
        # config.DATA_DIR = orig_data_dir
        # config.LOG_DIR = orig_log_dir
        # config.MODEL_DIR = orig_model_dir
        # tempfile.TemporaryDirectory handles directory removal

def test_get_file_paths(temp_env):
    """Test retrieving file paths."""
    # temp_env fixture provides the patched config.DATA_DIR
    filenames = ["a.csv", "b.csv", "c.csv"]
    for fname in filenames:
        with open(os.path.join(config.DATA_DIR, fname), "w") as f:
            f.write("col1,col2\n1,2\n")
    # The function should return the sorted list of CSV file paths.
    paths = utils.get_file_paths()
    expected = sorted([os.path.join(config.DATA_DIR, fname) for fname in filenames])
    assert paths == expected

def test_split_file_paths():
    """Test splitting file paths for train/validation."""
    # Prepare a list of file paths (simulate 10 file names).
    file_paths = [f"file_{i}.csv" for i in range(10)]
    # Assuming config.VALIDATION_SPLIT is 0.1 (default)
    train, val = utils.split_file_paths(file_paths)
    # With VALIDATION_SPLIT = 0.1, expect 9 training and 1 validation file.
    assert len(train) == 9
    assert len(val) == 1
    assert train + val == file_paths

def test_setup_logging_and_model_dirs(temp_env):
    """Test creation of logging and model directories."""
    # temp_env fixture provides patched config.LOG_DIR and config.MODEL_DIR
    log_path, model_path = utils.setup_logging_and_model_dirs()
    assert os.path.isdir(log_path)
    assert os.path.isdir(model_path)
    # Check that the paths contain the expected subdirectory structure.
    assert "cpu_norm_ds_dnn" in log_path
    assert "cpu_norm_ds_dnn" in model_path
    # Check they are inside the temp_env directory
    assert log_path.startswith(config.LOG_DIR)
    assert model_path.startswith(config.MODEL_DIR)

def test_create_callbacks(temp_env):
    """Test creation of Keras callbacks."""
    # temp_env provides the base directories for logs/models
    dummy_log_path = os.path.join(config.LOG_DIR, "dummy_log") # Use patched config
    dummy_model_filepath = os.path.join(config.MODEL_DIR, "dummy_model", "best_model.keras") # Use patched config
    os.makedirs(os.path.dirname(dummy_model_filepath), exist_ok=True)
    os.makedirs(dummy_log_path, exist_ok=True)

    callbacks = utils.create_callbacks(dummy_log_path, dummy_model_filepath)
    # Check that two callbacks are returned.
    assert len(callbacks) == 2
    # Check that one callback is TensorBoard and the other is ModelCheckpoint.
    types = [type(cb).__name__ for cb in callbacks]
    assert "TensorBoard" in types
    assert "ModelCheckpoint" in types

def test_labeler():
    """Test the labeler function for parsing CSV lines."""
    # Create a CSV line with 319 comma-separated numbers.
    num_columns = 32 + 287  # 319
    values = list(range(num_columns))
    csv_line = ",".join(str(v) for v in values)
    # Convert to tf.Tensor
    csv_tensor = tf.constant(csv_line)
    features, label = dnn.labeler(csv_tensor)
    # Evaluate tensors
    features_np = features.numpy()
    label_np = label.numpy()
    assert features_np.shape == (287,)
    assert label_np.shape == (32,)
    # Verify that the slices are as expected.
    np.testing.assert_array_equal(label_np, np.array(values[0:32], dtype=np.float32))
    np.testing.assert_array_equal(features_np, np.array(values[32:], dtype=np.float32))

def test_create_dataset(temp_env):
    """Test creating a tf.data.Dataset from CSV."""
    # temp_env provides patched config.DATA_DIR
    # Create a dummy CSV file with a header and two data lines.
    csv_content = "header1,header2,header3,header4\n"  # dummy header
    # Create a line with 319 columns (dummy numbers)
    num_columns = 32 + 287
    line1 = ",".join(str(i) for i in range(num_columns))
    line2 = ",".join(str(i + 1) for i in range(num_columns))
    csv_content += line1 + "\n" + line2 + "\n"

    dummy_csv_path = os.path.join(config.DATA_DIR, "dummy.csv")
    with open(dummy_csv_path, "w") as f:
        f.write(csv_content)

    # Create dataset using the dummy CSV file.
    # Assuming config.BATCH_SIZE is set appropriately (e.g., >= 2 for this test)
    dataset = dnn.create_dataset([dummy_csv_path])
    # Get one batch from the dataset.
    batch_count = 0
    for batch in dataset.take(1):
        batch_count += 1
        features, labels = batch
        # Since we have 2 lines and batch size might be larger,
        # the batch dimension should be 2.
        assert features.shape[0] == 2
        assert labels.shape[0] == 2
        assert features.shape[1] == 287
        assert labels.shape[1] == 32
    assert batch_count > 0 # Ensure the loop ran

def test_main(temp_env, monkeypatch):
    """Test the main function execution flow (without actual training)."""
    # temp_env provides patched config dirs
    # Mock tf.keras.models.load_model and model.fit
    mock_load_model = mock.MagicMock(name="load_model")
    mock_model_instance = mock.MagicMock(name="model_instance")
    mock_load_model.return_value = mock_model_instance

    monkeypatch.setattr(tf.keras.models, "load_model", mock_load_model)

    # Create a dummy CSV file so that get_file_paths returns a valid list.
    dummy_csv_path = os.path.join(config.DATA_DIR, "dummy.csv")
    with open(dummy_csv_path, "w") as f:
        # Write header and one line
        num_columns = 32 + 287
        header = ",".join("h{}".format(i) for i in range(num_columns))
        line = ",".join(str(i) for i in range(num_columns))
        f.write(header + "\n" + line + "\n")

    # Call main; it should complete without error.
    dnn.main()

    # Since model loading is commented out in cpu_norm_ds_dnn.py,
    # load_model should NOT have been called.
    mock_load_model.assert_not_called()
    # Check that model.fit was also NOT called, as training is skipped.
    mock_model_instance.fit.assert_not_called()
