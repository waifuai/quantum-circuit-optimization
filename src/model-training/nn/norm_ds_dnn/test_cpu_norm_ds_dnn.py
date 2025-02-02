import os
import tempfile
import shutil
import unittest
from unittest import mock
import datetime
import tensorflow as tf
import numpy as np

# Import the functions and constants from your module
import cpu_norm_ds_dnn as dnn

class TestCpuNormDsDnn(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to isolate file-system operations.
        self.test_dir = tempfile.mkdtemp()
        # Override DATA_DIR, LOG_DIR, and MODEL_DIR in the module to point to temporary locations.
        self.orig_data_dir = dnn.DATA_DIR
        self.orig_log_dir = dnn.LOG_DIR
        self.orig_model_dir = dnn.MODEL_DIR

        dnn.DATA_DIR = os.path.join(self.test_dir, "data")
        dnn.LOG_DIR = os.path.join(self.test_dir, "logs", "fit")
        dnn.MODEL_DIR = os.path.join(self.test_dir, "models")

        os.makedirs(dnn.DATA_DIR, exist_ok=True)
    
    def tearDown(self):
        # Restore original paths
        dnn.DATA_DIR = self.orig_data_dir
        dnn.LOG_DIR = self.orig_log_dir
        dnn.MODEL_DIR = self.orig_model_dir
        # Remove the temporary directory and all its contents.
        shutil.rmtree(self.test_dir)
    
    def test_get_file_paths(self):
        # Create several dummy CSV files in the DATA_DIR.
        filenames = ["a.csv", "b.csv", "c.csv"]
        for fname in filenames:
            with open(os.path.join(dnn.DATA_DIR, fname), "w") as f:
                f.write("col1,col2\n1,2\n")
        # The function should return the sorted list of CSV file paths.
        paths = dnn.get_file_paths()
        expected = sorted([os.path.join(dnn.DATA_DIR, fname) for fname in filenames])
        self.assertEqual(paths, expected)
    
    def test_split_file_paths(self):
        # Prepare a list of file paths (simulate 10 file names).
        file_paths = [f"file_{i}.csv" for i in range(10)]
        train, val = dnn.split_file_paths(file_paths)
        # With VALIDATION_SPLIT = 0.1, expect 9 training and 1 validation file.
        self.assertEqual(len(train), 9)
        self.assertEqual(len(val), 1)
        self.assertEqual(train + val, file_paths)
    
    def test_setup_logging_and_model_dirs(self):
        # Call the function and check if directories are created.
        log_path, model_path = dnn.setup_logging_and_model_dirs()
        self.assertTrue(os.path.isdir(log_path))
        self.assertTrue(os.path.isdir(model_path))
        # Check that the paths contain the expected subdirectory structure.
        self.assertIn("cpu_norm_ds_dnn", log_path)
        self.assertIn("cpu_norm_ds_dnn", model_path)
    
    def test_create_callbacks(self):
        # Provide dummy log and model paths.
        dummy_log_path = os.path.join(self.test_dir, "dummy_log")
        dummy_model_path = os.path.join(self.test_dir, "dummy_model")
        os.makedirs(dummy_log_path, exist_ok=True)
        os.makedirs(dummy_model_path, exist_ok=True)
        callbacks = dnn.create_callbacks(dummy_log_path, dummy_model_path)
        # Check that two callbacks are returned.
        self.assertEqual(len(callbacks), 2)
        # Check that one callback is TensorBoard and the other is ModelCheckpoint.
        types = [type(cb).__name__ for cb in callbacks]
        self.assertIn("TensorBoard", types)
        self.assertIn("ModelCheckpoint", types)
    
    def test_labeler(self):
        # Create a CSV line with 319 comma-separated numbers.
        num_columns = 32 + 287  # 319
        values = list(range(num_columns))
        csv_line = ",".join(str(v) for v in values)
        # Convert to tf.Tensor
        csv_tensor = tf.constant(csv_line)
        features, label = dnn.labeler(csv_tensor)
        # Evaluate tensors (for TF 2.x eager mode, this is not necessary but we can convert to numpy)
        features_np = features.numpy()
        label_np = label.numpy()
        self.assertEqual(features_np.shape, (287,))
        self.assertEqual(label_np.shape, (32,))
        # Verify that the slices are as expected.
        np.testing.assert_array_equal(label_np, np.array(values[0:32], dtype=np.float32))
        np.testing.assert_array_equal(features_np, np.array(values[32:], dtype=np.float32))
    
    def test_create_dataset(self):
        # Create a dummy CSV file with a header and two data lines.
        csv_content = "header1,header2,header3,header4\n"  # dummy header
        # Create a line with 319 columns (dummy numbers)
        num_columns = 32 + 287
        line1 = ",".join(str(i) for i in range(num_columns))
        line2 = ",".join(str(i + 1) for i in range(num_columns))
        csv_content += line1 + "\n" + line2 + "\n"
        
        dummy_csv_path = os.path.join(self.test_dir, "dummy.csv")
        with open(dummy_csv_path, "w") as f:
            f.write(csv_content)
        
        # Create dataset using the dummy CSV file.
        dataset = dnn.create_dataset([dummy_csv_path])
        # Get one batch from the dataset.
        for batch in dataset.take(1):
            features, labels = batch
            # Since we have 2 lines and our batch size is large by default,
            # the batch dimension should be 2.
            self.assertEqual(features.shape[0], 2)
            self.assertEqual(labels.shape[0], 2)
            self.assertEqual(features.shape[1], 287)
            self.assertEqual(labels.shape[1], 32)
    
    def test_main(self):
        # To test main without actually training a model, we mock:
        # 1. tf.keras.models.load_model so it returns a dummy model.
        # 2. The model's fit method.
        dummy_model = mock.MagicMock()
        dummy_model.fit = mock.MagicMock()
        
        with mock.patch("cpu_norm_ds_dnn.tf.keras.models.load_model", return_value=dummy_model) as load_model_mock:
            # Also create a dummy CSV file so that get_file_paths returns a valid list.
            dummy_csv_path = os.path.join(dnn.DATA_DIR, "dummy.csv")
            with open(dummy_csv_path, "w") as f:
                # Write header and one line
                num_columns = 32 + 287
                header = ",".join("h{}".format(i) for i in range(num_columns))
                line = ",".join(str(i) for i in range(num_columns))
                f.write(header + "\n" + line + "\n")
            
            # Call main; it should complete without error.
            dnn.main()
            
            # Check that load_model was called with the expected model load path.
            load_model_mock.assert_called_with(dnn.MODEL_LOAD_PATH)
            # Check that model.fit was called.
            self.assertTrue(dummy_model.fit.called)

if __name__ == "__main__":
    unittest.main()
