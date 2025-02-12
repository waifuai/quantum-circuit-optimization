import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from dnn.32s import legacy_dnn_regression_32s
from dnn.32s import utils_legacy

class TestLegacyDnnRegression32s(unittest.TestCase):

    def test_create_save_paths(self):
        logdir = "test_logs"
        modeldir = "test_models"
        topic = "test_topic"
        log_path, model_path = utils_legacy.create_save_paths(logdir, modeldir, topic)
        self.assertIn(topic, log_path)
        self.assertIn(topic, model_path)

    def test_load_and_preprocess_data(self):
        # Create a dummy CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            tmp.write("col1,col2,statevector_00000\n1,2,3\n4,5,6")
            tmp_path = tmp.name

        data, dense_features = legacy_dnn_regression_32s.load_and_preprocess_data(tmp_path)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(dense_features, list)
        os.remove(tmp_path)

    def test_build_dnn_model(self):
        input_dim = 2
        output_units = 32
        model = legacy_dnn_regression_32s.build_dnn_model(input_dim, output_units)
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, output_units))

if __name__ == '__main__':
    unittest.main()