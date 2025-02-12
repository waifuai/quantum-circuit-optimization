import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from dnn.1s import legacy_dnn_regression
from dnn.1s import utils_legacy

class TestLegacyDnnRegression(unittest.TestCase):

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

        X, y = legacy_dnn_regression.load_and_preprocess_data(tmp_path)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2, 1))
        os.remove(tmp_path)

    def test_build_dnn_model(self):
        input_dim = 2
        model = legacy_dnn_regression.build_dnn_model(input_dim)
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, 1))

if __name__ == '__main__':
    unittest.main()