import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from dnn.32s import legacy_dnn_regression_32s_resume
from dnn.32s import utils_legacy
from dnn.32s import data_utils_legacy_resume

class TestLegacyDnnRegression32sResume(unittest.TestCase):

    def test_create_save_paths(self):
        logdir = "test_logs"
        modeldir = "test_models"
        topic = "test_topic"
        log_path, model_path = utils_legacy.create_save_paths(logdir, modeldir, topic)
        self.assertIn(topic, log_path)
        self.assertIn(topic, model_path)

    def test_preprocess_data(self):
        # Create a dummy CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            tmp.write("col1,col2,statevector_00000\n1,2,3\n4,5,6")
            tmp_path = tmp.name

        X, y, target_cols = data_utils_legacy_resume.preprocess_data(tmp_path, 5)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2, 1))
        self.assertEqual(len(target_cols), 1)
        os.remove(tmp_path)

    def test_load_or_build_model(self):
        input_dim = 2
        model = legacy_dnn_regression_32s_resume.load_or_build_model(input_dim)
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, 32))

if __name__ == '__main__':
    unittest.main()