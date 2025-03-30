import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from model_training.nn.utils import path_utils
from model_training.nn.utils import data_utils

class TestUtils(unittest.TestCase):

    def test_create_save_paths(self):
        """
        Test that create_save_paths returns valid log and model paths containing the topic string.
        """
        logdir = "test_logs"
        modeldir = "test_models"
        topic = "test_topic"
        log_path, model_path = path_utils.create_save_paths(logdir, modeldir, topic)
        self.assertIn(topic, log_path)
        self.assertIn(topic, model_path)
        self.assertTrue(isinstance(log_path, str))
        self.assertTrue(isinstance(model_path, str))
        self.assertTrue(log_path.startswith("test_logs"))
        self.assertTrue(model_path.startswith("test_models"))

    def test_dataframe_to_dataset(self):
        """
        Test that dataframe_to_dataset returns a tf.data.Dataset with the correct structure.
        """
        # Create a dummy DataFrame
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'statevector_00000': [7, 8, 9]}
        df = pd.DataFrame(data)
        batch_size = 2

        # Create the dataset
        dataset = data_utils.dataframe_to_dataset(df, batch_size)

        # Verify that the dataset is a tf.data.Dataset
        self.assertIsInstance(dataset, tf.data.Dataset)

        # Verify the structure of the dataset
        for features, labels in dataset.take(1):
            self.assertEqual(features['feature1'].shape, (batch_size,))
            self.assertEqual(labels.shape, (batch_size,))