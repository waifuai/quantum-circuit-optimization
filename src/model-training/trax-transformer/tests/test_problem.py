import os
import tempfile
import unittest
import numpy as np
from src.trainer.problem import get_data_pipelines, get_model

class TestProblem(unittest.TestCase):
    def setUp(self):
        # Create temporary input and output files with sample data
        self.input_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.output_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.input_file.write("1 2 3\n4 5 6\n")
        self.output_file.write("7 8 9\n10 11 12\n")
        self.input_file.close()
        self.output_file.close()

    def tearDown(self):
        os.unlink(self.input_file.name)
        os.unlink(self.output_file.name)

    def test_get_data_pipelines(self):
        batch_size = 2
        data_pipeline = get_data_pipelines(self.input_file.name, self.output_file.name, batch_size)
        batch = next(data_pipeline)
        # Check shapes and types
        self.assertEqual(batch['inputs'].shape[0], batch_size)
        self.assertEqual(batch['targets'].shape[0], batch_size)
        self.assertEqual(batch['inputs'].dtype.name, 'int64')
        self.assertEqual(batch['targets'].dtype.name, 'int64')

    def test_get_model(self):
        model = get_model()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
