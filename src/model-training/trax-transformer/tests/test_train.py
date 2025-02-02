import os
import tempfile
import unittest
from src.train import train_model

class TestTrain(unittest.TestCase):
    def setUp(self):
        # Create temporary input and output files with sample data
        self.input_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.output_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.model_dir = tempfile.mkdtemp()

        self.input_file.write("1 2 3\n4 5 6\n")
        self.output_file.write("7 8 9\n10 11 12\n")
        self.input_file.close()
        self.output_file.close()

    def tearDown(self):
        os.unlink(self.input_file.name)
        os.unlink(self.output_file.name)
        for filename in os.listdir(self.model_dir):
            os.remove(os.path.join(self.model_dir, filename))
        os.rmdir(self.model_dir)

    def test_train_model(self):
        # Run a short training loop for testing purposes
        train_model(self.input_file.name, self.output_file.name, self.model_dir,
                    batch_size=2, n_steps=10)
        # Check if the model file was created (this depends on how the Loop saves checkpoints)
        model_file = os.path.join(self.model_dir, "model.pkl.gz")
        self.assertTrue(os.path.exists(model_file), "Model file not found in the model directory.")

if __name__ == '__main__':
    unittest.main()
