import unittest
from unittest.mock import patch, MagicMock
from model_training.trax.src import quantum_cli
from model_training.trax.src import data
from model_training.trax.src import model


class TestQuantumCLI(unittest.TestCase):
    @patch('model_training.trax.src.quantum_cli.preprocess_data')
    def test_prep_command(self, mock_preprocess_data):
        test_args = ['prep', '--input_file', 'input.txt', '--input_processed_file',
                    'input_p.txt', '--output_processed_file', 'output_p.txt']
        with patch('sys.argv', ['quantum_cli.py'] + test_args):
            quantum_cli.main()
            self.assertTrue(mock_preprocess_data.called)

    @patch('src.model-training.trax.src.quantum_cli.train_model')
    @patch('src.model-training.trax.src.data.create_data_pipeline')
    def test_train_command(self, mock_create_data_pipeline, mock_train_model):
        mock_create_data_pipeline.return_value = (MagicMock(), 10)  # Mock pipeline and vocab size
        test_args = ['train', '--input_file', 'input_p.txt', '--output_file', 'output_p.txt',
                    '--model_dir', 'model_dir', '--batch_size', '32', '--n_steps', '100', '--data_dir', 'data_test']
        with patch('sys.argv', ['quantum_cli.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                quantum_cli.main()
                self.assertTrue(mock_create_data_pipeline.called)
                self.assertTrue(mock_train_model.called)

    @patch('src.model-training.trax.src.quantum_cli.predict')
    @patch('src.model-training.trax.src.data.create_data_pipeline')
    def test_predict_command(self, mock_create_data_pipeline, mock_predict):
        mock_create_data_pipeline.return_value = (MagicMock(), 15)  # Mock return values
        test_args = ['predict', '--model_dir', 'model_dir', '--input_circuit', '1 2 3', '--data_dir', 'data_test']
        with patch('sys.argv', ['quantum_cli.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                quantum_cli.main()
                self.assertTrue(mock_create_data_pipeline.called)
                self.assertTrue(mock_predict.called)

    @patch('src.model-training.trax.src.data.create_data_pipeline')
    def test_create_data_pipeline(self, mock_create_data_pipeline):
        mock_create_data_pipeline.return_value = (MagicMock(), 15)
        data.create_data_pipeline("input.txt", "output.txt", 2)
        self.assertTrue(mock_create_data_pipeline.called)

    @patch('src.model-training.trax.src.model.transformer_model')
    def test_get_model(self, mock_transformer_model):
        mock_transformer_model.return_value = "mocked_model"  # Mock model instance
        model_instance = quantum_cli.get_model(vocab_size=20)
        self.assertEqual(model_instance, "mocked_model")

    @patch('trax.layers.combinators.Serial')
    def test_transformer_model(self, mock_serial):
        model.transformer_model(vocab_size=10)
        self.assertTrue(mock_serial.called)


if __name__ == '__main__':
    unittest.main()