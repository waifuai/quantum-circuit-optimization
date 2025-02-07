import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import src.quantum_cli as quantum_cli  # Import the CLI module


class TestQuantumCLI(unittest.TestCase):
    @patch('src.quantum_cli.preprocess_data')
    def test_prep_command(self, mock_preprocess_data):
        test_args = ['prep', '--input_file', 'input.txt', '--input_processed_file',
                    'input_p.txt', '--output_processed_file', 'output_p.txt']
        with patch('sys.argv', ['quantum_cli.py'] + test_args):
            quantum_cli.main()
            mock_preprocess_data.assert_called_once_with('input.txt', 'input_p.txt', 'output_p.txt')

    @patch('src.quantum_cli.preprocess_data')
    def test_prep_command_file_not_found(self, mock_preprocess_data):
        test_args = ['prep', '--input_file', 'nonexistent_file.txt', '--input_processed_file',
                    'input_p.txt', '--output_processed_file', 'output_p.txt']

        with patch('sys.argv', ['quantum_cli.py'] + test_args), \
            patch('sys.exit') as mock_exit, \
            patch('builtins.open', side_effect=FileNotFoundError):  # Mock open to raise FileNotFoundError

            quantum_cli.main()
            mock_exit.assert_called_once_with(1)
            mock_preprocess_data.assert_not_called()

    @patch('src.quantum_cli.train_model')
    @patch('src.quantum_cli.create_data_pipeline')
    def test_train_command(self, mock_create_data_pipeline, mock_train_model):

        mock_create_data_pipeline.return_value = (MagicMock(), 10) # Mock pipeline and vocab size
        test_args = ['train', '--input_file', 'input_p.txt', '--output_file', 'output_p.txt',
                    '--model_dir', 'model_dir', '--batch_size', '32', '--n_steps', '100', '--data_dir', 'data_test']
        with patch('sys.argv', ['quantum_cli.py'] + test_args):
            with patch('os.path.exists', return_value = True):
                quantum_cli.main()
                mock_create_data_pipeline.assert_called_once_with('data_test/input_p.txt', 'data_test/output_p.txt', 32)
                mock_train_model.assert_called_once_with('data_test/input_p.txt', 'data_test/output_p.txt', 'model_dir', 32, 100, 10)

    @patch('src.quantum_cli.predict')
    @patch('src.quantum_cli.create_data_pipeline')
    def test_predict_command(self, mock_create_data_pipeline, mock_predict):
        mock_create_data_pipeline.return_value = (MagicMock(), 15)  # Mock return values

        test_args = ['predict', '--model_dir', 'model_dir', '--input_circuit', '1 2 3', '--data_dir', 'data_test']
        with patch('sys.argv', ['quantum_cli.py'] + test_args):
            with patch('os.path.exists', return_value = True):
                quantum_cli.main()
                mock_create_data_pipeline.assert_called()
                mock_predict.assert_called_once_with('model_dir', '1 2 3', 15)

    @patch('src.quantum_cli.create_data_pipeline')
    def test_create_data_pipeline(self, mock_create_data_pipeline):
        # Mock file content
        mocked_data = "1 2 3\n4 5 6\n"
        mocked_data2 = "7 8 9\n10 11 12\n"

        # Use mock_open to simulate file reading
        with patch("builtins.open", mock_open(read_data=mocked_data)) as mocked_input, \
            patch("builtins.open", mock_open(read_data=mocked_data2)) as mocked_output:
            pipeline, vocab_size = quantum_cli.create_data_pipeline("input.txt", "output.txt", 2)
            self.assertEqual(vocab_size, 12)
            # check pipeline
            batch = next(pipeline)
            self.assertEqual(batch['inputs'].shape[0], 2)



    @patch('src.quantum_cli.transformer_model')
    def test_get_model(self, mock_transformer_model):
        mock_transformer_model.return_value = "mocked_model"  # Mock model instance
        model = quantum_cli.get_model(vocab_size=20)
        mock_transformer_model.assert_called_once_with(20, mode='train')  # Check args
        self.assertEqual(model, "mocked_model")

    def test_tokenize_input(self):
        input_str = "1 2 3"
        expected_tokens = np.array([[1, 2, 3]])
        tokens = quantum_cli.tokenize_input(input_str)
        np.testing.assert_array_equal(tokens, expected_tokens)

    def test_detokenize_prediction(self):
        prediction = np.array([1, 2, 3])
        expected_str = "1 2 3"
        detokenized = quantum_cli.detokenize_prediction(prediction)
        self.assertEqual(detokenized, expected_str)

    @patch('trax.layers.combinators.Serial')
    def test_transformer_model(self, mock_serial):
        quantum_cli.transformer_model(vocab_size=10)
        self.assertTrue(mock_serial.called)


if __name__ == '__main__':
    unittest.main()