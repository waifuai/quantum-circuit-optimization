import unittest
import numpy as np
import tensorflow as tf
import cirq

from src.model-training.nn.utils import model_utils
from src.model-training.nn.utils import circuit_utils

class TestDnnCirqIntegration(unittest.TestCase):
    def test_dnn_to_cirq_circuit(self):
        """
        Tests the integration between a DNN model and Cirq circuit creation.
        """
        # Define model parameters
        input_dim = 10
        num_params = 25

        # Create a small, untrained DNN model
        model = model_utils.create_dnn_model(input_dim, num_params)

        # Generate random input data
        input_data = np.random.rand(1, input_dim).astype(np.float32)

        # Use the DNN to predict circuit parameters
        predicted_params = model(input_data)

        # Create a Cirq circuit from the predicted parameters
        circuit = circuit_utils.create_circuit(predicted_params.numpy().flatten())

        # Assert that the created circuit is a valid cirq.Circuit object
        self.assertIsInstance(circuit, cirq.Circuit)

        # Check that the circuit has qubits
        self.assertTrue(len(list(circuit.all_qubits())) > 0)

        # Check that the circuit has operations
        self.assertTrue(len(list(circuit.all_operations())) > 0)

if __name__ == '__main__':
    unittest.main()