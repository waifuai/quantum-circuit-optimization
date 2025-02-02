import unittest
import cirq
import numpy as np
import tensorflow as tf

# Import functions from your utils modules.
from utils import circuit_utils, model_utils

class TestCircuitUtils(unittest.TestCase):
    def test_create_circuit_default(self):
        # Use a fixed set of 25 parameters (5 layers * 5 qubits)
        params = [0.1] * 25
        circuit = circuit_utils.create_circuit(params)
        
        # Verify the returned object is a cirq.Circuit instance.
        self.assertIsInstance(circuit, cirq.Circuit)
        
        # Each of the 5 layers should include:
        # - 5 RX operations (one per qubit)
        # - 4 CNOT operations (between consecutive qubits)
        # Total operations = 5 * (5 + 4) = 45.
        num_ops = len(list(circuit.all_operations()))
        self.assertEqual(num_ops, 45)
        
        # Verify that if no qubits are provided, the default is a 5-qubit register.
        expected_qubits = list(cirq.LineQubit.range(5))
        circuit_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
        self.assertEqual(circuit_qubits, expected_qubits)

    def test_create_circuit_custom_qubits(self):
        params = [0.2] * 25
        # Create custom qubits (for example, using GridQubit)
        custom_qubits = [cirq.GridQubit(0, i) for i in range(5)]
        circuit = circuit_utils.create_circuit(params, qubits=custom_qubits)
        
        # Check that the circuit uses the custom qubits
        circuit_qubits = sorted(circuit.all_qubits(), key=lambda q: (q.row, q.col))
        self.assertEqual(circuit_qubits, custom_qubits)

    def test_calculate_loss_identity(self):
        # Create an empty circuit (which acts as an identity) on 5 qubits.
        qubits = cirq.LineQubit.range(5)
        circuit = cirq.Circuit()
        
        # Simulate the empty circuit; the final state will be the |0...0> state.
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        target_state = result.final_state_vector
        
        # The loss between the circuit output and the target state should be near zero.
        loss = circuit_utils.calculate_loss(circuit, target_state)
        self.assertAlmostEqual(loss, 0.0, places=5)

class TestModelUtils(unittest.TestCase):
    def test_create_dnn_model_output_shape(self):
        input_dim = 10
        num_params = 25
        model = model_utils.create_dnn_model(input_dim, num_params)
        
        # Verify the model's input and output shapes.
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, num_params))

    def test_dnn_forward_pass(self):
        input_dim = 10
        num_params = 25
        model = model_utils.create_dnn_model(input_dim, num_params)
        
        # Create a dummy input batch of size 2.
        sample_input = np.random.rand(2, input_dim).astype(np.float32)
        output = model(sample_input)
        
        # The output should have shape (2, num_params)
        self.assertEqual(output.shape, (2, num_params))

if __name__ == '__main__':
    unittest.main()
