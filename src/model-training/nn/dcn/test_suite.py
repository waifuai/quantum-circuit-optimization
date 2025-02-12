import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import tensorflow as tf

# -------------------------------
# Tests for cirq_circuit_optimizer.py
# -------------------------------
import dcn.cirq_circuit_optimizer as cirq_circuit_optimizer

# Create dummy circuit and loss functions for testing
class DummyCircuit:
    pass

def dummy_create_circuit(params):
    # Return a dummy circuit object (could be more complex if needed)
    return DummyCircuit()

def dummy_calculate_fidelity(circuit, target_state):
    # Return a constant fidelity value (for testing)
    return 0.5

class TestCirqCircuitOptimizer(unittest.TestCase):

    @patch('dcn.cirq_circuit_optimizer.calculate_fidelity', side_effect=dummy_calculate_fidelity)
    @patch('dcn.cirq_circuit_optimizer.create_circuit', side_effect=dummy_create_circuit)
    def test_optimize_circuit_returns_array(self, mock_create, mock_fidelity):
        """
        Test that optimize_circuit returns a numpy array of parameters
        with the expected length (25 in this case).
        """
        # Create minimal dummy training data.
        dummy_features = {
            'dummy_feature1': np.array([0.0, 0.0], dtype='float32'),
            'dummy_feature2': np.array([0.0, 0.0], dtype='float32')
        }
        dummy_target = np.array([[0.0], [0.0]])
        optimized_params = cirq_circuit_optimizer.optimize_circuit(dummy_features, dummy_target)
        self.assertIsInstance(optimized_params, np.ndarray)
        self.assertEqual(optimized_params.shape, (25,))

    @patch('dcn.cirq_circuit_optimizer.calculate_fidelity', side_effect=dummy_calculate_fidelity)
    @patch('dcn.cirq_circuit_optimizer.create_circuit', side_effect=dummy_create_circuit)
    def test_loss_function_calls(self, mock_create, mock_fidelity):
        """
        Test that calculate_fidelity and create_circuit are called within the optimization loop.
        """
        dummy_features = {
            'dummy_feature1': np.array([0.0, 0.0], dtype='float32'),
            'dummy_feature2': np.array([0.0, 0.0], dtype='float32')
        }
        dummy_target = np.array([[0.0], [0.0]])
        _ = cirq_circuit_optimizer.optimize_circuit(dummy_features, dummy_target)
        # Check that calculate_fidelity is called at least once.
        self.assertTrue(mock_fidelity.called)
        mock_create.assert_called()

# -------------------------------
# Tests for legacy_cpu_dcn.py
# -------------------------------
import dcn.legacy_cpu_dcn as legacy_cpu_dcn
from deepctr.inputs import DenseFeat
from tensorflow.keras.models import Model

class TestLegacyCpuDCN(unittest.TestCase):

    def test_load_and_prepare_data(self):
        """
        Create a temporary CSV file with the expected columns and verify that load_and_prepare_data
        returns inputs of the correct types and dimensions.
        """
        # Reconstruct the list of dense features as used in the module.
        dense_features = [
            f"gate_{str(i).zfill(2)}_{suffix}"
            for i in range(41)
            for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
        ]
        target = ['statevector_00000']
        # Create a DataFrame with a few rows (ensure number of rows is not less than batch_size).
        num_rows = 4
        data_dict = {col: np.random.rand(num_rows) for col in dense_features}
        data_dict[target[0]] = np.random.rand(num_rows)
        df = pd.DataFrame(data_dict)

        # Write DataFrame to a temporary CSV file.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            # For testing, use a small batch_size (e.g., 2) to force truncation.
            batch_size = 2
            train_ds, test_ds, feat_names = legacy_cpu_dcn.load_and_prepare_data(tmp_path, batch_size)
            # Verify that inputs are tf.data.Dataset objects
            self.assertIsInstance(train_ds, tf.data.Dataset)
            self.assertIsInstance(test_ds, tf.data.Dataset)

            # Verify that the datasets contain the correct features
            for features, labels in train_ds.take(1):
                self.assertEqual(set(features.keys()), set(dense_features))
                self.assertEqual(features[dense_features[0]].shape, (batch_size,))
                self.assertEqual(labels.shape, (batch_size,))

        finally:
            os.remove(tmp_path)

    def test_build_dcn_model(self):
        """
        Test that build_dcn_model returns a model that is compiled with the expected optimizer and loss.
        """
        # Create a dummy feature column list.
        dummy_feats = ["dummy_feature"]
        num_circuit_params = 25
        model = legacy_cpu_dcn.build_dcn_model(dummy_feats, dummy_feats, num_circuit_params)
        # Check if the optimizer and loss are set as expected.
        self.assertEqual(model.optimizer._name, "adam")
        self.assertTrue(isinstance(model, Model))
        self.assertEqual(len(model.outputs), 2)  # Check for two outputs
        self.assertEqual(model.loss['dcn'], 'mse')
        self.assertEqual(model.get_layer('circuit_params').output_shape, (None, num_circuit_params))

if __name__ == '__main__':
    unittest.main()
