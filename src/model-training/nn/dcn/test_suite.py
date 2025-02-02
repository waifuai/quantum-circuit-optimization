import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

# -------------------------------
# Tests for cirq_circuit_optimizer.py
# -------------------------------
import cirq_circuit_optimizer

# Create dummy circuit and loss functions for testing
class DummyCircuit:
    pass

def dummy_create_circuit(params):
    # Return a dummy circuit object (could be more complex if needed)
    return DummyCircuit()

def dummy_calculate_loss(circuit, target_state):
    # Return a constant loss value (for testing, a constant loss is acceptable)
    return 1.0

class TestCirqCircuitOptimizer(unittest.TestCase):

    @patch('cirq_circuit_optimizer.calculate_loss', side_effect=dummy_calculate_loss)
    @patch('cirq_circuit_optimizer.create_circuit', side_effect=dummy_create_circuit)
    def test_optimize_circuit_returns_array(self, mock_create, mock_loss):
        """
        Test that optimize_circuit returns a numpy array of parameters
        with the expected length (25 in this case).
        """
        # Create minimal dummy training data.
        # Note: The current implementation zips over train_features.values() and train_target,
        # so only as many iterations occur as the length of train_target.
        dummy_features = {
            'dummy_feature1': np.array([0.0, 0.0], dtype='float32'),
            'dummy_feature2': np.array([0.0, 0.0], dtype='float32')
        }
        dummy_target = np.array([[0.0], [0.0]])
        optimized_params = cirq_circuit_optimizer.optimize_circuit(dummy_features, dummy_target)
        self.assertIsInstance(optimized_params, np.ndarray)
        self.assertEqual(optimized_params.shape, (25,))
    
    @patch('cirq_circuit_optimizer.calculate_loss', side_effect=dummy_calculate_loss)
    @patch('cirq_circuit_optimizer.create_circuit', side_effect=dummy_create_circuit)
    def test_loss_function_calls(self, mock_create, mock_loss):
        """
        Test that calculate_loss and create_circuit are called within the optimization loop.
        For a minimal input (2 samples, 3 epochs), we expect the functions to be invoked multiple times.
        """
        dummy_features = {
            'dummy_feature1': np.array([0.0, 0.0], dtype='float32'),
            'dummy_feature2': np.array([0.0, 0.0], dtype='float32')
        }
        dummy_target = np.array([[0.0], [0.0]])
        _ = cirq_circuit_optimizer.optimize_circuit(dummy_features, dummy_target)
        # Check that calculate_loss is called at least once per sample per epoch.
        # The optimization loop calls calculate_loss once initially and twice per parameter for gradient computation.
        expected_minimum_calls = 2 * cirq_circuit_optimizer.EPOCHS  # at least one call per epoch per sample (with 2 samples)
        self.assertGreaterEqual(mock_loss.call_count, expected_minimum_calls)

# -------------------------------
# Tests for legacy_cpu_dcn.py
# -------------------------------
import legacy_cpu_dcn
from deepctr.inputs import DenseFeat, get_feature_names

class TestLegacyCpuDCN(unittest.TestCase):

    def test_create_save_paths(self):
        """
        Test that create_save_paths returns valid log and model paths containing the topic string.
        """
        logdir = "test_logs"
        modeldir = "test_models"
        topic = "test_topic"
        log_path, model_path = legacy_cpu_dcn.create_save_paths(logdir, modeldir, topic)
        self.assertIn(topic, log_path)
        self.assertIn(topic, model_path)
        self.assertTrue(isinstance(log_path, str))
        self.assertTrue(isinstance(model_path, str))

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
            train_input, test_input, y_train, y_test, feat_columns = legacy_cpu_dcn.load_and_prepare_data(tmp_path, batch_size)
            # Verify that inputs are dictionaries with float32 arrays.
            self.assertTrue(isinstance(train_input, dict))
            for key, arr in train_input.items():
                self.assertEqual(arr.dtype, np.float32)
            # Verify that target arrays have correct shape.
            self.assertTrue(isinstance(y_train, np.ndarray))
            # Check that the number of rows in train and test inputs are divisible by the batch size.
            for key in train_input:
                self.assertEqual(len(train_input[key]) % batch_size, 0)
        finally:
            os.remove(tmp_path)

    def test_build_dcn_model(self):
        """
        Test that build_dcn_model returns a model that is compiled with the expected optimizer and loss.
        """
        # Create a dummy feature column list.
        dummy_feats = [DenseFeat("dummy_feature", 1)]
        model = legacy_cpu_dcn.build_dcn_model(dummy_feats, dummy_feats)
        # Check if the optimizer and loss are set as expected.
        self.assertEqual(model.optimizer._name, "adam")
        self.assertEqual(model.loss, "mse")

if __name__ == '__main__':
    unittest.main()
