import os
import tempfile
import pytest # Import pytest
from unittest.mock import Mock # Import Mock
import numpy as np
import pandas as pd
import tensorflow as tf # Still needed for dtypes if used in dummy data

# -------------------------------
# Tests for cirq_circuit_optimizer.py
# -------------------------------
from model_training.nn.dcn import cirq_circuit_optimizer
# Import the specific functions to be mocked if they are not directly in cirq_circuit_optimizer
# Assuming calculate_fidelity and create_circuit are imported by cirq_circuit_optimizer
# from model_training.nn.utils.circuit_utils import create_circuit # Example if needed

# Create dummy circuit and helper functions for testing
class DummyCircuit:
    pass

# These dummy functions can remain as helpers for the tests
def dummy_create_circuit_func(params, num_qubits): # Match expected signature
    """Dummy function to replace create_circuit."""
    return DummyCircuit()

def dummy_calculate_fidelity_func(circuit, target_state):
    """Dummy function to replace calculate_fidelity."""
    # Return a constant fidelity value (for testing)
    return 0.5

# Refactored tests using pytest and monkeypatch

def test_optimize_circuit_returns_array(monkeypatch):
    """
    Test that optimize_circuit returns a numpy array of parameters
    with the expected length (25 in this case).
    """
    # Patch the functions within the cirq_circuit_optimizer module
    monkeypatch.setattr(cirq_circuit_optimizer, 'calculate_fidelity', dummy_calculate_fidelity_func)
    # Assuming create_circuit is imported into cirq_circuit_optimizer from utils
    # If it's defined locally, patch it directly:
    # monkeypatch.setattr(cirq_circuit_optimizer, 'create_circuit', dummy_create_circuit_func)
    # If it's imported like 'from model_training.nn.utils.circuit_utils import create_circuit':
    monkeypatch.setattr("model_training.nn.utils.circuit_utils.create_circuit", dummy_create_circuit_func)


    # Create minimal dummy training data.
    # Note: The structure of train_features needs to match what optimize_circuit expects.
    # The original script expects a dictionary where values are numpy arrays.
    dummy_features = {
        'dummy_feature1': np.array([0.0, 0.0], dtype='float32'),
        'dummy_feature2': np.array([0.0, 0.0], dtype='float32')
    }
    # Target state needs to be compatible with calculate_fidelity (e.g., numpy array)
    # The original script used train[target].values, which is likely a 2D array.
    dummy_target = np.array([[0.0+0.0j], [0.0+0.0j]]) # Example complex target state array

    optimized_params = cirq_circuit_optimizer.optimize_circuit(dummy_features, dummy_target)

    assert isinstance(optimized_params, np.ndarray)
    assert optimized_params.shape == (25,) # Default number of params

def test_loss_function_calls(monkeypatch):
    """
    Test that calculate_fidelity and create_circuit are called within the optimization loop.
    """
    # Use mocks to track calls
    mock_fidelity = Mock(side_effect=dummy_calculate_fidelity_func) # Use Mock from unittest.mock
    mock_create = Mock(side_effect=dummy_create_circuit_func) # Use Mock from unittest.mock

    # Patch the functions
    monkeypatch.setattr(cirq_circuit_optimizer, 'calculate_fidelity', mock_fidelity)
    # Patch create_circuit within the cirq_circuit_optimizer module's namespace
    monkeypatch.setattr(cirq_circuit_optimizer, "create_circuit", mock_create)

    dummy_features = {
'dummy_feature1': np.array([0.0, 0.0], dtype='float32'),
        'dummy_feature2': np.array([0.0, 0.0], dtype='float32')
    }
    dummy_target = np.array([[0.0+0.0j], [0.0+0.0j]]) # Example complex target state array

    # Call the function under test
    # The optimizer (scipy.minimize) will call the loss function multiple times.
    _ = cirq_circuit_optimizer.optimize_circuit(dummy_features, dummy_target)

    # Check that the mocked functions were called at least once by the optimizer.
    assert mock_fidelity.called
    assert mock_create.called

# Removed tests for legacy_cpu_dcn.py
# Removed if __name__ == '__main__': block
