import os
import tempfile # Keep for comparison or remove if fully replaced by tmp_path
import numpy as np
import pytest
import tensorflow as tf
import cirq
# from unittest import mock # Replaced by monkeypatch

# Import the functions from your module.
# Adjust the import below according to your project structure.
# Import the module itself to allow mocking functions within it
import model_training.nn.ds_dnn.cirq_dataset_creator as dataset_creator_module
from model_training.nn.ds_dnn.data_utils import preprocess_data, decode_csv # Import decode_csv for signature
from model_training.nn.ds_dnn.circuit_utils import features_to_circuit

def test_preprocess_data():
    """
    Test that preprocess_data produces the expected shapes and values.
    """
    # Create a list of 41 dummy gate operation strings
    # Format: "GateType|Control|Target|Angle1|Angle2|Angle3"
    dummy_gate_ops_strings = [
        f"U3Gate|{i%5}|{(i+1)%5}|{np.random.rand()}|{np.random.rand()}|{np.random.rand()}"
        if i % 2 == 0 else
        f"CnotGate|{i%5}|{(i+1)%5}|0|0|0"
        for i in range(41) # Assuming 41 gate operations
    ]

    # Convert strings to TensorFlow constants
    gate_ops_list = [tf.constant(s) for s in dummy_gate_ops_strings]
    dummy_labels = tf.zeros(32) # Provide dummy labels

    # Call preprocess_data
    processed_gates, labels = preprocess_data(gate_ops_list, dummy_labels)

    # Basic checks: Ensure the function runs without error and returns expected types/structure
    assert isinstance(processed_gates, list), "Expected processed_gates to be a list"
    assert len(processed_gates) == 41, f"Expected 41 processed gates, got {len(processed_gates)}"
    assert isinstance(labels, tf.Tensor), "Expected labels to be a TensorFlow Tensor"
    assert labels.shape == (32,), f"Expected labels shape (32,), got {labels.shape}"

    # Check the structure of the first processed gate tuple
    if processed_gates:
        first_gate = processed_gates[0]
        assert isinstance(first_gate, tuple), "Each element in processed_gates should be a tuple"
        assert len(first_gate) == 6, "Each gate tuple should have 6 elements"

# Commented out test_features_to_circuit as it was incomplete and potentially outdated
# def test_features_to_circuit():
#     """
#     Test that features_to_circuit converts a feature tensor to a quantum circuit
#     with the correct operations.
#     """
#     pass # Keep as pass or remove entirely

def test_create_circuit_dataset(tmp_path, monkeypatch):
    """
    Test that create_circuit_dataset correctly creates a TensorFlow dataset from CSV files.
    Uses pytest tmp_path fixture and monkeypatch for mocking.
    """
    # Create dummy CSV content.
    num_labels = 32
    num_gate_ops = 41
    num_tokens = num_labels + num_gate_ops
    header = ",".join([f"col{i}" for i in range(num_tokens)])
    # Create dummy data: 32 floats + 41 gate strings (quoted)
    dummy_labels_str = ["0.0"] * num_labels
    # Wrap gate strings in double quotes for CSV parsing
    dummy_gate_ops_str = [
        f'"U3Gate|{i%5}|{(i+1)%5}|{np.random.rand():.2f}|{np.random.rand():.2f}|{np.random.rand():.2f}"'
        for i in range(num_gate_ops)
    ]
    data_line = ",".join(dummy_labels_str + dummy_gate_ops_str)
    csv_content = header + "\n" + data_line + "\n"

    # Write CSV content to a temporary file using tmp_path fixture
    tmp_file = tmp_path / "dummy.csv"
    tmp_file.write_text(csv_content)

    # --- Mocking using monkeypatch ---
    # Mock decode_csv to return simple tensors
    def mock_decode_csv_func(example):
        # Return structure expected by preprocess_data: (gate_operations_list, labels_tensor)
        # Here, gate_operations is a list of strings (tf.constant strings)
        return ([tf.constant("dummy_gate_op_string")] * num_gate_ops, tf.zeros(num_labels))

    # Mock preprocess_data to return tensors compatible with map_to_circuits input
    def mock_preprocess_data_func(gate_operations, labels):
        # Return structure: (processed_gates_list_of_tuples, labels_tensor)
        # processed_gates should be a list of tuples/lists suitable for features_to_circuit
        # Let's return a list containing one dummy gate tuple
        dummy_gate_tuple = ("U3Gate", 0.0, 1.0, 0.1, 0.2, 0.3) # Example tuple
        return ([dummy_gate_tuple] * num_gate_ops, labels) # Return list of tuples

    # Mock features_to_circuit within the py_function call context if possible,
    # or mock the py_function itself. Mocking py_function is simpler here.
    def mock_py_function(func, inp, Tout):
        # Simulate the py_function call returning a dummy circuit string
        # The actual func (features_to_circuit) won't be called.
        return tf.constant("dummy_circuit_repr_string")

    monkeypatch.setattr(dataset_creator_module, 'decode_csv', mock_decode_csv_func)
    monkeypatch.setattr(dataset_creator_module, 'preprocess_data', mock_preprocess_data_func)
    monkeypatch.setattr(tf, 'py_function', mock_py_function) # Mock tf.py_function directly

    # Create the dataset using the temporary CSV file (decode/preprocess/py_func are mocked)
    dataset = dataset_creator_module.create_circuit_dataset([str(tmp_file)])

    # Retrieve one batch from the dataset.
    batch_count = 0
    for batch in dataset.take(1):
        batch_count += 1
        circuits, labels = batch # circuits will contain the mocked string from py_function

        # Verify the mocked circuit representation (string)
        assert isinstance(circuits, tf.Tensor)
        # Check the first element in the batch
        assert circuits[0].numpy().decode('utf-8') == "dummy_circuit_repr_string"

        # Verify that the labels have shape (batch_size, 32).
        labels_np = labels.numpy()
        assert labels_np.ndim == 2, "Labels should be a 2D array (batch, 32)"
        assert labels_np.shape[1] == 32, f"Expected label dimension of 32, got {labels_np.shape[1]}"
        # Check batch size (should be 1 as we only had 1 data row)
        assert labels_np.shape[0] == 1

    assert batch_count > 0 # Ensure the loop ran

# Removed the if __name__ == '__main__': block
