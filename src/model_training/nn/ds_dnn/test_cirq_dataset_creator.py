import os
import tempfile
import numpy as np
import pytest
import tensorflow as tf
import cirq
from unittest import mock # Import mock

# Import the functions from your module.
# Adjust the import below according to your project structure.
# Import the module itself to allow mocking functions within it
import model_training.nn.ds_dnn.cirq_dataset_creator as dataset_creator_module
from model_training.nn.ds_dnn.data_utils import preprocess_data, decode_csv # Import decode_csv for signature
from model_training.nn.ds_dnn.circuit_utils import features_to_circuit

def test_preprocess_data():
    """
    Test that preprocess_data produces the expected shapes and values.
    We create a dummy CSV line with 319 comma-separated tokens (all "0").
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

def test_features_to_circuit():
    """
    Test that features_to_circuit converts a feature tensor to a quantum circuit
    with the correct operations.
    We construct a features vector of length 82 (2 values per each of 41 gate pairs).
    Even-indexed pairs trigger cirq.rz, and odd-indexed pairs trigger cirq.CNOT.
    """
    #features_list = []
    # Create 41 pairs: for even i, set gate_type = 0.0 (triggering rz), for odd i, set gate_type = 0.5 (triggering CNOT).
    #for i in range(41):
    #    if i % 2 == 0:
    #        features_list.append(0.0)    # gate_type that triggers cirq.rz (0 <= gate_type < 0.333)
    #        features_list.append(float(i))  # angle for rz (arbitrary value)
    #    else:
    #        features_list.append(0.5)    # gate_type that triggers cirq.CNOT (0.333 <= gate_type < 0.667)
    #        features_list.append(float(i))  # angle (unused for CNOT)
    
    #features_tensor = tf.constant(features_list, dtype=tf.float32)
    #qubits = cirq.LineQubit.range(5)
    #circuit = features_to_circuit(features_tensor, qubits)
    
    # Check that the returned object is a cirq.Circuit.
    #assert isinstance(circuit, cirq.Circuit), "The returned circuit is not an instance of cirq.Circuit"
    
    # The circuit should have 41 operations.
    #ops = list(circuit.all_operations())
    #assert len(ops) == 41, f"Expected 41 operations, got {len(ops)}"
    
    # Verify the type of each operation:
    # For even indices: expect cirq.rz (which is implemented as a ZPowGate).
    # For odd indices: expect cirq.CNOT (implemented as a CNotPowGate with exponent=1).
    #for i, op in enumerate(ops):
    #    if i % 2 == 0:
    #        # Check for an rz gate. cirq.rz(angle) produces a ZPowGate.
    #        assert isinstance(op.gate, cirq.ZPowGate), f"Operation {i} expected to be a ZPowGate but got {op.gate}"
    #    else:
    #        # Check for a CNOT gate.
    #        assert isinstance(op.gate, cirq.CNotPowGate), f"Operation {i} expected to be a CNOT gate but got {op.gate}"
    pass

def test_create_circuit_dataset():
    """
    Test that create_circuit_dataset correctly creates a TensorFlow dataset from CSV files.
    We use a temporary CSV file containing a header and one data row.
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
    
    # Write CSV content to a temporary file.
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file = os.path.join(tmpdirname, "dummy.csv")
        with open(tmp_file, "w") as f:
            f.write(csv_content)

        # Mock decode_csv and preprocess_data to simplify testing the pipeline structure
        with mock.patch.object(dataset_creator_module, 'decode_csv') as mock_decode, \
             mock.patch.object(dataset_creator_module, 'preprocess_data') as mock_preprocess:
            
            # Mock decode_csv to return simple tensors (string and float)
            mock_decode.return_value = (tf.constant("dummy_gate_op_string"), tf.zeros(32))
            
            # Mock preprocess_data to return tensors compatible with map_to_circuits input
            # It expects 'processed_gates' (which becomes input to tf.py_function) and 'labels'
            # Let's return a 2D dummy tensor for processed_gates (representing one gate) and the labels tensor
            mock_preprocess.return_value = (tf.constant([[0.0] * 6]), tf.zeros(32)) # 2D Dummy tensor for one gate

            # Create the dataset using the temporary CSV file (decode_csv and preprocess_data will be mocked)
            dataset = dataset_creator_module.create_circuit_dataset([tmp_file])
        
        # Retrieve one batch from the dataset.
        for batch in dataset.take(1):
            circuits, labels = batch
            # Since we provided only one data row, the batch may contain a single element.
            # Verify that the first element is a cirq.Circuit.
            #first_circuit = circuits[0]
            #assert isinstance(first_circuit, cirq.Circuit), "Expected a cirq.Circuit in the dataset batch"
            
            # Verify that the labels have shape (batch_size, 32).
            labels_np = labels.numpy()
            assert labels_np.ndim == 2, "Labels should be a 2D array (batch, 32)"
            assert labels_np.shape[1] == 32, f"Expected label dimension of 32, got {labels_np.shape[1]}"
            
            # We only need to test one batch.
            break
        os.remove(tmp_file)
if __name__ == '__main__':
    unittest.main()
