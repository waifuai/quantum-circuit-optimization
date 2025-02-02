import os
import tempfile
import numpy as np
import pytest
import tensorflow as tf
import cirq

# Import the functions from your module.
# Adjust the import below according to your project structure.
from cirq_dataset_creator import preprocess_data, features_to_circuit, create_circuit_dataset

def test_preprocess_data():
    """
    Test that preprocess_data produces the expected shapes and values.
    We create a dummy CSV line with 319 comma-separated tokens (all "0").
    """
    num_tokens = 319  # Must match the total length expected from adder and divisor.
    # Create a CSV line of 319 zeros.
    example_line = ",".join(["0"] * num_tokens)
    
    features, labels = preprocess_data(example_line)
    
    # Evaluate the tensors (eager mode in TF2 makes this straightforward)
    labels_np = labels.numpy()
    features_np = features.numpy()
    
    # Check that labels has shape (32,) and features has shape (287,)
    assert labels_np.shape == (32,), f"Expected labels shape (32,), got {labels_np.shape}"
    assert features_np.shape == (num_tokens - 32,), f"Expected features shape ({num_tokens - 32},), got {features_np.shape}"
    
    # Since the first 32 elements in the adder are 0.0 and divisor is 1024,
    # the processed labels should be 0.
    np.testing.assert_array_almost_equal(labels_np, np.zeros(32), decimal=5)
    
    # For features, we can check the first element.
    # At index 32, adder is from the second pattern: [1.0, 0.0] * 41.
    # And divisor at the corresponding position is the first element from the pattern [float(i+1), 1.0] (i=0 => 1.0).
    # Thus, (0 + 1.0) / 1.0 should equal 1.0.
    assert np.isclose(features_np[0], 1.0), f"Expected first feature to be 1.0, got {features_np[0]}"

def test_features_to_circuit():
    """
    Test that features_to_circuit converts a feature tensor to a quantum circuit
    with the correct operations.
    We construct a features vector of length 82 (2 values per each of 41 gate pairs).
    Even-indexed pairs trigger cirq.rz, and odd-indexed pairs trigger cirq.CNOT.
    """
    features_list = []
    # Create 41 pairs: for even i, set gate_type = 0.0 (triggering rz), for odd i, set gate_type = 0.5 (triggering CNOT).
    for i in range(41):
        if i % 2 == 0:
            features_list.append(0.0)    # gate_type that triggers cirq.rz (0 <= gate_type < 0.333)
            features_list.append(float(i))  # angle for rz (arbitrary value)
        else:
            features_list.append(0.5)    # gate_type that triggers cirq.CNOT (0.333 <= gate_type < 0.667)
            features_list.append(float(i))  # angle (unused for CNOT)
    
    features_tensor = tf.constant(features_list, dtype=tf.float32)
    qubits = cirq.LineQubit.range(5)
    circuit = features_to_circuit(features_tensor, qubits)
    
    # Check that the returned object is a cirq.Circuit.
    assert isinstance(circuit, cirq.Circuit), "The returned circuit is not an instance of cirq.Circuit"
    
    # The circuit should have 41 operations.
    ops = list(circuit.all_operations())
    assert len(ops) == 41, f"Expected 41 operations, got {len(ops)}"
    
    # Verify the type of each operation:
    # For even indices: expect cirq.rz (which is implemented as a ZPowGate).
    # For odd indices: expect cirq.CNOT (implemented as a CNotPowGate with exponent=1).
    for i, op in enumerate(ops):
        if i % 2 == 0:
            # Check for an rz gate. cirq.rz(angle) produces a ZPowGate.
            assert isinstance(op.gate, cirq.ZPowGate), f"Operation {i} expected to be a ZPowGate but got {op.gate}"
        else:
            # Check for a CNOT gate.
            assert isinstance(op.gate, cirq.CNotPowGate), f"Operation {i} expected to be a CNOT gate but got {op.gate}"

def test_create_circuit_dataset():
    """
    Test that create_circuit_dataset correctly creates a TensorFlow dataset from CSV files.
    We use a temporary CSV file containing a header and one data row.
    """
    # Create dummy CSV content.
    num_tokens = 319
    header = ",".join([f"col{i}" for i in range(num_tokens)])
    # Create one dummy data line; here, all tokens are "0"
    data_line = ",".join(["0"] * num_tokens)
    csv_content = header + "\n" + data_line + "\n"
    
    # Write CSV content to a temporary file.
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file = os.path.join(tmpdirname, "dummy.csv")
        with open(tmp_file, "w") as f:
            f.write(csv_content)
        
        # Create the dataset using the temporary CSV file.
        dataset = create_circuit_dataset([tmp_file])
        
        # Retrieve one batch from the dataset.
        for batch in dataset.take(1):
            circuits, labels = batch
            # Since we provided only one data row, the batch may contain a single element.
            # Verify that the first element is a cirq.Circuit.
            first_circuit = circuits[0]
            assert isinstance(first_circuit, cirq.Circuit), "Expected a cirq.Circuit in the dataset batch"
            
            # Verify that the labels have shape (batch_size, 32).
            labels_np = labels.numpy()
            assert labels_np.ndim == 2, "Labels should be a 2D array (batch, 32)"
            assert labels_np.shape[1] == 32, f"Expected label dimension of 32, got {labels_np.shape[1]}"
            
            # We only need to test one batch.
            break
