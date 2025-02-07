import glob
import os
import re

import cirq
import tensorflow as tf

from qc.circuit_generation import GateOperationData

# Configuration parameters
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 2**15  # ~32k examples per batch
VALIDATION_SPLIT = 0.1
DATA_AUGMENTATION = True # Enable/Disable data augmentation

# Data directory and file paths
DIR_PATH = "shards/"  # Update this with your local directory
file_paths = sorted(glob.glob(os.path.join(DIR_PATH, "*.csv")))

n_train_files = int((1 - VALIDATION_SPLIT) * len(file_paths))
train_file_paths = file_paths[:n_train_files]
val_file_paths = file_paths[n_train_files:]

print(f"Training files ({n_train_files}): {train_file_paths[0]} -> {train_file_paths[-1]}")
print(f"Validation files ({len(val_file_paths)}): {val_file_paths[0]} -> {val_file_paths[-1]}")

def decode_csv(example):
    """Decodes a CSV line into features and labels."""
    # Define the structure of the CSV record
    record_defaults = [tf.float32] * 32 + [tf.string] * 41  # 32 labels + 41 gate operation strings
    
    # Decode the CSV record
    decoded_record = tf.io.decode_csv(example, record_defaults=record_defaults)
    
    # Separate features and labels
    labels = tf.stack(decoded_record[:32])
    gate_operations = decoded_record[32:]
    return gate_operations, labels

def parse_gate_operation(gate_string):
    """Parses a gate operation string into a GateOperationData object."""
    parts = tf.strings.split(gate_string, sep='|')
    gate_type = parts[0]
    control = tf.strings.to_number(parts[1], out_type=tf.float32)
    target = tf.strings.to_number(parts[2], out_type=tf.float32)
    angle1 = tf.strings.to_number(parts[3], out_type=tf.float32)
    angle2 = tf.strings.to_number(parts[4], out_type=tf.float32)
    angle3 = tf.strings.to_number(parts[5], out_type=tf.float32)
    
    return gate_type, control, target, angle1, angle2, angle3

def preprocess_data(gate_operations, labels):
    """Preprocess gate operations and labels with optional data augmentation."""
    
    def augment_gate(gate_type, control, target, angle1, angle2, angle3):
        """Applies data augmentation to a single gate operation."""
        if DATA_AUGMENTATION:
            # Add small random rotations to gate angles
            angle1 += tf.random.uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)
            angle2 += tf.random.uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)
            angle3 += tf.random.uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)

            # Swap control and target qubits in CNOT gates
            swap_prob = 0.1  # Probability of swapping control and target
            if tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32) < swap_prob:
                control, target = target, control
        return gate_type, control, target, angle1, angle2, angle3

    processed_gates = []
    for gate_string in gate_operations:
        gate_type, control, target, angle1, angle2, angle3 = parse_gate_operation(gate_string)
        gate_type, control, target, angle1, angle2, angle3 = augment_gate(gate_type, control, target, angle1, angle2, angle3)
        processed_gates.append((gate_type, control, target, angle1, angle2, angle3))

    return processed_gates, labels

def features_to_circuit(processed_gates, qubits):
    """Converts processed gate operations to a parameterized quantum circuit."""
    circuit = cirq.Circuit()
    # Assume features encode 41 gates
    for i, (gate_type, control, target, angle1, angle2, angle3) in enumerate(processed_gates):
        if gate_type == "U3Gate":  # U3 analog
            circuit.append(cirq.rz(angle1)(qubits[i % len(qubits)]))
        elif gate_type == "CnotGate":  # CNOT analog
            circuit.append(cirq.CNOT(qubits[int(control) % len(qubits)], qubits[int(target) % len(qubits)]))
        # Add further gate mappings as needed
    return circuit

def create_circuit_dataset(file_paths):
    """Creates a TensorFlow dataset of circuits from CSV files."""
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(
        tf.data.TextLineDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Skip header
    dataset = dataset.skip(1)
    # Decode CSV lines
    dataset = dataset.map(decode_csv, num_parallel_calls=tf.data.AUTOTUNE)
    # Preprocess features and labels
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    qubits = cirq.LineQubit.range(5)
    def map_to_circuits(processed_gates, labels):
        circuit = tf.py_function(
            lambda g: features_to_circuit(g.numpy(), qubits),
            inp=[processed_gates],
            Tout=tf.string  # The circuit object
        )
        return circuit, labels

    dataset = dataset.map(map_to_circuits, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_circuit_dataset = create_circuit_dataset(train_file_paths)
val_circuit_dataset = create_circuit_dataset(val_file_paths)

print("Example circuit from dataset:")
for circuits, labels in train_circuit_dataset.take(1):
    print(circuits[0].numpy().decode('utf-8'))
