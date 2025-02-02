# ds_dnn/cirq_dataset_creator.py
import glob
import os
import re

import cirq
import tensorflow as tf

# Configuration parameters
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 2**15  # ~32k examples per batch
VALIDATION_SPLIT = 0.1

# Data directory and file paths
DIR_PATH = "shards/"  # Update this with your local directory
file_paths = sorted(glob.glob(os.path.join(DIR_PATH, "*.csv")))

n_train_files = int((1 - VALIDATION_SPLIT) * len(file_paths))
train_file_paths = file_paths[:n_train_files]
val_file_paths = file_paths[n_train_files:]

print(f"Training files ({n_train_files}): {train_file_paths[0]} -> {train_file_paths[-1]}")
print(f"Validation files ({len(val_file_paths)}): {val_file_paths[0]} -> {val_file_paths[-1]}")

def preprocess_data(example):
    """Preprocess a single CSV line example."""
    adder = [0.0] * 32 + [1.0, 0.0] * 41 + [1.0] * (41 * 5)
    divisor = [1024.0] * 32 + [float(i + 1), 1.0] * 41 + [5.0] * (41 * 2) + [3.0] * (41 * 3)
    adder_tensor = tf.constant(adder, dtype=tf.float32)
    divisor_tensor = tf.constant(divisor, dtype=tf.float32)

    # Replace string tokens sequentially
    replace_dict = {
        "U3Gate": "0",
        "CnotGate": "0.333",
        "Measure": "0.667",
        "BLANK": "1",
    }
    for key, val in replace_dict.items():
        example = tf.strings.regex_replace(example, key, val)

    numeric_data = tf.strings.to_number(tf.strings.split(example, ","), out_type=tf.float32)
    processed_data = (numeric_data + adder_tensor) / divisor_tensor
    features = processed_data[32:]
    labels = processed_data[:32]
    return features, labels

def features_to_circuit(features, qubits):
    """Converts features to a parameterized quantum circuit."""
    circuit = cirq.Circuit()
    # Assume features encode 41 pairs (gate_type, angle)
    for i in range(41):
        gate_type = features[2 * i]
        angle = features[2 * i + 1]
        if 0 <= gate_type < 0.333:  # U3 analog
            circuit.append(cirq.rz(angle)(qubits[i % len(qubits)]))
        elif 0.333 <= gate_type < 0.667:  # CNOT analog
            control = i % len(qubits)
            target = (i + 1) % len(qubits)
            circuit.append(cirq.CNOT(qubits[control], qubits[target]))
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
    # Skip header and preprocess lines
    dataset = dataset.skip(1).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    qubits = cirq.LineQubit.range(5)
    def map_to_circuits(features, labels):
        circuit = tf.py_function(
            lambda f: features_to_circuit(f.numpy(), qubits),
            inp=[features],
            Tout=object  # The circuit object
        )
        return circuit, labels

    dataset = dataset.map(map_to_circuits, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False).batch(BATCH_SIZE)
    return dataset

train_circuit_dataset = create_circuit_dataset(train_file_paths)
val_circuit_dataset = create_circuit_dataset(val_file_paths)

print("Example circuit from dataset:")
for circuits, labels in train_circuit_dataset.take(1):
    print(circuits[0])
