import glob
import os
import re

import cirq
import tensorflow as tf

from data_generation.qc.circuit_generation import GateOperationData
import config
from model_training.nn.ds_dnn.data_utils import decode_csv, preprocess_data, parse_gate_operation
from model_training.nn.ds_dnn.circuit_utils import features_to_circuit

# Configuration parameters
BUFFER_SIZE = config.BUFFER_SIZE
BATCH_SIZE = config.BATCH_SIZE
VALIDATION_SPLIT = config.VALIDATION_SPLIT
DATA_AUGMENTATION = config.DATA_AUGMENTATION
DIR_PATH = config.DATA_DIR

# Data directory and file paths
file_paths = sorted(glob.glob(os.path.join(DIR_PATH, "*.csv")))

n_train_files = int((1 - VALIDATION_SPLIT) * len(file_paths))
train_file_paths = file_paths[:n_train_files]
val_file_paths = file_paths[n_train_files:]

# print(f"Training files ({n_train_files}): {train_file_paths[0]} -> {train_file_paths[-1]}")
# print(f"Validation files ({len(val_file_paths)}): {val_file_paths[0]} -> {val_file_paths[-1]}")

def create_circuit_dataset(file_paths):
    """Creates a TensorFlow dataset of circuits from CSV files."""
    # Ensure file paths are strings
    file_paths_str = [str(fp) for fp in file_paths]
    
    # Handle empty file list to avoid dtype issues
    if not file_paths_str:
        # Define the expected structure for an empty dataset
        # Output signature: (circuit_string, labels_tensor)
        output_signature = (tf.TensorSpec(shape=(), dtype=tf.string),
                            tf.TensorSpec(shape=(32,), dtype=tf.float32)) # Assuming labels are float32 of shape (32,)
        return tf.data.Dataset.from_generator(lambda: iter([]), output_signature=output_signature)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths_str)
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

    qubits = cirq.LineQubit.range(config.NUM_QUBITS)
    # Modify map_to_circuits to accept the 3 tensors from preprocess_data
    def map_to_circuits(gate_types, numeric_params, labels):
        # features_to_circuit expects a list of tuples/lists, not tensors directly.
        # We need to adapt the input to features_to_circuit or adapt features_to_circuit.
        # For now, let's assume features_to_circuit needs adaptation or we pass combined info.
        # Let's pass the numeric_params tensor to py_function.
        # Note: features_to_circuit might need adjustment to handle this tensor input.
        circuit = tf.py_function(
            # Pass numeric_params tensor to the lambda
            # Assuming features_to_circuit expects the numeric parameters (control, target, angles)
            lambda np_params: features_to_circuit(np_params.numpy(), qubits), # Assuming features_to_circuit can handle this
            inp=[numeric_params], # Pass the relevant tensor(s) needed by features_to_circuit
            Tout=tf.string
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
