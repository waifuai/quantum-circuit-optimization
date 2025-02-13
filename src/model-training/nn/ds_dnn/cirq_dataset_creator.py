import glob
import os
import re

import cirq
import tensorflow as tf

from qc.circuit_generation import GateOperationData
from src import config
from src.model-training.nn.ds_dnn.data_utils import decode_csv, preprocess_data, parse_gate_operation
from src.model-training.nn.ds_dnn.circuit_utils import features_to_circuit

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

print(f"Training files ({n_train_files}): {train_file_paths[0]} -> {train_file_paths[-1]}")
print(f"Validation files ({len(val_file_paths)}): {val_file_paths[0]} -> {val_file_paths[-1]}")

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

    qubits = cirq.LineQubit.range(config.NUM_QUBITS)
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
