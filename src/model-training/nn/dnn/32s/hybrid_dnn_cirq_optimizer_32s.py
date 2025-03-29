import cirq
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from model-training.nn.utils.circuit_utils import create_circuit, simulate_circuit, calculate_fidelity, calculate_fidelity_loss # Corrected path
from model-training.nn.utils.model_utils import create_dnn_model, DNNModel # Corrected path
import config # Corrected import
from model-training.nn.utils.data_utils import load_and_preprocess_data # Corrected path

# Configuration parameters (using centralized config)
BATCH_SIZE = config.DNN_32S_BATCH_SIZE
EPOCHS = config.DNN_32S_EPOCHS
CSV_FILE_PATH = config.DNN_32S_CSV_FILE_PATH
NUM_QUBITS = config.DNN_NUM_QUBITS # Use general DNN value
NUM_PARAMS = config.DNN_NUM_PARAMS # Use general DNN value
CIRCUIT_TYPE = config.DNN_CIRCUIT_TYPE # Use general DNN value

print("Using Cirq", cirq.__version__)
print("Using TensorFlow", tf.__version__)

X_train, X_test, y_train, y_test, input_dim = load_and_preprocess_data(CSV_FILE_PATH, target_type='multi', n_qubits=config.NUM_QUBITS)

# Convert data to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# Define model dimensions
input_dim = input_dim

dnn_model = create_dnn_model(input_dim, NUM_PARAMS)
dnn_model = DNNModel(input_dim, NUM_PARAMS)

def fidelity_loss(y_true, y_pred):
    return calculate_fidelity_loss(y_true, y_pred, NUM_QUBITS, CIRCUIT_TYPE)

# Compile the model with the custom loss function
#dnn_model.compile(optimizer='adam', loss=fidelity_loss)

# Implement the full training loop
#dnn_model.fit(train_dataset, epochs=EPOCHS, verbose=2, validation_data=test_dataset)
dnn_model.train(train_dataset, epochs=EPOCHS, validation_data=test_dataset)

# Example: Use the DNN to generate parameters and create a quantum circuit.
qubits = cirq.LineQubit.range(NUM_QUBITS)
example_input = X_test[0]
example_params = dnn_model.predict(np.expand_dims(example_input, axis=0)).flatten()
circuit = create_circuit(example_params, qubits, circuit_type=CIRCUIT_TYPE)
print("Generated Circuit:")
print(circuit)
