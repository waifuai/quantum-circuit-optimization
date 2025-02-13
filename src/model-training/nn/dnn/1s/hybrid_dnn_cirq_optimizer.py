import cirq
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.circuit_utils import create_circuit, simulate_circuit, calculate_fidelity, calculate_fidelity_loss
from utils.model_utils import create_dnn_model, DNNModel
from src import config
from src.model-training.nn.utils.data_utils import load_and_preprocess_data

# Configuration parameters
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
CSV_FILE_PATH = config.CSV_FILE_PATH
NUM_QUBITS = config.NUM_QUBITS
NUM_PARAMS = config.NUM_PARAMS
CIRCUIT_TYPE = config.CIRCUIT_TYPE

print("Using Cirq", cirq.__version__)
print("Using TensorFlow", tf.__version__)

X_train, X_test, y_train, y_test, input_dim = load_and_preprocess_data(CSV_FILE_PATH, target_type='single')

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
