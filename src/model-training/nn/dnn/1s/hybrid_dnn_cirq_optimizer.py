import cirq
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.circuit_utils import create_circuit
from utils.model_utils import create_dnn_model

# Configuration parameters
BATCH_SIZE = 4096
EPOCHS = int(1e3)  # Reduced for practicality
CSV_FILE_PATH = "./qc8m_1s.csv"
NUM_QUBITS = 5
NUM_PARAMS = 25  # 5 layers * 5 qubits

print("Using Cirq", cirq.__version__)
print("Using TensorFlow", tf.__version__)

try:
    data = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()

# Replace categorical labels with numeric codes
data = data.replace(['U3Gate', 'CnotGate', 'Measure', 'BLANK'], [1, 2, 3, 4])

dense_features = []
for i in range(41):
    n = str(i).zfill(2)
    prefix = f"gate_{n}_"
    dense_features.extend([
        prefix + suffix 
        for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
    ])

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

target = ['statevector_00000']
X = data[dense_features].values
y = data[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# Define model dimensions
input_dim = len(dense_features)

dnn_model = create_dnn_model(input_dim, NUM_PARAMS)

def fidelity_loss(y_true, y_pred):
    """
    Custom loss function that calculates the fidelity between the output statevector
    of the quantum circuit and the target state.
    """
    # Reshape y_pred to (batch_size, num_params)
    y_pred = tf.reshape(y_pred, (-1, NUM_PARAMS))

    fidelities = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
    for i in tf.range(tf.shape(y_true)[0]):
        # Create the quantum circuit with predicted parameters
        circuit = create_circuit(y_pred[i], num_qubits=NUM_QUBITS)

        # Simulate the circuit and get the final state vector
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state_vector = result.final_state_vector

        # Calculate fidelity
        target_state = y_true[i]
        state_vector = state_vector / np.linalg.norm(state_vector)
        target_state = target_state / np.linalg.norm(target_state)
        fidelity = tf.abs(tf.tensordot(tf.cast(tf.math.conj(target_state), dtype=tf.complex128), tf.cast(state_vector, dtype=tf.complex128), axes=1))**2
        fidelities = fidelities.write(i, fidelity)

    # Return the mean loss (1 - fidelity)
    return tf.reduce_mean(1 - fidelities.stack())

# Compile the model with the custom loss function
dnn_model.compile(optimizer='adam', loss=fidelity_loss)

# Implement the full training loop
dnn_model.fit(train_dataset, epochs=EPOCHS, verbose=2, validation_data=test_dataset)

# Example: Use the DNN to generate parameters and create a quantum circuit.
qubits = cirq.LineQubit.range(NUM_QUBITS)
example_input = X_test[0]
example_params = dnn_model.predict(np.expand_dims(example_input, axis=0)).flatten()
circuit = create_circuit(example_params, qubits)
print("Generated Circuit:")
print(circuit)
