# dnn/1s/hybrid_dnn_cirq_optimizer.py
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

# Define model dimensions
num_params = 25  # 5 layers * 5 qubits
input_dim = len(dense_features)

dnn_model = create_dnn_model(input_dim, num_params)
dnn_model.compile(optimizer='adam', loss='mse')

# Training is commented out – integrate your custom loss if needed.
# dnn_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

# Example: Use the DNN to generate parameters and create a quantum circuit.
qubits = cirq.LineQubit.range(5)
example_input = X_test[0]
example_params = dnn_model.predict(np.expand_dims(example_input, axis=0)).flatten()
circuit = create_circuit(example_params, qubits)
print("Generated Circuit:")
print(circuit)
