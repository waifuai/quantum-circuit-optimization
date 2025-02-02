# dcn/cirq_circuit_optimizer.py
import cirq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.circuit_utils import create_circuit, calculate_loss

# Configuration parameters
BATCH_SIZE = 128
EPOCHS = 3

print("Using Cirq", cirq.__version__)

# Load data (update with your actual data loading logic)
data = pd.read_csv("./qc5f_1k_2.csv")

# Define features and target (adapt these to your CSV structure)
dense_features = [
    f"gate_{str(i).zfill(2)}_{suffix}"
    for i in range(41)
    for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
]
target = ['statevector_00000']

# Prepare training and test sets
train, test = train_test_split(data, test_size=0.2)
train = train[: (len(train) // BATCH_SIZE) * BATCH_SIZE]
test = test[: (len(test) // BATCH_SIZE) * BATCH_SIZE]

train_features = {name: train[name].values.astype('float32') for name in dense_features}
test_features = {name: test[name].values.astype('float32') for name in dense_features}
train_target = train[target].values
test_target = test[target].values

def optimize_circuit(train_features, train_target):
    """Optimizes circuit parameters using a simple gradient descent."""
    params = np.random.rand(25)  # 5 layers * 5 qubits
    for epoch in range(EPOCHS):
        total_loss = 0
        for features, target_state in zip(train_features.values(), train_target):
            circuit = create_circuit(params)
            loss = calculate_loss(circuit, target_state)
            total_loss += loss

            # Compute numerical gradients
            gradients = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += 0.01
                loss_plus = calculate_loss(create_circuit(params_plus), target_state)

                params_minus = params.copy()
                params_minus[i] -= 0.01
                loss_minus = calculate_loss(create_circuit(params_minus), target_state)

                gradients[i] = (loss_plus - loss_minus) / 0.02

            params -= 0.1 * gradients  # Update parameters

        avg_loss = total_loss / len(train_features)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
    return params

if __name__ == '__main__':
    optimized_params = optimize_circuit(train_features, train_target)
    print("Optimized parameters:", optimized_params)
    circuit = create_circuit(optimized_params)
    print("Optimized Circuit:")
    print(circuit)
