# dcn/cirq_circuit_optimizer.py
import cirq
import numpy asnp
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from utils.circuit_utils import create_circuit

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

def calculate_fidelity(circuit: cirq.Circuit, target_state: np.ndarray) -> float:
    """Calculates the fidelity between the circuit's output statevector and the target state."""
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state_vector = result.final_state_vector
    
    # Ensure both vectors are normalized
    state_vector = state_vector / np.linalg.norm(state_vector)
    target_state = target_state / np.linalg.norm(target_state)

    # Compute fidelity
    fidelity = np.abs(np.dot(np.conjugate(target_state), state_vector))**2
    return fidelity

def optimize_circuit(train_features, train_target, initial_params=None):
    """Optimizes circuit parameters using the BFGS algorithm."""

    def loss_function(params):
        """Calculates the average loss (1-fidelity) over the training set for given circuit parameters."""
        total_loss = 0
        for features, target_state in zip(train_features.values(), train_target):
            circuit = create_circuit(params, num_qubits=5)
            fidelity = calculate_fidelity(circuit, target_state)
            loss = 1 - fidelity
            total_loss += loss
        return total_loss / len(train_features)

    if initial_params is None:
        initial_params = np.random.rand(25)  # 5 layers * 5 qubits
    result = minimize(loss_function, initial_params, method='BFGS')

    if result.success:
        print("Optimization successful!")
        return result.x
    else:
        print("Optimization failed:", result.message)
        return initial_params  # Return initial parameters if optimization fails

if __name__ == '__main__':
    optimized_params = optimize_circuit(train_features, train_target)
    print("Optimized parameters:", optimized_params)
    circuit = create_circuit(optimized_params, num_qubits=5)
    print("Optimized Circuit:")
    print(circuit)
