# dcn/cirq_circuit_optimizer.py
import cirq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from model_training.nn.utils.circuit_utils import create_circuit
import config

# Configuration parameters
BATCH_SIZE = config.BATCH_SIZE # Note: BATCH_SIZE is defined in config now
EPOCHS = 3 # This seems specific to this script, keeping it local

print("Using Cirq", cirq.__version__)

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

    # Define loss function within optimize_circuit to capture train_features/train_target
    def loss_function(params):
        """Calculates the average loss (1-fidelity) over the training set for given circuit parameters."""
        total_loss = 0
        # Assuming train_features is a dict and train_target is an array accessible here
        # Need to ensure train_features and train_target are passed correctly or accessible
        # For simplicity, assuming they are passed correctly to optimize_circuit
        num_samples = len(next(iter(train_features.values()))) # Get number of samples from first feature
        for i in range(num_samples):
             # Reconstruct features for the i-th sample if necessary, or assume zip works
             # This part might need adjustment depending on how train_features is structured
             # Assuming zip works as intended for now
             # features_sample = {k: v[i] for k, v in train_features.items()} # Example if needed
             target_state_sample = train_target[i]

             # Create circuit with current params
             # Assuming create_circuit uses params correctly
             circuit = create_circuit(params, num_qubits=5) # Assuming 5 qubits

             # Calculate fidelity for this sample
             fidelity = calculate_fidelity(circuit, target_state_sample)
             loss = 1 - fidelity
             total_loss += loss
        return total_loss / num_samples # Average loss

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
    # Load data (update with your actual data loading logic)
    try:
        data = pd.read_csv("./qc5f_1k_2.csv")
    except FileNotFoundError:
        print("Error: ./qc5f_1k_2.csv not found. Please ensure the data file exists.")
        exit() # Exit if data file not found

    # Define features and target (adapt these to your CSV structure)
    dense_features = [
        f"gate_{str(i).zfill(2)}_{suffix}"
        for i in range(41)
        for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
    ]
    target = ['statevector_00000']

    # Prepare training and test sets
    train, test = train_test_split(data, test_size=0.2)

    # Ensure BATCH_SIZE is not zero and train/test sets are not empty
    if BATCH_SIZE > 0 and len(train) > 0 and len(test) > 0:
         train = train[: (len(train) // BATCH_SIZE) * BATCH_SIZE]
         test = test[: (len(test) // BATCH_SIZE) * BATCH_SIZE]
    else:
         print("Warning: BATCH_SIZE is zero or train/test split resulted in empty sets. Skipping truncation.")
         # Handle cases where truncation might lead to empty sets if needed

    # Check if train/test are empty after potential truncation
    if train.empty or test.empty:
        print("Error: Training or testing set is empty after processing. Check data and BATCH_SIZE.")
        exit()

    train_features = {name: train[name].values.astype('float32') for name in dense_features}
    test_features = {name: test[name].values.astype('float32') for name in dense_features}
    train_target = train[target].values
    test_target = test[target].values

    # Now call optimize_circuit with the prepared data
    optimized_params = optimize_circuit(train_features, train_target)
    print("Optimized parameters:", optimized_params)
    circuit = create_circuit(optimized_params, num_qubits=5)
    print("Optimized Circuit:")
    print(circuit)
