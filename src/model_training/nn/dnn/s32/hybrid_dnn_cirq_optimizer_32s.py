import cirq
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from model_training.nn.utils.circuit_utils import create_circuit, simulate_circuit, calculate_fidelity, calculate_fidelity_loss # Corrected path
from model_training.nn.utils.model_utils import create_dnn_model, DNNModel # Corrected path
import config # Corrected import
from model_training.nn.utils.data_utils import load_and_preprocess_data # Corrected path

# Configuration parameters (using centralized config)
BATCH_SIZE = config.DNN_32S_BATCH_SIZE
EPOCHS = config.DNN_32S_EPOCHS
CSV_FILE_PATH = config.DNN_32S_CSV_FILE_PATH
NUM_QUBITS = config.DNN_NUM_QUBITS # Use general DNN value
NUM_PARAMS = config.DNN_NUM_PARAMS # Use general DNN value
CIRCUIT_TYPE = config.DNN_CIRCUIT_TYPE # Use general DNN value

print("Using Cirq", cirq.__version__)
print("Using TensorFlow", tf.__version__)

# Define fidelity loss function (can be defined at module level)
def fidelity_loss(y_true, y_pred):
    # Assuming calculate_fidelity_loss is defined correctly in circuit_utils
    # Need to ensure NUM_QUBITS and CIRCUIT_TYPE are accessible or passed
    return calculate_fidelity_loss(y_true, y_pred, NUM_QUBITS, CIRCUIT_TYPE)

# Main execution block
if __name__ == "__main__":
    # Load and preprocess data
    # Encapsulate data loading and preprocessing within main block
    try:
        # Note: target_type='multi' for 32s version
        X_train, X_test, y_train, y_test, input_dim = load_and_preprocess_data(
            CSV_FILE_PATH, target_type='multi', n_qubits=NUM_QUBITS
        )
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading/preprocessing: {e}")
        exit()

    # Check if data loading was successful and sets are not empty
    if 'X_train' not in locals() or X_train.size == 0 or X_test.size == 0:
         print("Error: Data loading failed or resulted in empty datasets.")
         exit()

    # Convert data to TensorFlow Datasets
    # Ensure BATCH_SIZE is valid before batching
    if BATCH_SIZE <= 0:
        print(f"Error: Invalid BATCH_SIZE ({BATCH_SIZE}). Must be positive.")
        exit()
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # Define and compile model
    # input_dim is determined by load_and_preprocess_data
    dnn_model = DNNModel(input_dim, NUM_PARAMS) # Use the class directly

    # Compile the model - using standard MSE for now as fidelity_loss might be complex
    # dnn_model.compile(optimizer='adam', loss=fidelity_loss)
    # Using standard loss for initial setup; fidelity_loss requires careful implementation
    dnn_model.compile(optimizer='adam', loss='mse')

    # Train the model
    print("Starting model training...")
    # dnn_model.fit(train_dataset, epochs=EPOCHS, verbose=2, validation_data=test_dataset)
    # Using the train method from the base class if fit is not overridden
    dnn_model.train(train_dataset, epochs=EPOCHS, validation_data=test_dataset)
    print("Model training finished.")

    # Example: Use the DNN to generate parameters and create a quantum circuit.
    print("Generating example circuit with trained model...")
    qubits = cirq.LineQubit.range(NUM_QUBITS)
    # Ensure X_test is not empty before accessing
    if X_test.shape[0] > 0:
        example_input = X_test[0]
        # Use the model's predict method
        example_params = dnn_model.predict(np.expand_dims(example_input, axis=0)).flatten()
        circuit = create_circuit(example_params, qubits, circuit_type=CIRCUIT_TYPE)
        print("Generated Circuit:")
        print(circuit)
    else:
        print("Test set is empty, cannot generate example circuit.")
