# Deep Crossing Network (DCN) and Quantum Circuit Optimization

This repository contains code for training a Deep Crossing Network (DCN) and optimizing quantum circuits using Cirq. It includes two main components:

1. **DCN Implementation (`legacy_cpu_dcn.py`)**: A TensorFlow 2 based implementation of a DCN model for regression tasks, specifically designed to predict properties of quantum circuits.
2. **Quantum Circuit Optimizer (`cirq_circuit_optimizer.py`)**: A simple gradient descent optimizer that adjusts the parameters of a quantum circuit in Cirq to minimize a loss function.

## Repository Structure

```
src/model-training/nn/dcn/
├── cirq_circuit_optimizer.py  # Quantum circuit optimizer using Cirq
├── legacy_cpu_dcn.py          # DCN implementation using TensorFlow 2 and DeepCTR
└── test_suite.py              # Unit tests for both modules
```

## `legacy_cpu_dcn.py`

### Description

This script trains a DCN model to predict the statevector output (`statevector_00000`) of a quantum circuit given its gate configurations. It utilizes the DeepCTR library for the DCN model and TensorFlow 2 for training.

### Key Features

*   **Data Loading and Preparation**: Loads data from a CSV file, splits it into training and testing sets, and prepares the input data for the DCN model. The data is assumed to contain dense features representing the gates of a quantum circuit.
*   **Model Definition**: Defines a DCN model with configurable hidden layers and cross layers.
*   **Training and Evaluation**: Trains the model using Adam optimizer and mean squared error (MSE) loss. Evaluates the model's performance using root mean squared error (RMSE).
*   **TensorBoard Integration**: Logs training and validation RMSE to TensorBoard for visualization.
*   **Model Checkpointing**: Saves the best model during training based on validation MSE.

### Usage

1. **Prepare your data**: Create a CSV file with the required format. The CSV should contain columns representing the features of the quantum circuit (e.g., `gate_00_Gate_Type`, `gate_00_Gate_Number`, etc.) and the target variable (`statevector_00000`).
2. **Update file path**: Modify the `csv_path` in the `load_and_prepare_data` function to point to your CSV file.
3. **Run the script**: Execute the script using `python legacy_cpu_dcn.py`.
4. **Monitor training**: Use TensorBoard to visualize the training progress by running `tensorboard --logdir cpu/logs/fit/`.

### Dependencies

*   TensorFlow 2.x
*   pandas
*   scikit-learn
*   DeepCTR

## `cirq_circuit_optimizer.py`

### Description

This script optimizes the parameters of a quantum circuit defined in Cirq. The goal is to find the parameters that minimize the difference between the circuit's output statevector and a target statevector. The optimization is performed using a basic gradient descent algorithm.

### Key Features

*   **Circuit Creation**: Creates a quantum circuit using Cirq based on input parameters.
*   **Loss Calculation**: Calculates the loss between the circuit's output and a target state.
*   **Gradient-based Optimization**: Computes numerical gradients and updates the circuit parameters to minimize the loss.

### Usage

1. **Define your target state**: Modify the `target` variable in the script to reflect the desired statevector you want to approximate.
2. **Update Data loading logic** with your actual data.
3. **Run the script**: Execute the script using `python cirq_circuit_optimizer.py`.

### Dependencies

*   Cirq
*   NumPy
*   pandas
*   scikit-learn

## `test_suite.py`

### Description

This script contains unit tests for both `cirq_circuit_optimizer.py` and `legacy_cpu_dcn.py`. It uses the `unittest` framework and `patch` from `unittest.mock` to test various functionalities of the two modules.

### Tests for `cirq_circuit_optimizer.py`

*   **`test_optimize_circuit_returns_array`**: Verifies that the `optimize_circuit` function returns a NumPy array of parameters with the expected shape.
*   **`test_loss_function_calls`**: Checks that the `calculate_loss` and `create_circuit` functions are called within the optimization loop as expected.

### Tests for `legacy_cpu_dcn.py`

*   **`test_create_save_paths`**: Ensures that the `create_save_paths` function returns valid log and model paths.
*   **`test_load_and_prepare_data`**: Tests the `load_and_prepare_data` function by creating a temporary CSV file and verifying the types and dimensions of the returned inputs.
*   **`test_build_dcn_model`**: Confirms that the `build_dcn_model` function returns a compiled model with the correct optimizer and loss function.

### Usage

Run the test suite using:

```bash
python test_suite.py
```

## Notes

*   The code assumes a specific format for the input CSV file. You may need to adapt the data loading and preparation steps based on your data.
*   The DCN model's architecture and hyperparameters can be adjusted in the `build_dcn_model` function.
*   The quantum circuit structure and loss function in `cirq_circuit_optimizer.py` can be modified to suit different optimization tasks.
*   The optimizer in `cirq_circuit_optimizer.py` uses a simple gradient descent with a fixed learning rate. More advanced optimization techniques could be explored for better performance.
*   The `test_suite.py` provides basic unit tests, but further testing may be beneficial for robust development.
*   This code is intended as a starting point and can be extended for more complex quantum circuit optimization problems or different types of state estimation.
*   Ensure that all dependencies are installed before running the scripts.

## Contact

For questions or feedback, please open an issue on the GitHub repository.
