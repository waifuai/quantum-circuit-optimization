# DCN Quantum Circuit Optimization with Cirq

This directory contains code for optimizing quantum circuits using the Cirq library. The original code used the DeepCTR library and TensorFlow for training a Deep & Cross Network (DCN) model. It has been refactored to use Cirq for quantum circuit operations and to run entirely locally on the CPU without any web server dependencies.

## Files

-   **cirq_circuit_optimizer.py**: This file contains the main code for optimizing quantum circuits using Cirq. It defines functions to create a parameterized quantum circuit, calculate the loss (e.g., fidelity with a target state), and optimize the circuit parameters using a basic gradient descent approach.
-   **cpu_dcn.py**: This was the original file that used DeepCTR and TensorFlow. It's kept here for reference but is no longer used.

## Refactoring Changes

1. **Cirq Integration**: The DeepCTR and TensorFlow code has been replaced with Cirq for quantum circuit operations. The `create_circuit` function defines a parameterized quantum circuit using Cirq, and the `calculate_loss` function computes the loss based on the fidelity between the circuit output and a target state.
2. **Local Execution**: The code has been modified to run entirely locally without any web server dependencies. TensorBoard logging has been removed.
3. **CPU Only**: The code is designed to run on the CPU only, without using GPUs or TPUs.
4. **Simplified Optimization**: A basic gradient descent optimizer is used for demonstration purposes. This can be replaced with a more sophisticated optimizer like those available in the `cirq.optim` module.

## Usage

1. Install the required libraries:

    ```bash
    pip install --user cirq numpy pandas scikit-learn
    ```

2. Make sure you have the input data file (`qc5f_1k_2.csv` or a similar file) in the same directory.
3. Run the `cirq_circuit_optimizer.py` script:

    ```bash
    python cirq_circuit_optimizer.py
    ```

    This will optimize the quantum circuit parameters and print the optimized circuit.

## Notes

-   The code assumes a 5-qubit quantum circuit based on the structure of the input data. You may need to adjust the `create_circuit` function if your data has a different structure.
-   The optimization process is simplified for demonstration purposes. You can improve it by using a more advanced optimizer, implementing a better loss function, and tuning the hyperparameters.
-   The evaluation logic is basic. You should replace it with a more comprehensive evaluation method based on your specific task.
