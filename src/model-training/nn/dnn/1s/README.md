# DNN Quantum Circuit Optimization with Cirq (1s)

This directory contains code for optimizing quantum circuits using a hybrid approach that combines a Deep Neural Network (DNN) with the Cirq library. The original code used TensorFlow/Keras to train a DNN for regression on quantum circuit data. It has been refactored to integrate Cirq and demonstrate how a DNN can be used to generate parameters for a parameterized quantum circuit.

## Files

-   **cirq_dnn_optimizer.py**: This file contains the main code for the hybrid DNN-Cirq approach. It defines a DNN model that outputs parameters for a parameterized quantum circuit in Cirq. It also includes a placeholder for a custom loss function that would integrate the quantum circuit output into the DNN training.
-   **cpu_dnn_1s.py**: This was the original file that used TensorFlow/Keras for DNN regression. It's kept here for reference but is no longer used.
-   **cpu_dnn_32s_resume.py**: kept here for reference but is no longer used.
-   **cpu_dnn_32s.py**: kept here for reference but is no longer used.

## Refactoring Changes

1. **Cirq Integration**: A parameterized quantum circuit has been added using Cirq. The DNN is designed to output the parameters for this circuit.
2. **Hybrid Approach**: The code demonstrates a hybrid approach where the DNN generates parameters for a quantum circuit. The output of the quantum circuit could then be used as input to another layer or as the final output.
3. **Local Execution**: The code runs entirely locally on the CPU without any web server dependencies. TensorBoard logging has been removed.
4. **Placeholder for Custom Loss**: A placeholder is included for a custom loss function that would integrate the quantum circuit output into the DNN training. This is a complex topic and would require further research and development.

## Usage

1. Install the required libraries:

    ```bash
    pip install --user cirq numpy pandas scikit-learn tensorflow
    ```

2. Make sure you have the input data file (`qc8m_1s.csv` or a similar file) in the same directory.
3. Run the `cirq_dnn_optimizer.py` script:

    ```bash
    python cirq_dnn_optimizer.py
    ```

    This will create a DNN model, generate parameters for a quantum circuit using the DNN, and print the generated circuit.

## Notes

-   The code assumes a 5-qubit quantum circuit. You may need to adjust the `create_circuit` function if your data has a different structure.
-   The integration of the quantum circuit output into the DNN training is a placeholder. Implementing this would require defining a custom loss function and potentially using techniques for backpropagation through quantum circuits.
-   The DNN training is commented out as it's unclear how to train it without a proper integration with the quantum circuit.
-   This code serves as a starting point for exploring hybrid classical-quantum approaches. Further research and development are needed to create a fully functional optimization loop.
