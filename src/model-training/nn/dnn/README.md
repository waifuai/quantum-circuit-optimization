# DNN Quantum Circuit Optimization

This directory contains code for optimizing quantum circuits using a hybrid approach that combines a Deep Neural Network (DNN) with the Cirq quantum computing library. It includes implementations for two different quantum circuit datasets: a 1-second circuit (`1s`) and a 32-second circuit (`32s`).

## Overview

The original code in this project used TensorFlow/Keras to train DNNs for regression on quantum circuit data. The code has been refactored to integrate Cirq and demonstrate how a DNN can be used to generate parameters for a parameterized quantum circuit. This represents a shift towards a hybrid classical-quantum approach for circuit optimization.

## Subdirectories

### 1s

This subdirectory contains code related to the optimization of a 1-second quantum circuit.

-   **`hybrid_dnn_cirq_optimizer.py`**: The main script for the hybrid DNN-Cirq approach for the 1s circuit.
-   **README.md**: Documentation for the `1s` subdirectory.

### 32s

This subdirectory contains code related to the optimization of a 32-second quantum circuit.

-   **`hybrid_dnn_cirq_optimizer_32s.py`**: The main script for the hybrid DNN-Cirq approach for the 32s circuit.
-   **README.md**: Documentation for the `32s` subdirectory.

## Common Features

Both subdirectories share these key features:

1. **Cirq Integration**: Parameterized quantum circuits are implemented using Cirq.
2. **Hybrid Approach**: DNNs are used to generate parameters for the quantum circuits, demonstrating a hybrid classical-quantum approach.
3. **Local Execution**: The code runs locally on the CPU without web server dependencies. TensorBoard logging is removed.
4. **Placeholder for Custom Loss**: A placeholder is included for a custom loss function that would integrate the quantum circuit output into DNN training. This is a complex topic requiring further development.

## Usage

1. **Install Libraries**:

    ```bash
    pip install --user cirq numpy pandas scikit-learn tensorflow
    ```

2. **Data Files**: Ensure the input data files (`qc8m_1s.csv` for `1s` and `qc7_8m.csv` for `32s`) are in the appropriate directories or adjust paths within the scripts.

3. **Run Scripts**:

    -   For the 1s circuit: `python 1s/hybrid_dnn_cirq_optimizer.py`
    -   For the 32s circuit: `python 32s/hybrid_dnn_cirq_optimizer_32s.py`

    These scripts will create DNN models, generate parameters for quantum circuits using the DNNs, and print the generated circuits.

## Notes

-   Both implementations assume a 5-qubit quantum circuit. Adjust the `create_circuit` function in each script if your data has a different structure.
-   The integration of quantum circuit output into DNN training is currently a placeholder. It requires defining a custom loss function and potentially using techniques for backpropagation through quantum circuits.
-   DNN training is commented out as it's unclear how to train without proper integration with the quantum circuit.
-   This code serves as a starting point for exploring hybrid classical-quantum approaches. Further research and development are needed to create a fully functional optimization loop.