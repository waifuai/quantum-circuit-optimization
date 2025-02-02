# DS_DNN with Cirq Dataset Creation

This directory contains code related to dataset creation and preprocessing for quantum circuit optimization using Cirq. The original code used TensorFlow for dataset handling and included a simple DNN model. It has been refactored to integrate Cirq and demonstrate how to create a dataset of quantum circuits from the CSV data.

## Files

-   **cirq_dataset_creator.py**: This file contains the main code for creating a dataset of quantum circuits. It adapts the preprocessing logic from `cpu_keras_ds.py` and the circuit conversion logic from `cirq_data_loader.py`. The `create_circuit_dataset` function returns a dataset where each element is a tuple of (quantum circuit, label).
-   **cpu_keras_ds.py**: This was the original file that used TensorFlow for dataset handling and included a simple DNN model. It's kept here for reference but is no longer used.

## Refactoring Changes

1. **Cirq Integration**: The dataset creation process has been modified to be compatible with Cirq. The `features_to_circuit` function converts features into a quantum circuit, and the `create_circuit_dataset` function returns a dataset of quantum circuits.
2. **Local Execution**: The code runs entirely locally on the CPU without any web server dependencies. TensorBoard logging has been removed.
3. **Simplified Dataset Handling**: The code now focuses on creating a dataset of quantum circuits, without the integrated DNN model.

## Usage

1. Install the required libraries:

    ```bash
    pip install --user cirq numpy pandas scikit-learn tensorflow
    ```

2. Make sure you have the input data files in the `shards/` directory or update the `DIR_PATH` in `cirq_dataset_creator.py`.
3. Run the `cirq_dataset_creator.py` script:

    ```bash
    python cirq_dataset_creator.py
    ```

    This will create training and validation datasets of quantum circuits and print an example circuit.

## Notes

-   The code assumes a 5-qubit quantum circuit. You may need to adjust the `features_to_circuit` function if your data has a different structure or if you want to use a different circuit structure.
-   The code includes a placeholder for further processing with Cirq. This is where you would integrate the quantum circuit outputs into your machine learning pipeline. For example, you could simulate the circuits, obtain the output state vectors, and then use those state vectors as input to a classical machine learning model or for further quantum processing.
-   This code serves as a starting point for exploring how to use Cirq for dataset creation and preprocessing in quantum machine learning tasks.
