# neural networks: Quantum Circuit Optimization with Deep Learning

This project explores the intersection of quantum computing and deep learning, focusing on optimizing quantum circuits using various neural network architectures and techniques. It leverages TensorFlow for classical machine learning, Cirq for quantum circuit manipulation, and includes implementations for Deep Crossing Networks (DCN), Deep Neural Networks (DNN), and dataset creation utilities.

## Project Structure

The project is organized into several directories, each focusing on a specific aspect of the quantum circuit optimization problem:

```
src/model-training/nn/
├── dcn/                 # Deep Crossing Network (DCN) implementation and Quantum Circuit Optimization
│   ├── cirq_circuit_optimizer.py # Quantum circuit optimizer using Cirq
│   ├── legacy_cpu_dcn.py         # DCN implementation using TensorFlow 2 and DeepCTR
│   ├── test_suite.py             # Unit tests for DCN and Cirq optimizer
│   └── README.md                # Documentation for the DCN module
├── dnn/                 # DNN implementations for 1-second and 32-second circuit optimization
│   ├── 1s/              # 1-second circuit optimization
│   │   ├── cirq_dnn_optimizer.py          # Hybrid DNN-Cirq optimizer for 1s circuit
│   │   ├── cpu_dnn_1s.py                 # (Reference) Original TensorFlow/Keras DNN for 1s circuit
│   │   ├── cpu_dnn_32s_resume.py          # (Reference) Original TensorFlow/Keras DNN for 32s circuit (resume)
│   │   ├── cpu_dnn_32s.py                 # (Reference) Original TensorFlow/Keras DNN for 32s circuit
│   │   └── README.md                     # Documentation for the 1s circuit optimization
│   ├── 32s/             # 32-second circuit optimization
│   │   ├── hybrid_dnn_cirq_optimizer_32s.py # Hybrid DNN-Cirq optimizer for 32s circuit
│   │   ├── cpu_dnn_32s.py                 # (Reference) Original TensorFlow/Keras DNN for 32s circuit
│   │   ├── cpu_dnn_32s_resume.py          # (Reference) Original TensorFlow/Keras DNN for 32s circuit (resume)
│   │   └── README.md                     # Documentation for the 32s circuit optimization
│   └── README.md        # Documentation for the DNN module
├── ds_dnn/              # Dataset creation and preprocessing for quantum circuit optimization
│   ├── cirq_dataset_creator.py # Creates a dataset of quantum circuits using Cirq
│   ├── cpu_keras_ds.py         # (Reference) Original TensorFlow dataset handling and DNN model
│   └── README.md                # Documentation for the dataset creation module
├── norm_ds_dnn/         # DNN training with a normalized dataset
│   ├── cpu_norm_ds_dnn.py      # Trains a DNN on a normalized dataset using TensorFlow
│   └── README.md                # Documentation for the normalized dataset DNN module
└── utils/               # Common utility functions for circuit creation and DNN model building
    ├── circuit_utils.py        # Helper functions for creating and evaluating quantum circuits using Cirq
    ├── model_utils.py          # Helper functions for building DNN models with TensorFlow
    └── README.md                # Documentation for the utility modules
```

## Modules

### Deep Crossing Network (DCN) (`dcn/`)

This module contains an implementation of a Deep Crossing Network (DCN) designed for regression tasks, particularly for predicting properties of quantum circuits. It also includes a Cirq-based quantum circuit optimizer that uses a gradient descent approach.

**Key Features:**

-   **`legacy_cpu_dcn.py`**: Trains a DCN model to predict the statevector output of a quantum circuit.
-   **`cirq_circuit_optimizer.py`**: Optimizes the parameters of a quantum circuit to minimize the difference between the circuit's output and a target statevector.
-   **`test_suite.py`**: Contains unit tests for both the DCN and Cirq optimizer.

### Deep Neural Networks (DNN) (`dnn/`)

This module focuses on using DNNs for quantum circuit optimization, with specific implementations for 1-second (`1s/`) and 32-second (`32s/`) quantum circuits. It demonstrates a hybrid classical-quantum approach where DNNs generate parameters for parameterized Cirq circuits.

**Key Features:**

-   **`1s/cirq_dnn_optimizer.py`**: Hybrid DNN-Cirq optimizer for 1-second circuits.
-   **`32s/hybrid_dnn_cirq_optimizer_32s.py`**: Hybrid DNN-Cirq optimizer for 32-second circuits.
-   **Reference Implementations**: Includes original TensorFlow/Keras DNN code for both 1s and 32s circuits.

### Dataset Creation and Preprocessing (`ds_dnn/`)

This module handles the creation of datasets of quantum circuits from CSV data. It integrates Cirq to convert features into quantum circuits and returns a dataset suitable for quantum machine learning tasks.

**Key Features:**

-   **`cirq_dataset_creator.py`**: Creates a dataset of quantum circuits, adapting preprocessing logic for Cirq compatibility.
-   **Reference Implementation**: Includes the original TensorFlow dataset handling code for reference.

### Normalized Dataset DNN (`norm_ds_dnn/`)

This module contains a script for training a DNN using a normalized dataset stored in CSV shards. It demonstrates the use of TensorFlow's `tf.data` API for efficient data loading and batching.

**Key Features:**

-   **`cpu_norm_ds_dnn.py`**: Trains a DNN on a normalized dataset using TensorFlow.
-   **Data Loading and Preprocessing**: Uses `tf.data` for efficient data handling.
-   **Model Checkpointing and TensorBoard Logging**: Includes callbacks for model saving and training progress visualization (removed in some refactored versions).

### Utility Functions (`utils/`)

This module provides common utility functions used across the project. It includes helpers for creating quantum circuits using Cirq and building DNN models with TensorFlow.

**Key Features:**

-   **`circuit_utils.py`**: Functions for creating parameterized quantum circuits and calculating the loss between a circuit's output and a target state.
-   **`model_utils.py`**: Functions for creating DNN models that output circuit parameters.

## Getting Started

### Installation

To use this project, you need to install the following libraries:

```bash
pip install --user cirq numpy pandas scikit-learn tensorflow
```

### Data

Ensure that the required data files (e.g., `qc8m_1s.csv`, `qc7_8m.csv`, and CSV shards) are placed in the appropriate directories as specified in the README files of individual modules. You may need to adjust file paths within the scripts based on your data location.

### Running the Scripts

Each module contains scripts that can be run independently. Refer to the README files within each module for specific instructions on how to run the scripts. For example:

-   To train the DCN model: `python dcn/legacy_cpu_dcn.py`
-   To run the hybrid DNN-Cirq optimizer for 1s circuits: `python dnn/1s/cirq_dnn_optimizer.py`
-   To create a dataset of quantum circuits: `python ds_dnn/cirq_dataset_creator.py`
-   To train a DNN on a normalized dataset: `python norm_ds_dnn/cpu_norm_ds_dnn.py`

## Notes

-   The project assumes a 5-qubit quantum circuit structure in many places. You may need to modify the code if your data has a different structure or if you want to use a different circuit.
-   The integration of quantum circuit output into DNN training is currently a placeholder in the `dnn/` module. It requires further development to define a custom loss function and potentially use techniques for backpropagation through quantum circuits.
-   This project serves as a starting point for exploring hybrid classical-quantum approaches to quantum circuit optimization. Further research and development are needed to create fully functional optimization loops and explore more advanced techniques.
- Refer to the README files within each module for more detailed information and specific usage instructions.
- Each of the modules has a dedicated README with usage instructions and important considerations.
- Be sure to update the dataset paths in scripts to match your local file structure.

## Future Work

-   Develop a custom loss function for integrating quantum circuit output into DNN training.
-   Explore techniques for backpropagation through quantum circuits.
-   Experiment with different DNN architectures and hyperparameters.
-   Investigate more advanced optimization algorithms for both the classical and quantum components.
-   Apply these techniques to larger and more complex quantum circuits.
-   Explore the use of quantum hardware for circuit simulation and optimization.
-   Develop more comprehensive testing and validation procedures.


