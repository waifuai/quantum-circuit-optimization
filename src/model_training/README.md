# model training: Quantum Circuit Optimization with Deep Learning

This repository explores the use of deep learning techniques, specifically neural networks, to optimize quantum circuits. It leverages TensorFlow for classical machine learning, Cirq for quantum circuit manipulation, and Trax for advanced sequence-to-sequence modeling with Transformers. The project is divided into several modules, each focusing on a different aspect of the problem, including data preprocessing, model training, and prediction.

## Project Structure

The project is organized into the following directories:

-   **`model-training/`**: Contains code for training different types of neural network models.
    -   **`nn/`**: Focuses on classical neural network architectures for quantum circuit optimization.
        -   **`dcn/`**: Deep Crossing Network (DCN) implementation.
            -   `cirq_circuit_optimizer.py`: Quantum circuit optimizer using Cirq.
            -   `legacy_cpu_dcn.py`: DCN implementation using TensorFlow 2 and DeepCTR.
            -   `test_suite.py`: Unit tests for DCN and Cirq optimizer.
        -   **`dnn/`**: Deep Neural Networks (DNN) for 1-second and 32-second circuit optimization.
            -   `1s/`: 1-second circuit optimization.
                -   `cirq_dnn_optimizer.py`: Hybrid DNN-Cirq optimizer for 1s circuit.
                -   `cpu_dnn_1s.py`: (Reference) Original TensorFlow/Keras DNN for 1s circuit.
                -   `cpu_dnn_32s_resume.py`: (Reference) Original TensorFlow/Keras DNN for 32s circuit (resume).
                -   `cpu_dnn_32s.py`: (Reference) Original TensorFlow/Keras DNN for 32s circuit.
            -   `32s/`: 32-second circuit optimization.
                -   `hybrid_dnn_cirq_optimizer_32s.py`: Hybrid DNN-Cirq optimizer for 32s circuit.
                -   `cpu_dnn_32s.py`: (Reference) Original TensorFlow/Keras DNN for 32s circuit.
                -   `cpu_dnn_32s_resume.py`: (Reference) Original TensorFlow/Keras DNN for 32s circuit (resume).
        -   **`ds_dnn/`**: Dataset creation and preprocessing.
            -   `cirq_dataset_creator.py`: Creates a dataset of quantum circuits using Cirq.
            -   `cpu_keras_ds.py`: (Reference) Original TensorFlow dataset handling and DNN model.
        -   **`norm_ds_dnn/`**: DNN training with a normalized dataset.
            -   `cpu_norm_ds_dnn.py`: Trains a DNN on a normalized dataset using TensorFlow.
        -   **`utils/`**: Common utility functions.
            -   `circuit_utils.py`: Helper functions for creating and evaluating quantum circuits using Cirq.
            -   `model_utils.py`: Helper functions for building DNN models with TensorFlow.
    -   **`prep/`**: Scripts for data preprocessing.
        -   `csv_norm_new.py`: Normalizes numeric features in CSV shards using universal normalization constants.
        -   `csv_norm_old.py`: Older normalization script (loads entire CSV into memory).
        -   `csv_to_tfrecord.py`: Converts preprocessed CSV to TFRecord format.
    -   **`trax-train/`**:  Transformer model for circuit optimization using Trax.
        -   `src/`:
            -   `prep.py`: Preprocesses data for the Transformer.
            -   `train.py`: Defines and trains the Transformer model.
            -   `predict.py`: Loads a trained model and performs inference.
            -   `utils.py`: Utility functions for data loading and generators.
        -   `data/`: Directory for quantum circuit data.
        -   `model/`: Directory for saving model checkpoints.
        -   `requirements.txt`: Lists required Python packages.
    -   **`trax-transformer/`**:  Simplified version of the Trax Transformer model.
        -   `scripts/`: Contains `prep.sh` for data preparation.
        -   `src/`:
            -   `trainer/`: Contains `problem.py` for defining the Trax problem and model.
            -   `train.py`: Trains the model using Trax's training loop.
            -   `predict.py`: Uses the trained model for prediction.
        -   `tests/`: Test suite for the project.
        -   `requirements.txt`: Lists required Python packages.

## Modules

### Deep Crossing Network (DCN) (`model-training/nn/dcn/`)

Implements a Deep Crossing Network for regression tasks, particularly for predicting properties of quantum circuits. Includes a Cirq-based optimizer that uses a gradient descent approach.

### Deep Neural Networks (DNN) (`model-training/nn/dnn/`)

Uses DNNs for quantum circuit optimization, with implementations for 1-second (`1s/`) and 32-second (`32s/`) circuits. Demonstrates a hybrid classical-quantum approach where DNNs generate parameters for parameterized Cirq circuits.

### Dataset Creation and Preprocessing (`model-training/nn/ds_dnn/`)

Handles the creation of datasets of quantum circuits from CSV data. Integrates Cirq to convert features into quantum circuits and returns a dataset suitable for quantum machine learning tasks.

### Normalized Dataset DNN (`model-training/nn/norm_ds_dnn/`)

Trains a DNN using a normalized dataset stored in CSV shards. Utilizes TensorFlow's `tf.data` API for efficient data loading and batching.

### Utility Functions (`model-training/nn/utils/`)

Provides common utility functions for creating quantum circuits using Cirq and building DNN models with TensorFlow.

### Data Preprocessing (`model-training/prep/`)

Contains scripts for preprocessing CSV data, including normalization and conversion to TFRecord format.

### Trax Transformer (`model-training/trax-train/` and `model-training/trax-transformer/`)

Uses a Transformer model implemented in Trax to optimize quantum circuits by learning to transform unoptimized circuits into more efficient versions. Includes data preprocessing, model training, and prediction functionalities.

## Getting Started

### Installation

To use this project, clone the repository and install the required libraries:

```bash
# Navigate to the specific module you want to work with, e.g.,
cd src/model-training/nn/
pip install --user cirq numpy pandas scikit-learn tensorflow
# For the Trax modules, use the provided requirements.txt, e.g.,
cd ../trax-train/
pip install -r requirements.txt
# Or
cd ../trax-transformer/
pip install --user -r requirements.txt
```

### Data

-   **Classical Neural Networks (`nn/`)**: Ensure that the required CSV data files (e.g., `qc8m_1s.csv`, `qc7_8m.csv`, and CSV shards) are placed in the appropriate directories as specified in the README files of individual modules. You may need to adjust file paths within the scripts.
-   **Preprocessing (`prep/`)**: If your dataset is large, shard it into smaller CSV files. Place these files in a directory accessible to the preprocessing scripts.
-   **Trax Transformer (`trax-train/` and `trax-transformer/`)**: Place your `input.txt` (unoptimized circuits) and `output.txt` (optimized circuits) files in the `data/` directory. Each line should represent a circuit. Run `prep.py` (or `prep.sh` for `trax-transformer/`) to prepare the data for training. For prediction, place `phrases_input.txt` in the `data/` directory.

### Running the Scripts

Each module contains scripts that can be run independently. Refer to the README files within each module for specific instructions.

#### Examples:

-   **Train DCN**:
    ```bash
    python src/model-training/nn/dcn/legacy_cpu_dcn.py
    ```
-   **Run Hybrid DNN-Cirq Optimizer (1s)**:
    ```bash
    python src/model-training/nn/dnn/1s/cirq_dnn_optimizer.py
    ```
-   **Create Quantum Circuit Dataset**:
    ```bash
    python src/model-training/nn/ds_dnn/cirq_dataset_creator.py
    ```
-   **Train DNN on Normalized Dataset**:
    ```bash
    python src/model-training/nn/norm_ds_dnn/cpu_norm_ds_dnn.py
    ```
-   **Normalize Data**:
    ```bash
    python src/model-training/prep/csv_norm_new.py --input_dir=<path_to_csv_shards> --output_dir=<output_directory>
    ```
-   **Convert to TFRecord**:
    ```bash
    python src/model-training/prep/csv_to_tfrecord.py --input_file=<normalized_csv_file> --output_file=<output_tfrecord_file>
    ```
-   **Train Trax Transformer**:
    ```bash
    # For trax-train/
    python src/model-training/trax-train/src/train.py

    # For trax-transformer/
    python src/model-training/trax-transformer/src/train.py --input_file=input_processed.txt --output_file=output_processed.txt --model_dir=model
    ```
-   **Predict with Trax Transformer**:
    ```bash
    # For trax-train/
    python src/model-training/trax-train/src/predict.py

    # For trax-transformer/
    python src/model-training/trax-transformer/src/predict.py model "1 2 3 4"
    ```

## Notes

-   The project assumes a 5-qubit quantum circuit structure in many places. Adjust the code if your data has a different structure or if you want to use a different circuit.
-   The integration of quantum circuit output into DNN training in the `dnn/` module is a placeholder and requires further development.
-   This project serves as a starting point for exploring hybrid classical-quantum approaches to quantum circuit optimization. Further research and development are needed.
-   Training Transformer models can be computationally intensive. Consider using a GPU for faster training.
-   Performance depends on the quality and quantity of training data.
-   Refer to the README files within each module for more detailed information.
-   Be sure to update the dataset paths in scripts to match your local file structure.

## Future Work

-   Develop a custom loss function for integrating quantum circuit output into DNN training.
-   Explore techniques for backpropagation through quantum circuits.
-   Experiment with different DNN and Transformer architectures and hyperparameters.
-   Investigate more advanced optimization algorithms.
-   Apply these techniques to larger and more complex quantum circuits.
-   Explore the use of quantum hardware for circuit simulation and optimization.
-   Develop more comprehensive testing and validation procedures.
-   Improve the Trax Transformer implementation by exploring different decoding methods (e.g. beam search in `trax-train`) and potentially using a more robust data representation.
-   Incorporate error mitigation techniques into the optimization process.
-   Investigate the use of reinforcement learning for quantum circuit optimization.