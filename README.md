# Quantum Circuit Optimization

This repository contains code and resources for exploring quantum circuit optimization using both classical and quantum machine learning techniques. The project aims to leverage the power of machine learning to discover more efficient quantum circuits for various tasks.

## Project Structure

The project is organized into three main directories:

*   **`data-generation`**: Tools for generating datasets of random quantum circuits.
*   **`model-training`**:  Code for training classical and quantum machine learning models for circuit optimization.
*   **`sql`**: Historical SQL queries (no longer actively used) from an earlier, cloud-based phase of the project.

### `data-generation`

This directory focuses on creating datasets of quantum circuits that can be used to train machine learning models.

**Key Features:**

*   **Circuit Generation:** Generates random quantum circuits using the `Cirq` library.
*   **Simulation:** Simulates circuits, optionally including noise models.
*   **Optimization:** Applies basic optimization routines to generated circuits.
*   **Dataset Creation:** Saves circuits and associated data in pickle format for use in training.

**Getting Started:**

1. Ensure you have Python 3.10+ installed.
2. Install necessary dependencies:
    ```bash
    pip install cirq tqdm
    ```
3. Run the dataset generation script from the project root:
    ```bash
    python src/data-generation/scripts/generate_dataset.py
    ```
    This creates a pickle file containing 100 circuits by default. You can modify the script to generate a different number of circuits.

### `model-training`

This directory contains the core of the project, where various machine learning models are trained for quantum circuit optimization. It explores several approaches:

*   **Classical Deep Learning (TensorFlow):** Utilizes deep neural networks (DNNs) and Deep Crossing Networks (DCNs) for regression tasks and optimization.
*   **Hybrid Classical-Quantum Optimization:** Combines DNNs with Cirq's optimization routines, where DNNs generate parameters for Cirq circuits.
*   **Sequence-to-Sequence Modeling (Trax):** Employs a Transformer model to learn transformations that optimize circuits, treating them as sequences.

**Subdirectories:**

*   **`nn/`:** Classical neural network models.
    *   **`dcn/`:** Deep Crossing Network (DCN) implementation.
    *   **`dnn/`:** Deep Neural Networks (DNN) for 1-second and 32-second circuit optimization, including hybrid DNN-Cirq optimizers.
    *   **`ds_dnn/`:** Dataset creation and preprocessing for DNN training.
    *   **`norm_ds_dnn/`:** DNN training with normalized datasets.
    *   **`utils/`:** Utility functions for circuit manipulation and model building.
*   **`prep/`:** Data preprocessing scripts for normalizing and converting CSV data to TFRecord format.
*   **`trax-train/`:** Transformer model implementation using Trax for sequence-to-sequence circuit optimization.
*   **`trax-transformer/`:** A simplified version of the Trax Transformer model.

**Models:**

*   **Deep Crossing Network (DCN):** A classical deep learning model used for regression tasks related to circuit properties.
*   **Deep Neural Networks (DNN):** DNNs are employed for optimizing 1-second and 32-second circuits, and in hybrid approaches that generate parameters for Cirq circuits.
*   **Transformer:** A sequence-to-sequence model that learns to transform unoptimized circuits into more efficient versions.

**Getting Started:**

Each module within `model-training` has its own README file with detailed instructions. Generally, you will need to:

1. **Prepare the data:** This might involve using `data-generation` to create datasets, downloading pre-existing datasets, or preprocessing CSV files from the `prep` directory.
2. **Install dependencies:** Use `pip install -r requirements.txt` within the specific module's directory, or install individual packages as needed (e.g., `pip install cirq numpy pandas scikit-learn tensorflow`). Trax modules have their own `requirements.txt` files.
3. **Run the training scripts:** Each module provides scripts for training and evaluating its models.

**Examples:**

*   Train a DCN model:
    ```bash
    python src/model-training/nn/dcn/legacy_cpu_dcn.py
    ```
*   Run a Hybrid DNN-Cirq Optimizer (for 1-second circuits):
    ```bash
    python src/model-training/nn/dnn/1s/cirq_dnn_optimizer.py
    ```
*   Train a Trax Transformer model:
    ```bash
    python src/model-training/trax-train/src/train.py
    ```

### `sql`

This directory contains SQL queries.

**Queries include:**

*   `blank_gates.sql`: Replaces empty gate types with 'BLANK'.
*   `combine_tables.sql`: Merges data from multiple tables.
*   `fail_replace.sql`: Attempts to replace 'BLANK' with '1'.
*   `flatten_1s.sql`: Flattens the dataset schema for level-1 circuits.
*   `flatten_32s.sql`: Flattens the dataset schema for level-32 circuits.
*   `normalize.sql`: Normalizes state vector columns.

## Getting Started with the Repository

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2. **Navigate to the directory of interest:**
    ```bash
    cd src/data-generation # To generate data
    cd src/model-training/nn/dcn # To train a DCN model, as an example
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt # If a requirements.txt file is present
    # Or, install packages individually
    pip install cirq numpy pandas scikit-learn tensorflow
    ```
4. **Follow the instructions in the README files within each subdirectory.**

## Future Work

This project is an ongoing effort, and there are many potential directions for future development, as outlined in `model-training/README.md`. Some key areas include:

*   Developing custom loss functions tailored for quantum circuit optimization.
*   Exploring backpropagation through quantum circuits to enable more efficient gradient-based optimization.
*   Experimenting with different model architectures and hyperparameter settings.
*   Investigating more advanced optimization algorithms, such as reinforcement learning.
*   Applying these techniques to larger and more complex quantum circuits.
*   Utilizing quantum hardware for simulation and optimization to leverage the power of real quantum devices.
*   Improving the Trax Transformer implementation for better performance and scalability.
*   Incorporating error mitigation techniques to improve the reliability of optimized circuits on noisy quantum hardware.
*   Investigating reinforcement learning as a method for circuit optimization.

This project provides a solid foundation for exploring the exciting intersection of classical and quantum machine learning for the purpose of quantum circuit optimization. It offers a starting point for researchers and developers interested in contributing to this rapidly evolving field.