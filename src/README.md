# src

This repository contains code and resources for exploring quantum circuit optimization using both classical and quantum machine learning techniques. The project is organized into three main directories: `data-generation`, `model-training`, and `sql`.

## `data-generation`

This directory provides tools for generating datasets of random quantum circuits.

**Key Features:**

*   **Circuit Generation:** Generates random quantum circuits using the Cirq library.
*   **Simulation:** Simulates the generated circuits, optionally including noise models.
*   **Optimization:** Applies basic optimization routines to the circuits.
*   **Dataset Creation:** Saves the generated circuits and associated data into a pickle file for use in training machine learning models.

**Dependencies:**

*   Python 3.10+
*   Cirq
*   tqdm

**Usage:**

To generate a dataset, run the following command from the project root:

```bash
python src/data-generation/scripts/generate_dataset.py
```

This will create a pickle file containing 100 circuits (by default). The number of circuits can be adjusted in the script.

## `model-training`

This directory focuses on training machine learning models for quantum circuit optimization. It explores various approaches, including classical deep learning with TensorFlow and sequence-to-sequence modeling with Trax.

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
*   **Deep Neural Networks (DNN):** DNNs used for optimizing 1-second and 32-second circuits. Includes hybrid classical-quantum approaches where DNNs generate parameters for Cirq circuits.
*   **Transformer:** A sequence-to-sequence model that learns to transform unoptimized circuits into more efficient versions.

**Dependencies:**

*   Vary depending on the specific module.
*   Common dependencies include: Cirq, NumPy, Pandas, scikit-learn, TensorFlow.
*   Trax modules require additional packages listed in their respective `requirements.txt` files.

**Usage:**

Each module within `model-training` has its own README file with detailed instructions. Generally, you will need to:

1. **Prepare the data:** This may involve generating circuits using `data-generation`, downloading pre-existing datasets, or preprocessing CSV files.
2. **Install dependencies:** Use `pip install` to install the required packages for the specific module you are working with.
3. **Run the training scripts:** Each module has scripts for training and evaluating models.

**Examples:**

*   Train DCN: `python src/model-training/nn/dcn/legacy_cpu_dcn.py`
*   Run Hybrid DNN-Cirq Optimizer (1s): `python src/model-training/nn/dnn/1s/cirq_dnn_optimizer.py`
*   Train Trax Transformer: `python src/model-training/trax-train/src/train.py`

## `sql`

This directory contains SQL queries.

**Queries:**

*   **`blank_gates.sql`:** Replaces empty gate types with 'BLANK'.
*   **`combine_tables.sql`:** Merges data from multiple tables.
*   **`fail_replace.sql`:** Attempts to replace 'BLANK' with '1'.
*   **`flatten_1s.sql`:** Flattens the dataset schema for level-1 circuits.
*   **`flatten_32s.sql`:** Flattens the dataset schema for level-32 circuits.
*   **`normalize.sql`:** Normalizes state vector columns.

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    ```

2. **Navigate to the directory of interest:**

    ```bash
    cd src/data-generation  # For data generation
    cd src/model-training/nn/dcn  # For training a DCN model, for example
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt  # If a requirements.txt file is present
    # Or, install individual packages as needed
    pip install cirq numpy pandas scikit-learn tensorflow
    ```

4. **Follow the instructions in the README files within each subdirectory.**

## Future Work

The `model-training/README.md` outlines a number of potential future directions for this project, including:

*   Developing custom loss functions for quantum circuit optimization.
*   Exploring backpropagation through quantum circuits.
*   Experimenting with different model architectures and hyperparameters.
*   Investigating more advanced optimization algorithms.
*   Applying these techniques to larger and more complex circuits.
*   Using quantum hardware for simulation and optimization.
*   Improving the Trax Transformer implementation.
*   Incorporating error mitigation techniques.
*   Investigating reinforcement learning for circuit optimization.

This project provides a foundation for exploring the intersection of classical and quantum machine learning for quantum circuit optimization.