# Quantum Circuit Optimization Project

This repository contains code and resources for exploring the optimization of quantum circuits using various techniques, including simulation, machine learning, and (historically) cloud-based data processing. The project is divided into three main areas: data generation, model training, and historical SQL queries.

## Directory Structure

The repository is organized into the following top-level directories:

-   **`src/data-generation`**:  Tools for generating, simulating, and optimizing quantum circuits.
-   **`src/model-training`**:  Code for training machine learning models to predict properties of quantum circuits.
-   **`src/sql`**:  Historical SQL queries used during a previous cloud-based phase of the project (now superseded).

### `src/data-generation`

This package provides functionality to:

-   Generate random quantum circuits using Cirq.
-   Simulate these circuits, optionally including noise models.
-   Optimize the generated circuits.

The primary script in this directory, `generate_dataset.py`, can be used to create datasets of quantum circuits, which are then saved as pickle files. These datasets can be used for training machine learning models.

**Usage:**

From the project root, run:

```bash
python src/data-generation/scripts/generate_dataset.py
```

This generates a dataset of 100 circuits by default. You can adjust the number of circuits in the script.

**Dependencies:**

-   Python 3.10+
-   Cirq
-   tqdm

Install dependencies via:

```bash
pip install cirq tqdm
```

### `src/model-training`

This directory houses the code for training various machine learning models aimed at predicting quantum circuit properties, such as their statevectors. It explores different model architectures, including:

-   **Deep Neural Networks (DNNs)**
-   **Deep & Cross Networks (DCNs)**

Different training setups are also explored, including training on CPU. The directory is further divided into subdirectories for different model types and data preprocessing methods. Each subdirectory contains its own README with specific instructions.

**Subdirectories:**

*   **`prep`**: Scripts for preprocessing data (normalization, TFRecord conversion).
*   **`norm_ds_dnn`**: DNN training on normalized CSV data.
*   **`ds_dnn`**: DNN training using Keras API.
*   **`dnn/32s`**: DNN for predicting full 32-element statevectors (5 qubits).
*   **`dnn/1s`**: DNN for predicting a single statevector element.
*   **`dcn`**: DCN model training.

**Getting Started:**

Refer to the README files within each subdirectory for detailed instructions on model training, data requirements, and usage. Generally, you'll need to install dependencies like TensorFlow, pandas, and scikit-learn, and configure data and output paths.

### `src/sql`

This directory contains a collection of SQL queries that were previously used during a cloud-based phase of the project. These queries have been replaced by a local, Cirq-based implementation but are retained for historical reference.

**Files:**

-   **`blank_gates.sql`**: Replaces empty gate types with 'BLANK'.
-   **`combine_tables.sql`**: Merges data from multiple tables.
-   **`fail_replace.sql`**: Attempts to replace 'BLANK' with '1' (no validation).
-   **`flatten_1s.sql`**: Flattens dataset schema (level-1 circuits).
-   **`flatten_32s.sql`**: Flattens dataset schema (level-32 circuits).
-   **`normalize.sql`**: Normalizes state vector columns.

These queries are not actively used in the current project workflow.

## Overall Project Goals

The main objective of this project is to leverage machine learning to improve the optimization of quantum circuits. By generating datasets of circuits, simulating their behavior, and training models to predict key properties, we aim to develop methods for more efficient circuit design and optimization. The project has transitioned from a cloud-based approach to a local, Cirq-based workflow for greater flexibility and control.