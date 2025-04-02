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
*   **Sequence-to-Sequence Modeling (Hugging Face Transformers):** Employs a Transformer model (Encoder-Decoder architecture) to learn transformations that optimize circuits, treating them as sequences of tokens.

**Subdirectories:**

*   **`nn/`:** Classical neural network models.
    *   **`dcn/`:** Deep Crossing Network (DCN) implementation.
    *   **`dnn/`:** Deep Neural Networks (DNN) for 1-second and 32-second circuit optimization, including hybrid DNN-Cirq optimizers.
    *   **`ds_dnn/`:** Dataset creation and preprocessing for DNN training.
    *   **`norm_ds_dnn/`:** DNN training with normalized datasets.
    *   **`utils/`:** Utility functions for circuit manipulation and model building.
*   **`prep/`:** Data preprocessing scripts for normalizing and converting CSV data to TFRecord format (primarily for `nn` models).
*   **`hf_transformer/`:** Transformer model implementation using Hugging Face Transformers for sequence-to-sequence circuit optimization. Contains scripts for building a custom tokenizer, preparing data, training, and prediction.

**Models:**

*   **Deep Crossing Network (DCN):** A classical deep learning model used for regression tasks related to circuit properties.
*   **Deep Neural Networks (DNN):** DNNs are employed for optimizing 1-second and 32-second circuits, and in hybrid approaches that generate parameters for Cirq circuits.
*   **Transformer (Hugging Face):** A sequence-to-sequence model built using the Hugging Face `transformers` library. It learns to transform unoptimized circuits into more efficient versions using a custom vocabulary and an Encoder-Decoder architecture.

**Getting Started:**

Each module within `model-training` has its own README file with detailed instructions. Generally, you will need to:

1. **Prepare the data:** This might involve using `data-generation` to create datasets, downloading pre-existing datasets, preprocessing CSV files from the `prep` directory (for `nn` models), or preparing text files for the `hf_transformer` module.
2. **Install dependencies:** Use `pip install -r requirements.txt` from the project root. This file includes dependencies for all modules, including `tensorflow`, `cirq`, `transformers`, `datasets`, and `tokenizers`.
3. **Run the training scripts:** Each module provides scripts for training and evaluating its models.

**Examples:**

*   Train a DCN model:
    ```bash
    python src/model-training/nn/dcn/legacy_cpu_dcn.py
    ```
*   Run a Hybrid DNN-Cirq Optimizer (for 1-second circuits):
    ```bash
    python src/model-training/nn/dnn/1s/hybrid_dnn_cirq_optimizer.py
    ```
*   Train the Hugging Face Transformer model (after preparing data and tokenizer):
    ```bash
    # Assuming you are in the project root
    # 1. Build tokenizer (if not already done)
    python src/model_training/hf_transformer/cli.py tokenize --data_dir=src/model_training/hf_transformer/data --tokenizer_save_dir=src/model_training/hf_transformer/tokenizer
    # 2. Prepare data (if not already done)
    python src/model_training/hf_transformer/cli.py prep --input_file=src/model_training/hf_transformer/data/input.txt --processed_input_file=src/model_training/hf_transformer/data/input_processed.txt --processed_output_file=src/model_training/hf_transformer/data/output_processed.txt
    # 3. Train
    python src/model_training/hf_transformer/cli.py train --data_dir=src/model_training/hf_transformer/data --tokenizer_dir=src/model_training/hf_transformer/tokenizer --output_dir=src/model_training/hf_transformer/hf_transformer_results
    ```

## Getting Started with the Repository

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2. **Navigate to the project root directory:**
    ```bash
    cd quantum-circuit-optimization
    ```
3. **Install dependencies:**
    ```bash
    # Recommended: Create and activate a virtual environment first
    # python -m venv .venv
    # source .venv/bin/activate  # or .venv\Scripts\activate on Windows

    # Install using uv (as per custom instructions)
    python -m uv venv .venv # Create venv if not already done
    # Activate: source .venv/Scripts/activate (or equivalent for your shell)
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt
    ```
4. **Follow the instructions in the README files within each subdirectory** (e.g., `src/model_training/hf_transformer/README.md` when it's created).

## Future Work

This project is an ongoing effort, and there are many potential directions for future development, as outlined in `model-training/README.md`. Some key areas include:

*   Developing custom loss functions tailored for quantum circuit optimization.
*   Exploring backpropagation through quantum circuits to enable more efficient gradient-based optimization.
*   Experimenting with different model architectures and hyperparameter settings.
*   Investigating more advanced optimization algorithms, such as reinforcement learning.
*   Applying these techniques to larger and more complex quantum circuits.
*   Utilizing quantum hardware for simulation and optimization to leverage the power of real quantum devices.
*   Improving the Hugging Face Transformer implementation (e.g., exploring different architectures, generation strategies, evaluation metrics like BLEU/ROUGE).
*   Incorporating error mitigation techniques to improve the reliability of optimized circuits on noisy quantum hardware.
*   Investigating reinforcement learning as a method for circuit optimization.

This project provides a solid foundation for exploring the exciting intersection of classical and quantum machine learning for the purpose of quantum circuit optimization. It offers a starting point for researchers and developers interested in contributing to this rapidly evolving field.