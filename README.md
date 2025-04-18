# Quantum Circuit Optimization

This repository focuses on quantum circuit data generation (using Cirq) and circuit optimization using the Google Gemini API (model: gemini-2.5-flash-preview-04-17) via in-context learning. All classical neural network and local model code has been removed for simplicity and clarity.

## Project Structure

- **data-generation/**: Tools for generating datasets of random quantum circuits using Cirq.
- **model-training/**: Contains code for optimizing quantum circuits via the Google Gemini API.

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
*   **Gemini API Inference:** Uses Google Gemini API for circuit optimization via in-context learning.

**Subdirectories:**

*   **`nn/`:** Classical neural network models.
    *   **`dcn/`:** Deep Crossing Network (DCN) implementation.
    *   **`dnn/`:** Deep Neural Networks (DNN) for 1-second and 32-second circuit optimization, including hybrid DNN-Cirq optimizers.
    *   **`ds_dnn/`:** Dataset creation and preprocessing for DNN training.
    *   **`norm_ds_dnn/`:** DNN training with normalized datasets.
    *   **`utils/`:** Utility functions for circuit manipulation and model building.
*   **`prep/`:** Data preprocessing scripts for normalizing and converting CSV data to TFRecord format (primarily for `nn` models).
*   **`gemini_cli/`:** Google Gemini API CLI for circuit optimization. Contains scripts for optimizing circuits using the Gemini API.

**Models:**

*   **Deep Crossing Network (DCN):** A classical deep learning model used for regression tasks related to circuit properties.
*   **Deep Neural Networks (DNN):** DNNs are employed for optimizing 1-second and 32-second circuits, and in hybrid approaches that generate parameters for Cirq circuits.

**Getting Started:**

Each module within `model-training` has its own README file with detailed instructions. Generally, you will need to:

1. **Prepare the data:** This might involve using `data-generation` to create datasets, downloading pre-existing datasets, preprocessing CSV files from the `prep` directory (for `nn` models), or preparing text files for the `gemini_cli` module.
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
*   Optimize a circuit using Google Gemini API:
    ```bash
    python src/model_training/gemini_cli/predict.py --input_circuit "H 0; CNOT 0 1; H 0"
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
4. **Follow the instructions in the README files within each subdirectory** (e.g., `src/model_training/gemini_cli/README.md` when it's created).

## Future Work

This project is an ongoing effort, and there are many potential directions for future development, as outlined in `model-training/README.md`. Some key areas include:

*   Developing custom loss functions tailored for quantum circuit optimization.
*   Exploring backpropagation through quantum circuits to enable more efficient gradient-based optimization.
*   Experimenting with different model architectures and hyperparameter settings.
*   Investigating more advanced optimization algorithms, such as reinforcement learning.
*   Applying these techniques to larger and more complex quantum circuits.
*   Utilizing quantum hardware for simulation and optimization to leverage the power of real quantum devices.
*   Improving the Gemini API implementation (e.g., exploring different architectures, generation strategies, evaluation metrics like BLEU/ROUGE).
*   Incorporating error mitigation techniques to improve the reliability of optimized circuits on noisy quantum hardware.
*   Investigating reinforcement learning as a method for circuit optimization.

This project provides a solid foundation for exploring the exciting intersection of classical and quantum machine learning for the purpose of quantum circuit optimization. It offers a starting point for researchers and developers interested in contributing to this rapidly evolving field.