# src

This directory contains the main source code for quantum circuit optimization. The codebase is now focused on two subdirectories:
- `data-generation/`: Tools for generating datasets of random quantum circuits using Cirq.
- `model-training/`: Code for optimizing quantum circuits using the Google Gemini API (`gemini-2.5-flash-preview-04-17`).

## `data-generation`

This directory provides tools for generating datasets of random quantum circuits.

**Key Features:**

*   **Circuit Generation:** Generates random quantum circuits using the Cirq library.
*   **Simulation:** Simulates the generated circuits, optionally including noise models.
*   **Optimization:** Applies basic optimization routines to the circuits.
*   **Dataset Creation:** Saves the generated circuits and associated data into TFRecord format.

**Dependencies:**

*   Python 3.10+
*   Cirq
*   tqdm
*   TensorFlow (for TFRecord I/O only)

**Usage:**

To generate a dataset, run the following command from the project root:

```bash
python src/data-generation/scripts/generate_dataset.py
```

This will create a TFRecord file (`cirq_dataset.tfrecord`) containing 100 circuits (by default). The number of circuits and output file can be adjusted via script arguments.

## `model-training`

This directory focuses on optimizing quantum circuits using the Google Gemini API via in-context learning.

**Subdirectories:**

*   **`gemini_cli/`:** Contains a command-line script (`predict.py`) to send circuits to the Gemini API for optimization.
*   **`gemini_optimizer.py`:** Provides the core function (`optimize_circuit_with_gemini`) that handles API interaction.

**Dependencies:**

*   `google-generativeai`
*   (See main project `requirements.txt`)

**Usage:**

Refer to the `src/model_training/README.md` and `src/model_training/gemini_cli/README.md` for detailed instructions on setting up the API key and running the optimization script.

Example:
```bash
python src/model_training/gemini_cli/predict.py --input_circuit "H 0 ; CNOT 0 1 ; H 0"
```

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    ```

2. **Navigate to the directory of interest:**

    ```bash
    cd src/data-generation  # For data generation
    cd src/model-training  # For Gemini API optimization
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt  # If a requirements.txt file is present
    # Or, install individual packages as needed
    pip install cirq google-generativeai
    ```

4. **Follow the instructions in the README files within each subdirectory.**

## Future Work

Refer to the `src/model-training/README.md` for potential future work related to Gemini API optimization.