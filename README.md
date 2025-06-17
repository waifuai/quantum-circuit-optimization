# Quantum Circuit Optimization

This repository focuses on quantum circuit data generation (using Cirq) and circuit optimization using the Google Gemini API (model: `gemini-2.5-pro`) via in-context learning. All classical neural network and local model code has been removed for simplicity and clarity.

## Project Structure

- **data-generation/**: Tools for generating datasets of random quantum circuits using Cirq.
- **model-training/**: Contains code for optimizing quantum circuits via the Google Gemini API.

### `src/data-generation`

This directory focuses on creating datasets of quantum circuits that can be used to train machine learning models.

**Key Features:**

*   **Circuit Generation:** Generates random quantum circuits using the `Cirq` library.
*   **Simulation:** Simulates circuits, optionally including noise models.
*   **Optimization:** Applies basic optimization routines to generated circuits.
*   **Dataset Creation:** Saves circuits and associated data in JSON Lines (`.jsonl`) format (one JSON object per line).

**Getting Started:**

1. Ensure you have Python 3.10+ installed.
2. Install necessary dependencies:
    ```bash
    # From the project root, after activating the virtual environment (see main Getting Started below)
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt
    ```
3. Run the dataset generation script from the project root:
    ```bash
    python src/data-generation/scripts/generate_dataset.py
    ```
    This creates a JSON Lines file (`cirq_dataset.jsonl`) containing 100 circuits by default. You can modify the script parameters to generate a different number of circuits or change the output file.

### `src/model-training`

This directory contains the core of the project, focusing on:
*   **Gemini API Inference:** Uses Google Gemini API for circuit optimization via in-context learning.

**Subdirectories:**

*   **`gemini_cli/`:** Google Gemini API CLI for circuit optimization. Contains scripts for optimizing circuits using the Gemini API.

**Getting Started:**

The `src/model_training/gemini_cli/README.md` file has detailed instructions. Generally, you will need to:

1. **Set up your Gemini API key:** Place your API key in a file named `.api-gemini` in your home directory (`~/.api-gemini`).
2. **Install dependencies:** Use `pip install -r requirements.txt` from the project root. This file includes dependencies for all modules, including `cirq`, `google-generativeai`, and `tqdm`.
3. **Run the optimization script:** Provide your input circuit string to the `predict.py` script.

**Examples:**

*   Optimize a circuit using Google Gemini API:
    ```bash
    python src/model_training/gemini_cli/predict.py --input_circuit "H 0 ; CNOT 0 1 ; H 0"
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
    python -m venv .venv
    # Activate the environment
    # On Linux/macOS:
    # source .venv/bin/activate
    # On Windows (Command Prompt/PowerShell):
    # .venv\Scripts\activate

    python -m pip install -r requirements.txt 
    ```
4. **Follow the instructions in the README files within each subdirectory** (e.g., `src/model_training/gemini_cli/README.md` when it's created).

## Future Work

This project is an ongoing effort. Potential future directions related to the current scope include:

*   **Data Generation:**
    *   Generating larger and more complex quantum circuits.
    *   Exploring different circuit structures and gate sets.
*   **Gemini Optimization:**
    *   Improving the Gemini API implementation (e.g., advanced prompt engineering, exploring different generation strategies, automated evaluation using metrics like gate count reduction or fidelity).
    *   Investigating reinforcement learning approaches using the Gemini API for feedback.
    *   Incorporating error mitigation awareness into the optimization prompts or process.