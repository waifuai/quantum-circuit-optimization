# Quantum Circuit Optimization

This repository focuses on quantum circuit data generation (using Cirq) and circuit optimization using LLM providers. The default provider is OpenRouter with model `openrouter/horizon-beta`. Google GenAI (Gemini) remains supported.

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
*   **LLM API Inference:** Uses OpenRouter (default) or Google Gemini for circuit optimization via in-context learning.

**Subdirectories:**

*   **`gemini_cli/`:** Google Gemini API CLI for circuit optimization. Contains scripts for optimizing circuits using the Gemini API.

**Getting Started (Unified CLI):**

1. Credentials:
   - OpenRouter: set OPENROUTER_API_KEY or put key in `~/.api-openrouter`
   - Gemini: set GEMINI_API_KEY or GOOGLE_API_KEY, or put key in `~/.api-gemini`
2. Model selection:
   - OpenRouter model file: `~/.model-openrouter` (first non-empty line); fallback `openrouter/horizon-beta`
   - Gemini model file: `~/.model-gemini` (first non-empty line); fallback `gemini-2.5-pro`
3. Install dependencies:
   ```bash
   .venv/Scripts/python.exe -m uv venv .venv ; .venv/Scripts/python.exe -m ensurepip ; .venv/Scripts/python.exe -m pip install uv ; .venv/Scripts/python.exe -m uv pip install -r requirements.txt
   ```
4. Run the unified CLI (default provider: openrouter):
   ```bash
   python -m src.model_training.cli.predict --input_circuit "H 0 ; CNOT 0 1 ; H 0"
   ```
   Use Gemini explicitly:
   ```bash
   python -m src.model_training.cli.predict --provider gemini --input_circuit "H 0 ; CNOT 0 1 ; H 0"
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
*   **Gemini Optimization (Google GenAI SDK):**
    *   Improving the prompt engineering and generation strategies; automated evaluation using metrics like gate count reduction or fidelity.
    *   Investigating reinforcement learning approaches using the Gemini API for feedback.
    *   Incorporating error mitigation awareness into the optimization prompts or process.