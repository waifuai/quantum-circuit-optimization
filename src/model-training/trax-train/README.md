# Quantum Circuit Optimization with Transformer

This project uses a Transformer model implemented in Trax to optimize quantum circuits. The model learns to transform unoptimized quantum circuits into more efficient versions.

## Project Structure

The project is organized into the following directories and files:

-   **`src/`**: Contains the source code for the project.
    -   **`prep.py`**:  Preprocesses the raw data by cleaning up whitespace and creating input/output pairs suitable for the Transformer model.
    -   **`train.py`**: Defines the Transformer model architecture, training loop, and associated functions for training the model.
    -   **`predict.py`**:  Contains functions for loading a trained model and performing inference (circuit optimization) using greedy decoding or beam search. Also handles prediction from a file or interactive input.
    -   **`utils.py`**: Provides utility functions for loading data, creating a vocabulary, and creating data generators for training.
-   **`data/`**: Directory for storing the quantum circuit data. You should place your `input.txt` (unoptimized circuits) and `output.txt` (optimized circuits) files here. The `prep.py` script will generate the `input_processed.txt` and `output_processed.txt` in this directory. `phrases_input.txt` will also be in this directory.
-   **`model/`**: Directory for saving trained model checkpoints.
-   **`requirements.txt`**: Lists the required Python packages for the project.

## Getting Started

### Prerequisites

1. **Python 3.7+**: Ensure you have Python 3.7 or a later version installed.
2. **Virtual Environment (Recommended)**: Create a virtual environment to isolate project dependencies:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install Dependencies**: Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1. **Place Data Files**: Put your `input.txt` and `output.txt` files in the `data/` directory. Each line in `input.txt` should represent an unoptimized quantum circuit, and the corresponding line in `output.txt` should be its optimized version.
2. **Preprocess Data**: Run the `prep.py` script to prepare the data for training:

    ```bash
    python src/prep.py
    ```

    This will create `input_processed.txt` and `output_processed.txt` in the `data/` directory, which will be used for training.
3. **Place Prediction Input File**: Put your `phrases_input.txt` file in the `data/` directory. Each line should represent a circuit to be optimized.

### Training

1. **Train the Model**: Run the `train.py` script to start the training process:

    ```bash
    python src/train.py
    ```

    This will train the Transformer model and save checkpoints in the `model/` directory. You can adjust hyperparameters in `train.py` if needed.

### Prediction (Inference)

1. **Run Prediction**: Use the `predict.py` script to optimize quantum circuits. You can choose to predict from a file or interactively:

    -   **From File**:
        ```bash
        python src/predict.py
        # Choose 'f' for file input
        # Enter beam width (e.g., 1 for greedy decoding, or a larger value for beam search)
        ```

        This will read circuits from `data/phrases_input.txt`, optimize them, and print the results to the console.

    -   **Interactively**:
        ```bash
        python src/predict.py
        # Choose 'i' for interactive input
        # Enter beam width (e.g., 1 for greedy decoding, or a larger value for beam search)
        # Enter a quantum circuit (or 'q' to quit)
        ```

        This will allow you to enter circuits one by one and see the optimized versions.

## Hyperparameters

You can adjust the following hyperparameters in `train.py` and `predict.py`:

-   **`BATCH_SIZE`**: The batch size for training and prediction.
-   **`MAX_LENGTH`**: The maximum length of input and output sequences.
-   **`N_LAYERS`**: The number of encoder and decoder layers in the Transformer.
-   **`D_MODEL`**: The dimensionality of the model's internal representations.
-   **`D_FF`**: The dimensionality of the feed-forward network in each Transformer layer.
-   **`N_HEADS`**: The number of attention heads in each Transformer layer.
-   **`N_TRAIN_STEPS`**: The total number of training steps.
-   **`EVAL_STEPS`**: The number of steps between evaluations during training.
-   **`LEARNING_RATE`**: The learning rate for the optimizer.
-   **`beam_width`**: (In `predict.py`) The beam width for beam search decoding. Use 1 for greedy decoding.

## Notes

-   The provided code assumes that quantum circuits are represented as strings of characters. You may need to modify the code if your circuits have a different representation.
-   Training a Transformer model can be computationally intensive. You might need to use a GPU for faster training.
-   The performance of the model depends on the quality and quantity of the training data.
-   The greedy decoding is deterministic, beam search is not.
