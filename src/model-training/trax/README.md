# Quantum Circuit Optimization with Trax

This project demonstrates how to use Google's Trax library to train a model for quantum circuit optimization. The model takes a quantum circuit as input and generates an optimized version of the circuit as output.

## Project Structure

The project is structured as follows:

*   `src/`: Contains the source code for the project.
    *   `data.py`: Contains functions for loading and preprocessing the data.
    *   `model.py`: Contains the definition of the Transformer model.
    *   `quantum_cli.py`: Contains the command-line interface for the project.
*   `tests/`: Contains unit tests for the project.
*   `requirements.txt`: Lists the required Python packages.

## Data Preparation

The `preprocess_data` function in `src/quantum_cli.py` prepares the input data for training. It removes the first and last lines from the input file.

## Model Training

The model used is a Transformer, a powerful neural network architecture well-suited for sequence-to-sequence tasks. The model is defined in `src/model.py`.

The `train_model` function in `src/quantum_cli.py` trains the model using Trax's training loop. It takes the input and output file paths, the model directory, batch size, and the number of training steps as arguments.

## Model Prediction

The `predict` function in `src/quantum_cli.py` shows how to use the trained model for prediction. It takes the model directory and the input circuit as command-line arguments and prints the optimized circuit to the console.

## Usage

1.  **Prepare your data:** Create `input.txt` and `output.txt` files containing the input and output quantum circuits, respectively. Make sure the circuits are properly formatted and aligned. Each line should represent a circuit, and the numbers should be space-separated.
2.  **Preprocess the data:** Run the `prep` command to preprocess the data:
    ```bash
    python src/quantum_cli.py prep --input_file input.txt --input_processed_file input_processed.txt --output_processed_file output_processed.txt
    ```
3.  **Install the dependencies:** `pip install --user -r requirements.txt`
4.  **Train the model:** Run the `train` command to train the model:
    ```bash
    python src/quantum_cli.py train --input_file input_processed.txt --output_file output_processed.txt --model_dir model --batch_size 64 --n_steps 1000
    ```
5.  **Predict using the trained model:** Run the `predict` command to predict using the trained model:
    ```bash
    python src/quantum_cli.py predict --model_dir model --input_circuit "1 2 3 4"
    ```
    (replace "1 2 3 4" with your input circuit)

## Testing

The project includes a test suite in the `tests` directory. To run the tests, execute:

```bash
python -m unittest discover -s src/model-training/trax/tests -p 'test_*.py'
```

## Note

This project provides a basic framework for quantum circuit optimization using Trax. You can adapt and extend it to suit your specific needs and datasets. For example, you can experiment with different model architectures, hyperparameters, and data preprocessing techniques to improve the performance of the model. This project is designed to run locally on CPU without any web server or GPU/TPU dependencies.