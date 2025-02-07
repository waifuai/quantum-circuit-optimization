# Quantum Circuit Optimization with Trax

This project demonstrates how to use Google's Trax library to train a model for quantum circuit optimization. The model takes a quantum circuit as input and generates an optimized version of the circuit as output.

## Problem Definition

The core of this project is the definition of a Trax data processing pipeline and model. This is done in `src/trainer/problem.py`. The `quantum_circuit_data_generator` function reads the input and output files, and `get_data_pipelines` creates the data pipelines for training and evaluation. The `transformer_model` function defines the Transformer model using Trax's layers.

## Data Preparation

The `scripts/prep.sh` script prepares the input data for training. It performs the following steps:

1. Removes the indenting character from the beginning of each line in the input file.
2. Removes the last line of the modified file to create the input file for the model.
3. Removes the first line of the modified file to create the output file for the model.

## Model Training

The model used is a Transformer, a powerful neural network architecture well-suited for sequence-to-sequence tasks. The hyperparameters for the model are defined in `src/trainer/problem.py`.

The `src/train.py` script trains the model using Trax's training loop. It takes the input and output file paths, the model directory, batch size, and the number of training steps as arguments.

## Model Prediction

The `src/predict.py` script shows how to use the trained model for prediction. It takes the model directory and the input circuit as command-line arguments and prints the optimized circuit to the console.

## Usage

1. **Prepare your data:**  Create  `input.txt`  and  `output.txt`  files containing the input and output quantum circuits, respectively. Make sure the circuits are properly formatted and aligned. Each line should represent a circuit, and the numbers should be space-separated.
2. **Run the data preparation script:**  `bash scripts/prep.sh`
3. **Install the dependencies:**  `pip install --user -r requirements.txt`
4. **Train the model:**  `python src/train.py --input_file=input_processed.txt --output_file=output_processed.txt --model_dir=model`
5. **Predict using the trained model:**  `python src/predict.py model "1 2 3 4"`  (replace "1 2 3 4" with your input circuit)

## Testing

The project includes a test suite in the `tests` directory. To run the tests:

1. Navigate to the project's root directory.
2. Run the  `run_tests.sh`  script:  `bash run_tests.sh`

## Note

This project provides a basic framework for quantum circuit optimization using Trax. You can adapt and extend it to suit your specific needs and datasets. For example, you can experiment with different model architectures, hyperparameters, and data preprocessing techniques to improve the performance of the model. This project is designed to run locally on CPU without any web server or GPU/TPU dependencies.