# Model Training and Prediction Pipeline

## Overview

This repository provides a simple framework for training a machine learning model, generating predictions, and handling data pipelines. The project is split into three main components:

- **Prediction Module**: Loads a model and produces optimized circuit predictions.
- **Problem Module**: Constructs data pipelines and instantiates the model.
- **Training Module**: Executes the model training loop and saves the trained model.

Unit tests have been written for each of these components using Python’s built-in `unittest` framework to ensure proper functionality.

## Project Structure

```
.
├── src
│   ├── predict.py           # Contains the 'predict' function for circuit optimization.
│   ├── train.py             # Contains the 'train_model' function for training the model.
│   └── trainer
│       └── problem.py       # Contains 'get_data_pipelines' and 'get_model' functions.
├── test_predict.py          # Unit tests for the predict module.
├── test_problem.py          # Unit tests for the problem module.
└── test_train.py            # Unit tests for the training module.
```

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows use: env\Scripts\activate
   ```

3. **Install dependencies:**

   If you have a `requirements.txt` file, run:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, ensure you have Python 3 installed along with any necessary packages (e.g. `numpy`).

## Running Tests

The project uses Python’s built-in `unittest` framework. To run all tests, execute:

```bash
python -m unittest discover
```

Alternatively, you can run individual test files:

```bash
python test_predict.py
python test_problem.py
python test_train.py
```

## Modules Overview

### Predict Module (`src/predict.py`)

This module provides the `predict` function that:
- Loads a model from a specified directory.
- Processes an input circuit.
- Returns an optimized circuit prediction.

The test file `test_predict.py` ensures that the function runs without errors (even with a dummy model).

### Problem Module (`src/trainer/problem.py`)

This module offers:
- **`get_data_pipelines`**: Creates data pipelines from input and output files with a given batch size.
- **`get_model`**: Instantiates and returns a new model.

Tests in `test_problem.py` verify that:
- The data pipelines yield correctly shaped NumPy arrays.
- The model is instantiated successfully.

### Training Module (`src/train.py`)

The training module contains the `train_model` function which:
- Runs a short training loop.
- Saves the trained model to the provided directory.

The test file `test_train.py` confirms that:
- The training routine completes without error.
- A model file is created in the specified directory.
