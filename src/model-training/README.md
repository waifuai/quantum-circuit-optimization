# Model Training for Quantum Circuit Optimization

This directory contains code for training various machine learning models to predict properties of quantum circuits, such as their statevectors.  Different model architectures and training setups are explored, including Deep Neural Networks (DNNs), Deep & Cross Networks (DCNs), and training on CPU.

## Subdirectories

* **`prep`**: Scripts for preprocessing data, including normalization and conversion to TFRecord format.
* **`norm_ds_dnn`**:  Trains a DNN on a normalized dataset stored in CSV shards. Includes scripts for CPU training.
* **`ds_dnn`**: Trains a DNN using the Keras API. Includes scripts for CPU training.
* **`dnn/32s`**: Trains a DNN to predict the full 32-element statevector of a 5-qubit quantum circuit.
* **`dnn/1s`**: Trains a DNN to predict a single element of the statevector.
* **`dcn`**: Trains a Deep & Cross Network (DCN) model. Includes scripts for CPU training.

## Getting Started

Each subdirectory contains its own README file with detailed instructions on running the code.  Refer to these README files for specific information about the models, data requirements, and usage instructions.  Generally, you will need to install the required dependencies (e.g., TensorFlow, pandas, scikit-learn) and configure any necessary paths to data and model output directories.

