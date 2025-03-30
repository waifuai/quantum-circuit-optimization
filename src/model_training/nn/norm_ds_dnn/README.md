# Normalized Dataset DNN

This repository contains Python scripts for training a deep neural network (DNN) using a normalized dataset stored in CSV shards on a CPU.

## Overview

- **Script:** `cpu_norm_ds_dnn.py`  
  Trains the DNN on a CPU using TensorFlow's `tf.keras` API.
  
- **Dataset:**  
  CSV shards without headers. The dataset should be normalized and placed in the specified directory.

## Getting Started

1. **Dataset Setup:**
   - Place your headerless, normalized CSV shards in the directory specified by the `DATA_DIR` variable (default: `data/`).

2. **Configuration:**
   - Adjust hyperparameters such as `BATCH_SIZE`, `BUFFER_SIZE`, `EPOCHS`, and `VALIDATION_SPLIT` in the script as needed.
   - Modify the `DATA_DIR` if your CSV files are stored in a different location.

3. **Running the Script:**
   - Execute the script from the command line:
     ```bash
     python cpu_norm_ds_dnn.py
     ```
   - The script automatically sets up directories for TensorBoard logs and model checkpointing.

## Details

- **Data Preparation:**
  - The `labeler` function processes each CSV line, splitting the values and separating features and labels.
  - The dataset is built using the `tf.data` API for efficient loading and batching.

- **Callbacks:**
  - **TensorBoard:** For logging training progress.
  - **ModelCheckpoint:** For saving the best model based on the validation mean squared error (MSE).

- **Model Loading:**
  - A pre-trained model is loaded from the path specified by `MODEL_LOAD_PATH` before training begins.

## Dependencies

- Python (3.x)
- TensorFlow
- Standard libraries: `os`, `glob`, `datetime`
