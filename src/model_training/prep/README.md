# Data Preprocessing for Quantum Circuit Optimization

This directory contains scripts for preprocessing CSV data representing quantum circuits before training machine learning models. The preprocessing steps include normalization and conversion to TFRecord format.

## Scripts

- **`csv_norm.py`**:  
  Normalizes numeric features in CSV files using either shard-based or memory-based methods.  
  - The shard-based method processes shards one at a time to ensure memory efficiency.
  - The memory-based method loads the entire CSV file into memory and uses MinMaxScaler for normalization.
  - Normalizes statevectors, gate types, control/target qubits, angles, and gate numbers.
  - Output files are saved with the specified output path.
  - The script now consolidates the functionality of the old `csv_norm_new.py` and `csv_norm_old.py` scripts.
  
- **`csv_to_tfrecord.py`**:  
  Converts a preprocessed CSV file into a TFRecord file, suitable for TensorFlow training.  
  - Assumes the CSV structure has the first column as the label and remaining columns as features.
  - Includes an option to limit the number of records (useful for testing).

## Workflow

1. **Normalize the Data:**  
   Run `csv_norm.py` to normalize the numeric features. Use the `shard` method for large files and the `memory` method for smaller files.

2. **Convert to TFRecord:**  
   Use `csv_to_tfrecord.py` to convert the normalized CSV data to TFRecord format for TensorFlow model training.

## Notes

- The normalization constants in `csv_norm.py` are based on the expected value ranges.
- Command-line arguments allow you to specify input/output paths, the normalization method, and other options for flexibility.
