# Data Preprocessing for Quantum Circuit Optimization

This directory contains scripts for preprocessing CSV data representing quantum circuits before training machine learning models. The preprocessing steps include normalization and conversion to TFRecord format.

## Scripts

- **`csv_norm_new.py`**:  
  Normalizes numeric features in a directory of CSV shards using universal normalization constants.  
  - Processes shards one at a time to ensure memory efficiency.
  - Normalizes statevectors, gate types, control/target qubits, angles, and gate numbers.
  - Output files are saved with the suffix `_output.csv`.
  
- **`csv_norm_old.py`**:  
  An older script for normalization that loads the entire CSV file into memory.  
  - Uses `MinMaxScaler` for normalization.
  - Kept for reference and not recommended for large datasets.
  
- **`csv_to_tfrecord.py`**:  
  Converts a preprocessed CSV file into a TFRecord file, suitable for TensorFlow training.  
  - Assumes the CSV structure has the first column as the label and remaining columns as features.
  - Includes an option to limit the number of records (useful for testing).

## Workflow

1. **Shard the Large CSV File (if necessary):**  
   If your dataset is very large, split it into smaller CSV shards.  
   Use `csv_norm_new.py` to process shards efficiently.

2. **Normalize the Data:**  
   Run `csv_norm_new.py` (or `csv_norm_old.py` for smaller datasets) to normalize the numeric features.

3. **Convert to TFRecord:**  
   Use `csv_to_tfrecord.py` to convert the normalized CSV data to TFRecord format for TensorFlow model training.

## Notes

- The normalization constants in `csv_norm_new.py` are based on the expected value ranges.
- Command-line arguments allow you to specify input/output paths and other options for flexibility.
