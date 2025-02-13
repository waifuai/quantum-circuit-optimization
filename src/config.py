# src/config.py

# Global Configuration Parameters

# Data
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
DATA_AUGMENTATION = True
DATA_DIR = "shards/"
CSV_PATTERN = "*.csv"

# Logging and Model Saving
LOG_DIR = "logs/fit/"
MODEL_DIR = "models/"

# DCN Specific
TOPIC = "qc_dcn_1k_qc_cpu_tf2"
HISTOGRAM_FREQ = 1
NUM_CIRCUIT_PARAMS = 25
CSV_FILE_PATH = "./qc5f_1k_2.csv"

# DNN Specific
EPOCHS = int(1e3)
NUM_QUBITS = 5
NUM_PARAMS = 25
CIRCUIT_TYPE = 'default'

# Trax Specific
TRAX_BATCH_SIZE = 64
TRAX_N_STEPS = 1000