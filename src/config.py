# src/config.py

# --- Global Defaults & Common Settings ---
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = int(1e3)
DEFAULT_NUM_QUBITS = 5
DEFAULT_NUM_PARAMS = 25 # Often 5 layers * 5 qubits
DEFAULT_HISTOGRAM_FREQ = 1

# --- Data Settings ---
BUFFER_SIZE = int(5e4) # Used by norm_ds_dnn, ds_dnn?
VALIDATION_SPLIT = 0.1 # Used by norm_ds_dnn, ds_dnn?
DATA_AUGMENTATION = True # Used in ds_dnn
DATA_DIR = "shards/" # Used by norm_ds_dnn, ds_dnn?
CSV_PATTERN = "*.csv" # Used by norm_ds_dnn

# --- Logging and Model Saving ---
LOG_DIR = "logs/fit/"
MODEL_DIR = "models/"
# MODEL_LOAD_PATH needs definition if norm_ds_dnn is to be used.
# Example: NORM_DS_DNN_MODEL_LOAD_PATH = "models/some_pretrained_model" 

# --- Model Specific Settings ---

# DCN Specific
DCN_TOPIC = "qc_dcn_1k_qc_cpu_tf2"
DCN_CSV_FILE_PATH = "./qc5f_1k_2.csv"
DCN_NUM_CIRCUIT_PARAMS = DEFAULT_NUM_PARAMS # Seems consistent
DCN_BATCH_SIZE = DEFAULT_BATCH_SIZE # Uses default
DCN_EPOCHS = DEFAULT_EPOCHS # Uses default
DCN_HISTOGRAM_FREQ = DEFAULT_HISTOGRAM_FREQ # Uses default

# DNN General (Can be overridden by specific models below)
DNN_CIRCUIT_TYPE = 'default'
DNN_NUM_QUBITS = DEFAULT_NUM_QUBITS
DNN_NUM_PARAMS = DEFAULT_NUM_PARAMS

# DNN 1s Specific
DNN_1S_BATCH_SIZE = 4096
DNN_1S_EPOCHS = DEFAULT_EPOCHS # Uses default
DNN_1S_CSV_FILE_PATH = "./qc8m_1s.csv"
DNN_1S_TOPIC = "qc_dnn_8m_1s" # From legacy

# DNN 32s Specific
DNN_32S_BATCH_SIZE = 256
DNN_32S_EPOCHS = 100 # Specific override
DNN_32S_CSV_FILE_PATH = "./prep/qc7_8m.csv"
DNN_32S_TOPIC = "qc_dnn_8m_32s" # From legacy

# DNN 32s Resume Specific (Legacy)
DNN_32S_RESUME_BATCH_SIZE = 1024 # From legacy script, not config file
DNN_32S_RESUME_EPOCHS = DEFAULT_EPOCHS # Uses default
DNN_32S_RESUME_CSV_FILE_PATH = "qc5i32f_8x_9x.csv"
DNN_32S_RESUME_TOPIC = "qc_dnn_1m_32s_resume"

# Norm DS DNN Specific
# Uses BUFFER_SIZE, VALIDATION_SPLIT, DATA_DIR, CSV_PATTERN, LOG_DIR, MODEL_DIR
NORM_DS_DNN_BATCH_SIZE = DEFAULT_BATCH_SIZE # Uses default
NORM_DS_DNN_EPOCHS = DEFAULT_EPOCHS # Uses default
# NORM_DS_DNN_MODEL_LOAD_PATH = "path/to/model" # Define this path

# Trax Specific
TRAX_BATCH_SIZE = 64
TRAX_N_STEPS = 1000