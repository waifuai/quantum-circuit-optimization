import datetime
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped directories for logs and models."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)

def preprocess_data(csv_file_path: str, n_qubits: int):
    """Load and preprocess CSV data for statevector regression."""
    data = pd.read_csv(csv_file_path)
    data = data.replace(['U3Gate', 'CnotGate', 'Measure', 'BLANK'], [1, 2, 3, 4])
    
    dense_features = []
    for i in range(41):
        prefix = f"gate_{str(i).zfill(2)}_"
        dense_features.extend([f"{prefix}{suffix}" for suffix in 
                               ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]])
    
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])
    
    statevectors = [bin(i)[2:].zfill(n_qubits) for i in range(2**n_qubits)]
    target = [f"statevector_{sv}" for sv in statevectors]
    return data[dense_features].values, data[target].values, target

def load_or_build_model(input_shape: int):
    """Attempt to load a pretrained model; if unavailable, build a new one."""
    model_path = 'pretrained_model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        print("Pretrained model loaded successfully.")
    except (ImportError, OSError):
        print("Pretrained model not found. Creating a new model.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32)  # 32 statevector outputs
        ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def main():
    TOPIC = "qc_dnn_1m_32s_resume"
    EPOCHS = int(1e3)
    CSV_FILE_PATH = "qc5i32f_8x_9x.csv"
    LOGDIR = "./logs/fit/"
    MODELDIR = "./models/"
    N_QUBITS = 5
    
    save_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/" + TOPIC
    log_dir = os.path.join(LOGDIR, save_path)
    model_dir = os.path.join(MODELDIR, save_path)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("Logging to", log_dir)
    print("Saving best model to", model_dir)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "model.h5"),
        monitor='val_mse',
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    
    X, y, target_cols = preprocess_data(CSV_FILE_PATH, N_QUBITS)
    model = load_or_build_model(input_shape=X.shape[1])
    
    model.fit(
        X,
        y,
        batch_size=1024,
        epochs=EPOCHS,
        verbose=2,
        validation_split=0.1,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )

if __name__ == "__main__":
    main()
