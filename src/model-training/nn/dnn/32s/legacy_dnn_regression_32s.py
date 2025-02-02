import datetime
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped log and model directories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)

def load_and_preprocess_data(csv_file_path: str):
    """Load CSV data, replace categorical values, and scale features."""
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        exit()

    data = data.replace(["U3Gate", "CnotGate", "Measure", "BLANK"], [1, 2, 3, 4])
    
    # Generate dense feature names
    dense_features = []
    for i in range(41):
        prefix = f"gate_{str(i).zfill(2)}_"
        dense_features.extend([f"{prefix}{suffix}" for suffix in 
                               ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]])
    
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])
    return data, dense_features

def build_dnn_model(input_dim: int, output_units: int = 32):
    """Build a sequential DNN model with 5 hidden layers of 128 units."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(output_units)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model

def main():
    TOPIC = "qc_dnn_8m_32s"
    HISTOGRAM_FREQ = 1
    EPOCHS = 100
    CSV_FILE_PATH = "./prep/qc7_8m.csv"
    LOGDIR = "logs/fit/"
    MODELDIR = "models/"

    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(MODELDIR, exist_ok=True)
    
    log_dir, model_dir = create_save_paths(LOGDIR, MODELDIR, TOPIC)
    print("Logging to", log_dir)
    print("Saving best model to", model_dir)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=HISTOGRAM_FREQ)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
    )
    
    data, dense_features = load_and_preprocess_data(CSV_FILE_PATH)
    
    # Prepare target columns for 5 qubits (32 statevector values)
    n_qubits = 5
    target_columns = [f"statevector_{bin(i)[2:].zfill(n_qubits)}" for i in range(2**n_qubits)]
    
    model = build_dnn_model(input_dim=len(dense_features), output_units=32)
    
    history = model.fit(
        x=data[dense_features],
        y=data[target_columns],
        batch_size=256,
        epochs=EPOCHS,
        verbose=1,
        validation_split=0.01,
        callbacks=[tensorboard_callback, model_checkpoint],
    )
    
    print(f"Model training complete. Logs saved to: {log_dir}")
    print(f"Best model saved to: {model_dir}")
    print("To view TensorBoard logs, run: tensorboard --logdir logs/fit")

if __name__ == "__main__":
    main()
