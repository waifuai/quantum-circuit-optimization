import datetime
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from dnn.1s import config_legacy
from dnn.1s.utils_legacy import create_save_paths

def load_and_preprocess_data(csv_file_path: str):
    """Load CSV data, replace categorical values, and scale dense features.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        tuple: X, y (features and labels)
    """
    data = pd.read_csv(csv_file_path)
    data = data.replace(['U3Gate', 'CnotGate', 'Measure', 'BLANK'], [1, 2, 3, 4])
    
    # Generate dense feature names
    dense_features = []
    for i in range(41):
        prefix = f"gate_{str(i).zfill(2)}_"
        dense_features.extend([prefix + suffix for suffix in 
                               ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]])
    
    # Normalize features
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])
    
    target = ['statevector_00000']
    X = data[dense_features].values
    y = data[target].values
    return X, y

def build_dnn_model(input_dim: int):
    """Create a Keras model using the Functional API.

    Args:
        input_dim (int): Dimension of the input layer.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    for _ in range(4):  # total five Dense layers of 128 units
        x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def create_callbacks(log_dir, histogram_freq):
    """Create callbacks for TensorBoard and ModelCheckpoint."""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(log_dir, "model_{epoch}"),
        monitor='val_mse',
        save_best_only=False,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=10,
        restore_best_weights=True
    )
    return [tensorboard_callback, checkpoint_callback, early_stopping_callback]

def main():
    TOPIC = config_legacy.TOPIC
    HISTOGRAM_FREQ = config_legacy.HISTOGRAM_FREQ
    EPOCHS = config_legacy.EPOCHS
    CSV_FILE_PATH = config_legacy.CSV_FILE_PATH
    LOGDIR = config_legacy.LOGDIR
    MODELDIR = config_legacy.MODELDIR
    BATCH_SIZE = config_legacy.BATCH_SIZE
    
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(MODELDIR, exist_ok=True)
    
    log_dir, model_dir = create_save_paths(LOGDIR, MODELDIR, TOPIC)
    print("Logging to", log_dir)
    print("Saving best model to", model_dir)
    
    X, y = load_and_preprocess_data(CSV_FILE_PATH)
    model = build_dnn_model(input_dim=X.shape[1])
    
    callbacks = create_callbacks(log_dir, HISTOGRAM_FREQ)
    
    history = model.fit(X, y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        validation_split=0.01,
                        callbacks=callbacks)
    
    print(f"Training complete. Logs saved to: {log_dir}")
    print(f"Models saved to: {model_dir}")

if __name__ == "__main__":
    main()
