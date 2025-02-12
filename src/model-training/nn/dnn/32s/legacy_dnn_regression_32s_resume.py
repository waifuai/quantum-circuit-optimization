import datetime
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from dnn.32s import config_legacy_resume
from dnn.32s.utils_legacy import create_save_paths
from dnn.32s.data_utils_legacy_resume import preprocess_data

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
    TOPIC = config_legacy_resume.TOPIC
    EPOCHS = config_legacy_resume.EPOCHS
    CSV_FILE_PATH = config_legacy_resume.CSV_FILE_PATH
    LOGDIR = config_legacy_resume.LOGDIR
    MODELDIR = config_legacy_resume.MODELDIR
    N_QUBITS = config_legacy_resume.N_QUBITS
    
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
