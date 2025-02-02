import datetime
import math
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deepctr.models import DCN
from deepctr.inputs import DenseFeat, get_feature_names

# Configuration parameters
BATCH_SIZE = 128
TOPIC = "qc_dcn_1k_qc_cpu_tf2"  # Updated topic to reflect TF2
HISTOGRAM_FREQ = 1
LOGDIR = "cpu/logs/fit/"
EPOCHS = 3
MODELDIR = "cpu/models"

def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped log and model save paths."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)

def load_and_prepare_data(csv_path: str, batch_size: int):
    """Load CSV data, split into train/test, and format inputs."""
    data = pd.read_csv(csv_path)
    
    # Define features and target
    dense_features = [
        f"gate_{str(i).zfill(2)}_{suffix}"
        for i in range(41)
        for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
    ]
    target = ['statevector_00000']

    # Create feature columns for DeepCTR (only DenseFeat is needed)
    fixlen_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    feature_names = get_feature_names(fixlen_feature_columns)

    # Split data and adjust to be divisible by batch_size
    train, test = train_test_split(data, test_size=0.2)
    train = train[: (len(train) // batch_size) * batch_size]
    test = test[: (len(test) // batch_size) * batch_size]

    train_input = {name: train[name].values.astype('float32') for name in feature_names}
    test_input = {name: test[name].values.astype('float32') for name in feature_names}
    return train_input, test_input, train[target].values, test[target].values, fixlen_feature_columns

def build_dcn_model(linear_features, dnn_features):
    """Define and compile the DCN model."""
    model = DCN(linear_features, dnn_features, cross_num=1,
                dnn_hidden_units=(128, 128, 128, 128, 128), task='regression')
    model.compile(optimizer="adam", loss="mse", metrics=['mse'])
    return model

def log_rmse(history, epochs):
    """Log RMSE for training and validation using TensorBoard summaries."""
    for epoch, mse_val in enumerate(history.history['mse']):
        tf.summary.scalar('rmse/train', math.sqrt(mse_val), step=epoch)
    for epoch, mse_val in enumerate(history.history['val_mse']):
        tf.summary.scalar('rmse/validation', math.sqrt(mse_val), step=epoch)
    # Log final test RMSE later

def main():
    print("Using TensorFlow", tf.__version__)
    log_dir, model_dir = create_save_paths(LOGDIR, MODELDIR, TOPIC)
    print("Logging to", log_dir)

    # Set up callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=HISTOGRAM_FREQ)
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        monitor='val_mse',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

    # Load data and prepare model inputs
    train_input, test_input, y_train, y_test, feat_columns = load_and_prepare_data("./qc5f_1k_2.csv", BATCH_SIZE)
    
    # Build model
    model = build_dcn_model(feat_columns, feat_columns)
    history = model.fit(train_input, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(test_input, y_test),
                        callbacks=[tensorboard_callback, model_callback])
    
    # Log training metrics
    log_rmse(history, EPOCHS)
    
    # Evaluate model on test data
    pred_ans = model.predict(test_input, batch_size=BATCH_SIZE)
    rmse = math.sqrt(mean_squared_error(y_test, pred_ans))
    print("Test RMSE:", rmse)
    tf.summary.scalar('rmse/test', rmse, step=EPOCHS)

if __name__ == "__main__":
    main()
