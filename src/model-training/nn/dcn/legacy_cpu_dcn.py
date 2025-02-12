import datetime
import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from deepctr.models import DCN
from deepctr.inputs import DenseFeat
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Input
from tensorflow.keras.models import Model

from dcn import config
from dcn.utils import create_save_paths, dataframe_to_dataset

# Configuration parameters
BATCH_SIZE = config.BATCH_SIZE
TOPIC = config.TOPIC
HISTOGRAM_FREQ = config.HISTOGRAM_FREQ
LOGDIR = config.LOGDIR
EPOCHS = config.EPOCHS
MODELDIR = config.MODELDIR
NUM_CIRCUIT_PARAMS = config.NUM_CIRCUIT_PARAMS
CSV_FILE_PATH = config.CSV_FILE_PATH

def load_and_prepare_data(csv_path: str, batch_size: int):
    """Load CSV data, split into train/test, and format inputs using tf.data."""
    data = pd.read_csv(csv_path)
    
    # Define features and target
    dense_features = [
        f"gate_{str(i).zfill(2)}_{suffix}"
        for i in range(41)
        for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
    ]
    target = ['statevector_00000']

    # Split data
    train, test = train_test_split(data, test_size=0.2)

    train_ds = dataframe_to_dataset(train[dense_features + target], batch_size, shuffle=True)
    test_ds = dataframe_to_dataset(test[dense_features + target], batch_size, shuffle=False)

    return train_ds, test_ds, dense_features

def build_dcn_model(linear_features, dnn_features, num_circuit_params):
    """Define and compile the DCN model.

    Args:
        linear_features: List of linear features.
        dnn_features: List of DNN features.
        num_circuit_params: Number of parameters in the quantum circuit.

    Returns:
        A compiled Keras model.
    """
    linear_feature_columns = [DenseFeat(feat, 1) for feat in linear_features]
    dnn_feature_columns = [DenseFeat(feat, 1) for feat in dnn_features]

    # Define input layer
    input_layer = Input(shape=(len(linear_features),), name='dcn_input')

    # Build DCN model
    # The DCN model combines linear and deep learning components to capture both low-order and high-order feature interactions.
    dcn = DCN(linear_feature_columns, dnn_feature_columns, cross_num=1,
                dnn_hidden_units=(256, 256, 128, 128, 64), task='regression')(input_layer) # Increased DNN units
    
    # Add batch normalization and dropout
    # Batch normalization helps to stabilize training and improve generalization.
    bn = BatchNormalization()(dcn)
    # Dropout helps to prevent overfitting.
    dropout = Dropout(0.25)(bn)

    # Parameter prediction layer
    # This layer predicts the parameters for the quantum circuit.
    parameter_prediction_layer = Dense(num_circuit_params, activation='linear', name='circuit_params')(dropout)

    # Define the model with two outputs: DCN output and circuit parameters
    # The model has two outputs: the DCN output and the predicted circuit parameters.
    model = Model(inputs=input_layer, outputs=[dcn, parameter_prediction_layer])

    # Compile the model with two losses and two sets of metrics
    # The model is compiled with two losses: mean squared error (MSE) for both the DCN output and the circuit parameters.
    model.compile(optimizer="adam", 
                  loss={'dcn': 'mse', 'circuit_params': 'mse'},  # Example loss for parameter prediction
                  metrics={'dcn': tf.keras.metrics.RootMeanSquaredError(), 'circuit_params': 'mse'})
    return model

def prepare_data(ds):
    """Separate features and labels for training."""
    for features, label in ds:
        yield features, {'dcn': label, 'circuit_params': tf.zeros(NUM_CIRCUIT_PARAMS)}  # Dummy target for circuit_params

def create_callbacks(log_dir, histogram_freq):
    """Create callbacks for TensorBoard and ModelCheckpoint."""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir,
        monitor='val_dcn_root_mean_squared_error',  # Monitor DCN's RMSE
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch'
    )
    return [tensorboard_callback, model_callback]

def main():
    print("Using TensorFlow", tf.__version__)
    log_dir, model_dir = create_save_paths(LOGDIR, MODELDIR, TOPIC)
    print("Logging to", log_dir)

    # Set up callbacks
    callbacks = create_callbacks(log_dir, HISTOGRAM_FREQ)

    # Load data and prepare model inputs
    train_ds, test_ds, feat_names = load_and_prepare_data(CSV_FILE_PATH, BATCH_SIZE)
    
    # Build model
    model = build_dcn_model(feat_names, feat_names, NUM_CIRCUIT_PARAMS)

    train_data = list(prepare_data(train_ds))
    test_data = list(prepare_data(test_ds))

    # Train the model
    history = model.fit(
        x=[item[0] for item in train_data],  # Input features
        y=[item[1] for item in train_data],  # DCN and circuit_params targets
        epochs=EPOCHS,
        verbose=1,
        validation_data=([item[0] for item in test_data], [item[1] for item in test_data]),
        callbacks=callbacks
    )

    # Evaluate model on test data
    loss = model.evaluate(x=[item[0] for item in test_data], y=[item[1] for item in test_data], verbose=0)
    print("Test Loss:", loss)

if __name__ == "__main__":
    main()
