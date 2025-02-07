import datetime
import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from deepctr.models import DCN
from deepctr.inputs import DenseFeat
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Input
from tensorflow.keras.models import Model

# Configuration parameters
BATCH_SIZE = 128
TOPIC = "qc_dcn_1k_qc_cpu_tf2"  # Updated topic to reflect TF2
HISTOGRAM_FREQ = 1
LOGDIR = "cpu/logs/fit/"
EPOCHS = 3
MODELDIR = "cpu/models"
NUM_CIRCUIT_PARAMS = 25 # Number of parameters in the quantum circuit

@tf.function
def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped log and model save paths."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)

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

    def dataframe_to_dataset(dataframe, batch_size, shuffle=True):
        """Convert Pandas DataFrame to tf.data.Dataset."""
        dataframe = dataframe.copy()
        labels = dataframe.pop(target[0])  # Assuming only one target
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    train_ds = dataframe_to_dataset(train[dense_features + target], batch_size)
    test_ds = dataframe_to_dataset(test[dense_features + target], batch_size, shuffle=False)

    return train_ds, test_ds, dense_features

def build_dcn_model(linear_features, dnn_features, num_circuit_params):
    """Define and compile the DCN model."""
    linear_feature_columns = [DenseFeat(feat, 1) for feat in linear_features]
    dnn_feature_columns = [DenseFeat(feat, 1) for feat in dnn_features]

    # Define input layer
    input_layer = Input(shape=(len(linear_features),), name='dcn_input')

    # Build DCN model
    dcn = DCN(linear_feature_columns, dnn_feature_columns, cross_num=1,
                dnn_hidden_units=(256, 256, 128, 128, 64), task='regression')(input_layer) # Increased DNN units
    
    # Add batch normalization and dropout
    bn = BatchNormalization()(dcn)
    dropout = Dropout(0.25)(bn)

    # Parameter prediction layer
    parameter_prediction_layer = Dense(num_circuit_params, activation='linear', name='circuit_params')(dropout)

    # Define the model with two outputs: DCN output and circuit parameters
    model = Model(inputs=input_layer, outputs=[dcn, parameter_prediction_layer])

    # Compile the model with two losses and two sets of metrics
    model.compile(optimizer="adam", 
                  loss={'dcn': 'mse', 'circuit_params': 'mse'},  # Example loss for parameter prediction
                  metrics={'dcn': tf.keras.metrics.RootMeanSquaredError(), 'circuit_params': 'mse'})
    return model

def main():
    print("Using TensorFlow", tf.__version__)
    log_dir, model_dir = create_save_paths(LOGDIR, MODELDIR, TOPIC)
    print("Logging to", log_dir)

    # Set up callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=HISTOGRAM_FREQ)
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        monitor='val_dcn_root_mean_squared_error',  # Monitor DCN's RMSE
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch'
    )

    # Load data and prepare model inputs
    train_ds, test_ds, feat_names = load_and_prepare_data("./qc5f_1k_2.csv", BATCH_SIZE)
    
    # Build model
    model = build_dcn_model(feat_names, feat_names, NUM_CIRCUIT_PARAMS)

    # Separate features and labels for training
    def prepare_data(ds):
        for features, label in ds:
            yield features, {'dcn': label, 'circuit_params': tf.zeros(NUM_CIRCUIT_PARAMS)}  # Dummy target for circuit_params

    train_data = list(prepare_data(train_ds))
    test_data = list(prepare_data(test_ds))

    # Train the model
    history = model.fit(
        x=[item[0] for item in train_data],  # Input features
        y=[item[1] for item in train_data],  # DCN and circuit_params targets
        epochs=EPOCHS,
        verbose=1,
        validation_data=([item[0] for item in test_data], [item[1] for item in test_data]),
        callbacks=[tensorboard_callback, model_callback]
    )

    # Evaluate model on test data
    loss = model.evaluate(x=[item[0] for item in test_data], y=[item[1] for item in test_data], verbose=0)
    print("Test Loss:", loss)

if __name__ == "__main__":
    main()
