# utils/model_utils.py
import tensorflow as tf

def create_dnn_model(input_dim, num_params, dense_units=128, num_layers=3):
    """Creates a DNN model that outputs parameters for a quantum circuit."""
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_params)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
