# utils/model_utils.py
import tensorflow as tf
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def train(self, train_data, epochs, validation_data=None, callbacks=None):
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Makes predictions using the model."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Saves the model to the specified filepath."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Loads the model from the specified filepath."""
        pass

class DNNModel(BaseModel, tf.keras.Model):
    def __init__(self, input_dim, num_params, dense_units=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_params = num_params
        self.dense_units = dense_units
        self.num_layers = num_layers
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(input_dim,)))
        for _ in range(num_layers):
            self.model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(num_params))

    def train(self, train_data, epochs, validation_data=None, callbacks=None):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data, callbacks=callbacks)

    def predict(self, input_data):
        return self.model.predict(input_data)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

def create_dnn_model(input_dim, num_params, dense_units=128, num_layers=3):
    """Creates a DNN model that outputs parameters for a quantum circuit."""
    return DNNModel(input_dim, num_params, dense_units, num_layers)
