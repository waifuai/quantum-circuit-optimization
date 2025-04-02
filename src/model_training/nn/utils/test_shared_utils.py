import cirq
import numpy as np
import tensorflow as tf
import pandas as pd
import pytest # Added for potential pytest features like fixtures if needed later

# Import functions from your utils modules.
from model_training.nn.utils import circuit_utils
from model_training.nn.utils import model_utils
from model_training.nn.utils import path_utils
from model_training.nn.utils import data_utils

# --- Tests for circuit_utils.py ---

def test_create_circuit_default():
    """Test creating a default circuit."""
    # Use a fixed set of 25 parameters (5 layers * 5 qubits)
    params = [0.1] * 25
    circuit = circuit_utils.create_circuit(params, num_qubits=5)

    # Verify the returned object is a cirq.Circuit instance.
    assert isinstance(circuit, cirq.Circuit)

    # Each of the 5 layers should include:
    # - 5 RX operations (one per qubit)
    # - 4 CNOT operations (between consecutive qubits)
    # Total operations = 5 * (5 + 4) = 45.
    num_ops = len(list(circuit.all_operations()))
    assert num_ops == 45

    # Verify that if no qubits are provided, the default is a 5-qubit register.
    expected_qubits = list(cirq.LineQubit.range(5))
    circuit_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
    assert circuit_qubits == expected_qubits

def test_create_circuit_custom_qubits():
    """Test creating a circuit with custom qubits."""
    params = [0.2] * 25
    # Create custom qubits (for example, using GridQubit)
    custom_qubits = [cirq.GridQubit(0, i) for i in range(5)]
    circuit = circuit_utils.create_circuit(params, qubits=custom_qubits)

    # Check that the circuit uses the custom qubits
    circuit_qubits = sorted(circuit.all_qubits(), key=lambda q: (q.row, q.col))
    assert circuit_qubits == custom_qubits

def test_calculate_loss_identity():
    """Test calculating loss for an identity circuit."""
    # Create an empty circuit (which acts as an identity) on 5 qubits.
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()

    # Simulate the empty circuit; the final state will be the |0...0> state.
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    target_state = result.final_state_vector

    # The loss between the circuit output and the target state should be near zero.
    loss = circuit_utils.calculate_loss(circuit, target_state)
    # Use pytest.approx for floating point comparisons
    assert loss == pytest.approx(0.0, abs=1e-5)

# --- Tests for model_utils.py ---
# (Add tests for model_utils if they exist or are created)
# Example placeholder:
def test_create_dnn_model():
    """Test creating a DNN model."""
    input_dim = 10
    num_params = 25
    model = model_utils.create_dnn_model(input_dim, num_params)
    assert isinstance(model, tf.keras.Model)
    # Check output shape - model needs to be built first, or check config
    # Example: Check the output layer's units if accessible without building
    assert model.model.layers[-1].units == num_params

# --- Test DNN to Cirq Integration ---

def test_dnn_to_cirq_circuit():
    """
    Tests the integration between a DNN model and Cirq circuit creation.
    """
    # Define model parameters
    input_dim = 10
    num_params = 25 # Should match the default circuit's expectation (5 qubits * 5 layers)

    # Create a small, untrained DNN model
    model = model_utils.create_dnn_model(input_dim, num_params)

    # Generate random input data
    input_data = np.random.rand(1, input_dim).astype(np.float32)

    # Use the DNN to predict circuit parameters
    predicted_params = model(input_data) # Get tensor output

    # Create a Cirq circuit from the predicted parameters
    # Ensure the number of parameters matches the circuit expectation
    circuit = circuit_utils.create_circuit(predicted_params.numpy().flatten(), num_qubits=5) # Use default 5 qubits

    # Assert that the created circuit is a valid cirq.Circuit object
    assert isinstance(circuit, cirq.Circuit)

    # Check that the circuit has qubits (default is 5)
    assert len(list(circuit.all_qubits())) == 5

    # Check that the circuit has operations (default is 45)
    assert len(list(circuit.all_operations())) == 45


# --- Tests for path_utils.py (from dcn/test_utils.py) ---

def test_create_save_paths():
    """
    Test that create_save_paths returns valid log and model paths containing the topic string.
    """
    logdir = "test_logs"
    modeldir = "test_models"
    topic = "test_topic"
    # Assuming path_utils uses config.LOG_DIR and config.MODEL_DIR by default,
    # but allows overriding. For isolated test, provide bases directly.
    log_path, model_path = path_utils.create_save_paths(logdir_base=logdir, modeldir_base=modeldir, topic=topic)
    assert topic in log_path
    assert topic in model_path
    assert isinstance(log_path, str)
    assert isinstance(model_path, str)
    # Note: Checking startswith might be fragile if absolute paths are generated.
    # Better to check if the base directory is part of the path.
    assert logdir in log_path
    assert modeldir in model_path
    # Clean up created directories if necessary, or use pytest tmp_path fixture

# --- Tests for data_utils.py (from dcn/test_utils.py) ---

def test_dataframe_to_dataset():
    """
    Test that dataframe_to_dataset returns a tf.data.Dataset with the correct structure.
    """
    # Create a dummy DataFrame
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'statevector_00000': [7, 8, 9]}
    df = pd.DataFrame(data)
    batch_size = 2

    # Create the dataset
    dataset = data_utils.dataframe_to_dataset(df, batch_size)

    # Verify that the dataset is a tf.data.Dataset
    assert isinstance(dataset, tf.data.Dataset)

    # Verify the structure of the dataset
    for features, labels in dataset.take(1):
        assert isinstance(features, dict)
        assert 'feature1' in features
        assert features['feature1'].shape == (batch_size,)
        assert labels.shape == (batch_size,)
        # Check dtypes if necessary
        assert features['feature1'].dtype == tf.int64 # Pandas infers int64
        assert labels.dtype == tf.int64

# (Add tests for load_and_preprocess_data if needed)
