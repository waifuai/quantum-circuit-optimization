import os
import sys
import datetime
import tempfile
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

# --- Helper Functions for Tests ---

def create_dense_feature_names():
    """Return the list of dense feature names as generated in your code."""
    features = []
    for i in range(41):
        n = str(i).zfill(2)
        prefix = f"gate_{n}_"
        features.extend([prefix + suffix for suffix in 
                         ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]])
    return features

def create_dummy_csv(tmp_path, n_qubits=5, num_rows=1, include_target=True):
    """
    Create a dummy CSV file that matches the expected structure.
    If include_target is True, the file will also have the target column(s).
    """
    dense_features = create_dense_feature_names()
    data = {col: np.random.rand(num_rows) for col in dense_features}
    if include_target:
        # For 1s files the target is "statevector_00000"
        target_col = "statevector_00000"
        data[target_col] = np.random.rand(num_rows)
        columns = dense_features + [target_col]
    else:
        # For 32s files, include a full set of statevector columns
        target_columns = [f"statevector_{bin(i)[2:].zfill(n_qubits)}" for i in range(2**n_qubits)]
        for col in target_columns:
            data[col] = np.random.rand(num_rows)
        columns = dense_features + target_columns
    df = pd.DataFrame(data, columns=columns)
    csv_file = tmp_path / "dummy.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file), dense_features

# --- Tests for legacy_dnn_regression.py functions ---

def test_create_save_paths(tmp_path):
    # Import the create_save_paths function from legacy_dnn_regression.py
    # (Adjust the import path if needed.)
    from legacy_dnn_regression import create_save_paths
    logdir = str(tmp_path / "logs")
    modeldir = str(tmp_path / "models")
    topic = "test_topic"
    log_path, model_path = create_save_paths(logdir, modeldir, topic)
    # Check that the returned paths contain the topic and a timestamp-like substring.
    assert topic in log_path
    assert topic in model_path
    # A basic check for timestamp formatting
    now_str = datetime.datetime.now().strftime("%Y%m%d")
    assert now_str in log_path

def test_load_and_preprocess_data(tmp_path):
    # Import the load_and_preprocess_data function from legacy_dnn_regression.py
    from legacy_dnn_regression import load_and_preprocess_data
    csv_file, dense_features = create_dummy_csv(tmp_path, include_target=True)
    X, y = load_and_preprocess_data(csv_file)
    expected_num_features = len(dense_features)
    # Verify that X has the correct number of features and y is a single column.
    assert X.shape[1] == expected_num_features
    assert y.shape[1] == 1

def test_build_dnn_model():
    # Import the build_dnn_model function from legacy_dnn_regression.py
    from legacy_dnn_regression import build_dnn_model
    input_dim = 10
    model = build_dnn_model(input_dim)
    # Check that model is a compiled Keras model with the expected input shape.
    assert isinstance(model, tf.keras.Model)
    dummy_input = np.random.rand(1, input_dim)
    output = model.predict(dummy_input)
    # Since build_dnn_model outputs one value, verify shape is (1, 1).
    assert output.shape == (1, 1)

# --- Tests for legacy_dnn_regression_32s.py functions ---

def test_load_and_preprocess_data_32s(tmp_path):
    # Import load_and_preprocess_data from legacy_dnn_regression_32s.py
    from legacy_dnn_regression_32s import load_and_preprocess_data
    csv_file, dense_features = create_dummy_csv(tmp_path, include_target=False)
    # For 32s, the function returns a DataFrame and the list of dense features.
    data, df_dense_features = load_and_preprocess_data(csv_file)
    for col in dense_features:
        assert col in data.columns

def test_build_dnn_model_32s():
    # Import the build_dnn_model function from legacy_dnn_regression_32s.py
    from legacy_dnn_regression_32s import build_dnn_model
    input_dim = 10
    output_units = 32
    model = build_dnn_model(input_dim, output_units)
    assert isinstance(model, tf.keras.Model)
    dummy_input = np.random.rand(1, input_dim)
    output = model.predict(dummy_input)
    # Check that output has the expected shape (1, 32)
    assert output.shape == (1, output_units)

# --- Tests for legacy_dnn_regression_32s_resume.py functions ---

def test_preprocess_data_resume(tmp_path):
    from legacy_dnn_regression_32s_resume import preprocess_data
    n_qubits = 5
    # For resume tests, include the full set of statevector target columns.
    dense_features = create_dense_feature_names()
    target_columns = [f"statevector_{bin(i)[2:].zfill(n_qubits)}" for i in range(2**n_qubits)]
    all_cols = dense_features + target_columns
    data_dict = {col: np.random.rand(1) for col in all_cols}
    df = pd.DataFrame(data_dict, columns=all_cols)
    csv_file = tmp_path / "dummy_resume.csv"
    df.to_csv(csv_file, index=False)
    X, y, targets = preprocess_data(str(csv_file), n_qubits)
    assert X.shape[1] == len(dense_features)
    assert y.shape[1] == 2**n_qubits
    # Check that target column names match
    assert targets == target_columns

def test_load_or_build_model(tmp_path):
    from legacy_dnn_regression_32s_resume import load_or_build_model
    input_shape = 10
    # Remove any existing pretrained model to force building a new one.
    pretrained_path = "pretrained_model.h5"
    if os.path.exists(pretrained_path):
        os.remove(pretrained_path)
    model = load_or_build_model(input_shape)
    assert isinstance(model, tf.keras.Model)
    dummy_input = np.random.rand(1, input_shape)
    output = model.predict(dummy_input)
    # In the resume version, the final Dense layer outputs 32 values.
    assert output.shape == (1, 32)

# --- Tests for the hybrid optimizer scripts (1s and 32s) ---

@pytest.fixture
def dummy_csv_1s(tmp_path):
    # Create a dummy CSV file for the 1s optimizer (with a single target column)
    csv_file, _ = create_dummy_csv(tmp_path, include_target=True)
    return csv_file

@pytest.fixture
def dummy_csv_32s(tmp_path):
    # Create a dummy CSV file for the 32s optimizer (with full statevector targets)
    csv_file, _ = create_dummy_csv(tmp_path, include_target=False)
    return csv_file

def import_module_from_path(module_name, filepath):
    """Helper to import a module given its file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_hybrid_optimizer_csv_not_found_1s(tmp_path, monkeypatch, capsys):
    # For the 1s version, simulate a missing CSV file by monkeypatching pd.read_csv to always raise FileNotFoundError.
    import pandas as pd
    monkeypatch.setattr(pd, "read_csv", lambda path: (_ for _ in ()).throw(FileNotFoundError))
    # Import the module from its file path (adjust the relative path if needed).
    module_path = os.path.join("1s", "hybrid_dnn_cirq_optimizer.py")
    with pytest.raises(SystemExit):
        import_module_from_path("hybrid_dnn_cirq_optimizer", module_path)
    captured = capsys.readouterr().out
    assert "Error: CSV file not found" in captured

def test_hybrid_optimizer_circuit_1s(tmp_path, monkeypatch, capsys):
    # For the 1s version, monkeypatch the create_circuit function to return a dummy string.
    # Also, point CSV_FILE_PATH to a dummy CSV file.
    dummy_csv = dummy_csv_1s(tmp_path)
    
    # Patch the create_circuit function in the utils.circuit_utils module.
    # (Assuming that the module structure allows this import.)
    try:
        import utils.circuit_utils as circuit_utils
    except ImportError:
        # If the utils package is not on the path, add the current directory.
        sys.path.insert(0, os.getcwd())
        import utils.circuit_utils as circuit_utils

    monkeypatch.setattr(circuit_utils, "create_circuit",
                        lambda params, qubits: f"dummy circuit with params: {params}")

    # Now import the hybrid optimizer module and override its CSV_FILE_PATH variable.
    module_path = os.path.join("1s", "hybrid_dnn_cirq_optimizer.py")
    module = import_module_from_path("hybrid_dnn_cirq_optimizer", module_path)
    monkeypatch.setattr(module, "CSV_FILE_PATH", dummy_csv)
    # Re-run the part that creates the circuit by calling the DNN prediction section.
    # (This test assumes that on import the module prints the generated circuit.)
    captured = capsys.readouterr().out
    assert "dummy circuit with params:" in captured

def test_hybrid_optimizer_circuit_32s(tmp_path, monkeypatch, capsys):
    # For the 32s version, similarly monkeypatch create_circuit and point CSV_FILE_PATH to a dummy CSV.
    dummy_csv = dummy_csv_32s(tmp_path)
    try:
        import utils.circuit_utils as circuit_utils
    except ImportError:
        sys.path.insert(0, os.getcwd())
        import utils.circuit_utils as circuit_utils

    monkeypatch.setattr(circuit_utils, "create_circuit",
                        lambda params, qubits: f"dummy circuit with params: {params}")
    module_path = os.path.join("32s", "hybrid_dnn_cirq_optimizer_32s.py")
    module = import_module_from_path("hybrid_dnn_cirq_optimizer_32s", module_path)
    monkeypatch.setattr(module, "CSV_FILE_PATH", dummy_csv)
    captured = capsys.readouterr().out
    assert "dummy circuit with params:" in captured
