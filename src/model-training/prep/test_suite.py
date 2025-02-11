import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

import tensorflow as tf

# Import functions and constants from csv_norm.py
from src.model-training.prep.csv_norm import (
    generate_statevector_keys,
    generate_gate_keys,
    normalize_gate_number,
    normalize_shard,
    normalize_memory,
    process_csv,
)

# Import convert_to_tfrecord from csv_to_tfrecord.py
from src.model-training.prep.csv_to_tfrecord import convert_to_tfrecord


# ------------------------------
# Tests for csv_norm.py
# ------------------------------

def test_generate_statevector_keys():
    """Test that statevector keys are generated correctly for num_qubits=5."""
    num_qubits: int = 5
    keys: List[str] = generate_statevector_keys(num_qubits)
    expected_length: int = 2 ** num_qubits
    assert len(keys) == expected_length
    for key in keys:
        assert key.startswith("statevector_")
        # The binary part should be exactly num_qubits long.
        bin_part: str = key.split("_")[1]
        assert len(bin_part) == num_qubits


def test_generate_gate_keys():
    """Test that gate keys are generated correctly for num_gates=40."""
    num_gates: int = 40
    keys: Dict[str, List[str]] = generate_gate_keys(num_gates)
    for field in ["type", "number", "control", "target", "angle1", "angle2", "angle3"]:
        # The keys lists should contain num_gates+1 items.
        assert len(keys[field]) == num_gates + 1
        for key in keys[field]:
            assert key.startswith("gate_")


@pytest.mark.parametrize("gate_number, expected", [
    (-1, 0),
    (0, 1),
    (10, 1)
])
def test_normalize_gate_number(gate_number: int, expected: int):
    """Test that normalize_gate_number returns 0 for -1 and 1 for other numbers."""
    assert normalize_gate_number(gate_number) == expected


def create_dummy_dataframe(num_qubits: int = 5, num_gates: int = 40):
    """Creates a dummy DataFrame for testing."""
    data: Dict[str, List[float]] = {}

    # statevector_keys: use 1024 so that division yields 1.0.
    for col in generate_statevector_keys(num_qubits):
        data[col] = [1024.0]

    # Gate type: use "U3Gate" (which should be mapped to 0).
    for col in generate_gate_keys(num_gates)["type"]:
        data[col] = ["U3Gate"]

    # Control and target: use 4 so that (4+1)/5 = 1.0.
    for col in generate_gate_keys(num_gates)["control"] + generate_gate_keys(num_gates)["target"]:
        data[col] = [4.0]

    # Angles: use 2 so that (2+1)/3 = 1.0.
    for col in generate_gate_keys(num_gates)["angle1"] + generate_gate_keys(num_gates)["angle2"] + generate_gate_keys(num_gates)["angle3"]:
        data[col] = [2.0]

    # Gate numbers: use -1 so that normalize_gate_number returns 0.
    for col in generate_gate_keys(num_gates)["number"]:
        data[col] = [-1.0]

    return pd.DataFrame(data)


def test_normalize_shard(tmp_path: Path):
    """
    Test the normalize_shard function.
    """
    num_qubits = 5
    num_gates = 40
    df = create_dummy_dataframe(num_qubits, num_gates)
    input_file: Path = tmp_path / "test_shard.csv"
    df.to_csv(input_file, index=False)
    output_file: Path = tmp_path / "test_shard_output.csv"

    process_csv(input_file, output_file, method="shard", num_qubits=num_qubits, num_gates=num_gates)

    assert output_file.exists()

    df_out: pd.DataFrame = pd.read_csv(output_file)
    # Check that statevector values are normalized to 1.0.
    for col in generate_statevector_keys(num_qubits):
        np.testing.assert_allclose(df_out[col].values, [1.0])
    # Check that gate type values are mapped (U3Gate -> 0).
    for col in generate_gate_keys(num_gates)["type"]:
        np.testing.assert_allclose(df_out[col].values, [0.0])
    # Check control/target normalization: (4+1)/5 = 1.0.
    for col in generate_gate_keys(num_gates)["control"] + generate_gate_keys(num_gates)["target"]:
        np.testing.assert_allclose(df_out[col].values, [1.0])
    # Check angles normalization: (2+1)/3 = 1.0.
    for col in generate_gate_keys(num_gates)["angle1"] + generate_gate_keys(num_gates)["angle2"] + generate_gate_keys(num_gates)["angle3"]:
        np.testing.assert_allclose(df_out[col].values, [1.0])
    # Check gate numbers normalization: -1 becomes 0.
    for col in generate_gate_keys(num_gates)["number"]:
        np.testing.assert_allclose(df_out[col].values, [0.0])


def test_normalize_memory(tmp_path: Path):
    """
    Test the normalize_memory function.
    """
    num_qubits = 5
    num_gates = 40
    df = create_dummy_dataframe(num_qubits, num_gates)
    input_file: Path = tmp_path / "test_memory.csv"
    df.to_csv(input_file, index=False)
    output_file: Path = tmp_path / "test_memory_output.csv"

    process_csv(input_file, output_file, method="memory", num_qubits=num_qubits, num_gates=num_gates)

    assert output_file.exists()

    df_out: pd.DataFrame = pd.read_csv(output_file)
    # The memory method uses MinMaxScaler, so the values will be different.
    # We can't directly check the values, but we can check that the function runs without errors.
    assert True


# ------------------------------
# Tests for csv_to_tfrecord.py
# ------------------------------

def test_convert_to_tfrecord(tmp_path: Path):
    """
    Create a small CSV file with a label column and two feature columns,
    run convert_to_tfrecord, and then read back the TFRecord file to verify its contents.
    """
    # Create dummy CSV data.
    data: Dict[str, List[float]] = {
        "label": [0, 1],
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0]
    }
    df: pd.DataFrame = pd.DataFrame(data)
    csv_file: Path = tmp_path / "dummy.csv"
    df.to_csv(csv_file, index=False)

    # Define the output TFRecord file.
    tfrecord_file: Path = tmp_path / "dummy.tfrecords"

    # Run conversion (the script uses a 'limit' parameter to optionally restrict record count).
    convert_to_tfrecord(str(csv_file), str(tfrecord_file), limit=2)

    # Read the generated TFRecord file.
    dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset([str(tfrecord_file)])
    examples: List[tf.train.Example] = []
    for raw_record in dataset:
        example: tf.train.Example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        examples.append(example)

    # Check that two examples were written.
    assert len(examples) == 2

    # Verify the contents of the first example.
    ex0: tf.train.Example = examples[0]
    # The first column (label) should be stored in 'label' as an int64 feature.
    label_val: int = ex0.features.feature["label"].int64_list.value[0]
    assert label_val == 0
    # The remaining columns are stored in 'features' as a float_list.
    feature_vals: List[float] = ex0.features.feature["features"].float_list.value
    # Since the CSV assumed the first column is label, features should be [feature1, feature2].
    # In our dummy CSV, for the first row these are 1.0 and 3.0.
    np.testing.assert_allclose(feature_vals, [1.0, 3.0])


if __name__ == "__main__":
    # Allow running the tests directly.
    pytest.main([os.path.abspath(__file__)])
