import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

import tensorflow as tf

# Import functions and constants from csv_norm_new.py
from csv_norm_new import (
    generate_statevector_keys,
    generate_gate_keys,
    normalize_gate_number,
    normalize_shard,
    STATEVECTOR_KEYS,
    GATE_KEYS,
)

# Import generate_feature_names from csv_norm_old.py
from csv_norm_old import generate_feature_names

# Import convert_to_tfrecord from csv_to_tfrecord.py
from csv_to_tfrecord import convert_to_tfrecord


# ------------------------------
# Tests for csv_norm_new.py
# ------------------------------

def test_generate_statevector_keys():
    """Test that statevector keys are generated correctly for n_qubits=5."""
    n_qubits = 5
    keys = generate_statevector_keys(n_qubits)
    expected_length = 2 ** n_qubits
    assert len(keys) == expected_length
    for key in keys:
        assert key.startswith("statevector_")
        # The binary part should be exactly n_qubits long.
        bin_part = key.split("_")[1]
        assert len(bin_part) == n_qubits


def test_generate_gate_keys():
    """Test that gate keys are generated correctly for num_gates=40."""
    num_gates = 40
    keys = generate_gate_keys(num_gates)
    for field in ["type", "number", "control", "target", "angle1", "angle2", "angle3"]:
        # The keys lists should contain num_gates+1 items.
        assert len(keys[field]) == num_gates + 1
        for key in keys[field]:
            assert key.startswith("gate_")


@pytest.mark.parametrize("n, expected", [
    (-1, 0),
    (0, 1),
    (10, 1)
])
def test_normalize_gate_number(n, expected):
    """Test that normalize_gate_number returns 0 for -1 and 1 for other numbers."""
    assert normalize_gate_number(n) == expected


def test_normalize_shard(tmp_path):
    """
    Create a dummy CSV shard with the expected columns, run normalize_shard,
    and verify that each normalization step has been applied correctly.
    """
    # Prepare one row of dummy data.
    data = {}

    # STATEVECTOR_KEYS: use 1024 so that division yields 1.0.
    for col in STATEVECTOR_KEYS:
        data[col] = [1024]

    # Gate type: use "U3Gate" (which should be mapped to 0).
    for col in GATE_KEYS["type"]:
        data[col] = ["U3Gate"]

    # Control and target: use 4 so that (4+1)/5 = 1.0.
    for col in GATE_KEYS["control"] + GATE_KEYS["target"]:
        data[col] = [4]

    # Angles: use 2 so that (2+1)/3 = 1.0.
    for col in GATE_KEYS["angle1"] + GATE_KEYS["angle2"] + GATE_KEYS["angle3"]:
        data[col] = [2]

    # Gate numbers: use -1 so that normalize_gate_number returns 0.
    for col in GATE_KEYS["number"]:
        data[col] = [-1]

    df = pd.DataFrame(data)

    # Write the CSV file in the temporary directory.
    input_file = tmp_path / "test_shard.csv"
    df.to_csv(input_file, index=False)

    # Run normalization.
    normalize_shard(input_file)

    # The output file is created with "_output.csv" appended to the stem.
    output_file = tmp_path / "test_shard_output.csv"
    assert output_file.exists()

    df_out = pd.read_csv(output_file)
    # Check that statevector values are normalized to 1.0.
    for col in STATEVECTOR_KEYS:
        np.testing.assert_allclose(df_out[col].values, [1.0])
    # Check that gate type values are mapped (U3Gate -> 0).
    for col in GATE_KEYS["type"]:
        np.testing.assert_allclose(df_out[col].values, [0])
    # Check control/target normalization: (4+1)/5 = 1.0.
    for col in GATE_KEYS["control"] + GATE_KEYS["target"]:
        np.testing.assert_allclose(df_out[col].values, [1.0])
    # Check angles normalization: (2+1)/3 = 1.0.
    for col in GATE_KEYS["angle1"] + GATE_KEYS["angle2"] + GATE_KEYS["angle3"]:
        np.testing.assert_allclose(df_out[col].values, [1.0])
    # Check gate numbers normalization: -1 becomes 0.
    for col in GATE_KEYS["number"]:
        np.testing.assert_allclose(df_out[col].values, [0])


# ------------------------------
# Tests for csv_norm_old.py
# ------------------------------

def test_generate_feature_names():
    """
    Test that generate_feature_names returns the expected number of feature names
    and that each name follows the expected pattern.
    """
    num_gates = 41
    features = generate_feature_names(num_gates)
    expected_num_features = num_gates * 7  # 7 features per gate
    assert len(features) == expected_num_features
    # Check that each feature name starts with the expected prefix.
    for name in features:
        assert name.startswith("gate_")


# ------------------------------
# Tests for csv_to_tfrecord.py
# ------------------------------

def test_convert_to_tfrecord(tmp_path):
    """
    Create a small CSV file with a label column and two feature columns,
    run convert_to_tfrecord, and then read back the TFRecord file to verify its contents.
    """
    # Create dummy CSV data.
    data = {
        "label": [0, 1],
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0]
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "dummy.csv"
    df.to_csv(csv_file, index=False)

    # Define the output TFRecord file.
    tfrecord_file = tmp_path / "dummy.tfrecords"

    # Run conversion (the script uses a 'limit' parameter to optionally restrict record count).
    convert_to_tfrecord(str(csv_file), str(tfrecord_file), limit=2)

    # Read the generated TFRecord file.
    dataset = tf.data.TFRecordDataset([str(tfrecord_file)])
    examples = []
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        examples.append(example)

    # Check that two examples were written.
    assert len(examples) == 2

    # Verify the contents of the first example.
    ex0 = examples[0]
    # The first column (label) should be stored in 'label' as an int64 feature.
    label_val = ex0.features.feature["label"].int64_list.value[0]
    assert label_val == 0
    # The remaining columns are stored in 'features' as a float_list.
    feature_vals = ex0.features.feature["features"].float_list.value
    # Since the CSV assumed the first column is label, features should be [feature1, feature2].
    # In our dummy CSV, for the first row these are 1.0 and 3.0.
    np.testing.assert_allclose(feature_vals, [1.0, 3.0])


if __name__ == "__main__":
    # Allow running the tests directly.
    pytest.main([os.path.abspath(__file__)])
