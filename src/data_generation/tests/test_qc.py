# tests/test_qc.py

import random
import os
import cirq
import pytest
import tensorflow as tf
import subprocess  # Import subprocess
import hypothesis
from hypothesis import given
from hypothesis import strategies as st

from data_generation.qc.circuit_generation import (
    generate_random_circuit,
    gate_operation_to_dict,
    circuit_to_dict,
    QuantumConfig,
    GateOperationData,
)
from data_generation.qc.optimization import optimize_circuit
from data_generation.qc.simulation import simulate_with_noise, simulate_statevector
from data_generation.scripts.generate_dataset import generate_dataset, circuit_dict_to_tfrecord  # Import necessary functions

# --- Tests for qc/circuit_generation.py ---

@given(n_gates=st.integers(min_value=1, max_value=10))
def test_generate_random_circuit_hypothesis(n_gates):
    # Use a fixed seed for reproducibility.
    random.seed(42)
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Test with default gate set
    circuit = generate_random_circuit(qubits, n_gates)
    measurement_ops = [
        op for op in circuit.all_operations()
        if isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(measurement_ops) > 0, "Circuit must have a measurement operation."
    non_measurement_ops = [
        op for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(non_measurement_ops) >= n_gates, "Circuit should have at least as many gate operations as specified."

def test_gate_operation_to_dict():
    qubit = cirq.GridQubit(0, 0)
    op = cirq.X(qubit)
    op_data: GateOperationData = gate_operation_to_dict(op)
    # Verify that the dictionary contains the gate type and qubit info.
    assert isinstance(op_data, GateOperationData)
    assert op_data.gate_type == "_PauliX" # Adjusted expectation
    assert op_data.qubits == (qubit,)
    # Check the generated dictionary representation item by item
    expected_dict = {'gate_type': '_PauliX', 'qubits': (qubit,), 'exponent': 1.0} # X gate has exponent 1.0
    assert op_data.gate_type == expected_dict['gate_type']
    assert op_data.qubits == expected_dict['qubits']
    assert op_data.exponent == expected_dict['exponent'] # Check against corrected expected_dict
    # If the gate has an exponent attribute, verify that it is captured in the dict.
    if hasattr(op.gate, 'exponent'):
         assert op_data.exponent == op.gate.exponent # Keep this check too
         assert op_data.__dict__['exponent'] == op.gate.exponent

def test_circuit_to_dict():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Build a circuit: one gate per qubit followed by a measurement.
    ops = [cirq.X(q) for q in qubits] + [cirq.measure(*qubits, key='result')]
    circuit = cirq.Circuit(ops)
    gates_dict, num_gates = circuit_to_dict(circuit)
    # The total number of gate operations should equal the number in the circuit.
    assert num_gates == len(list(circuit.all_operations()))
    # Each gate should have a key of the form "gate_XX".
    i = 0
    for key, gate_data in gates_dict.items():
        assert key == f"gate_{i:02}"
        i += 1

# --- Tests for qc/optimization.py ---

def test_optimize_circuit():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Create a simple circuit with one gate per qubit and a measurement.
    circuit = cirq.Circuit([cirq.X(q) for q in qubits] + [cirq.measure(*qubits, key='result')])
    optimized = optimize_circuit(circuit, qubits)
    # Verify that the optimized circuit is still a Cirq Circuit.
    assert isinstance(optimized, cirq.Circuit)
    # Ensure the measurement operation is still present.
    measurement_ops = [
        op for op in optimized.all_operations()
        if isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(measurement_ops) > 0

    # Test with a custom gateset
    custom_gateset = cirq.SqrtIswapTargetGateset()
    optimized_custom = optimize_circuit(circuit, qubits, gateset=custom_gateset)
    assert isinstance(optimized_custom, cirq.Circuit)

# --- Tests for qc/simulation.py ---

def test_simulate_with_noise():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    circuit = cirq.Circuit([cirq.X(q) for q in qubits] + [cirq.measure(*qubits, key='result')])
    counts = simulate_with_noise(circuit)
    # Check that counts is a dictionary with integer keys and counts.
    assert isinstance(counts, dict)
    for key, count in counts.items():
        assert isinstance(key, int)
        assert isinstance(count, int)

    # Test with different noise models
    counts_bit_flip = simulate_with_noise(circuit, noise_model="bit_flip")
    assert isinstance(counts_bit_flip, dict)
    counts_phase_flip = simulate_with_noise(circuit, noise_model="phase_flip")
    assert isinstance(counts_phase_flip, dict)

def test_simulate_statevector():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Create a circuit of Hadamard gates (creates superposition).
    circuit = cirq.Circuit([cirq.H(q) for q in qubits])
    state_probs = simulate_statevector(circuit)
    # Sum of probabilities should be approximately 1.
    total_prob = sum(state_probs.values())
    # Use pytest.approx for floating point comparison
    assert total_prob == pytest.approx(1.0)

# --- Tests for scripts/generate_dataset.py ---

def test_generate_dataset(tmp_path):
    # Generate a small dataset (e.g. 3 circuits).
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    dataset = generate_dataset(3, qubits)
    assert isinstance(dataset, list)
    assert len(dataset) == 3
    # Each dataset entry should contain the expected keys.
    expected_keys = {
        "raw_circuit",
        "gates",
        "num_gates",
        "simulation_counts",
        "optimized_circuit",
    }
    for qc_dict in dataset:
        assert set(qc_dict.keys()) == expected_keys

def test_circuit_dict_to_tfrecord():
    # Create a sample circuit dictionary
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    circuit = generate_random_circuit(qubits, 3)
    gates_dict, num_gates = circuit_to_dict(circuit)
    simulation_counts = simulate_with_noise(circuit)
    optimized = optimize_circuit(circuit, qubits)

    qc_dict = {
        "raw_circuit": str(circuit),
        "gates": list(gates_dict.values()),  # Values are already dicts from circuit_to_dict
        "num_gates": num_gates,
        "simulation_counts": str(simulation_counts), # Serialize the dict to string
        "optimized_circuit": str(optimized)
    }
    # Convert the circuit dictionary to a TFRecord
    example = circuit_dict_to_tfrecord(qc_dict)
    assert isinstance(example, tf.train.Example)

def test_generate_dataset_tfrecord(tmp_path):
    # Test the generate_dataset function and saving to TFRecord format
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    num_circuits = 2
    dataset = generate_dataset(num_circuits, qubits)
    output_file = str(tmp_path / "test_dataset.tfrecord")

    with tf.io.TFRecordWriter(output_file) as writer:
        for qc_dict in dataset:
            example = circuit_dict_to_tfrecord(qc_dict)
            writer.write(example.SerializeToString())

    # Verify that the TFRecord file was created
    assert os.path.exists(output_file)

    # Read the TFRecord file and verify its contents
    record_iterator = tf.data.TFRecordDataset(output_file).as_numpy_iterator()
    count = 0
    for record in record_iterator:
        count += 1
        example = tf.train.Example()
        example.ParseFromString(record)
        assert 'raw_circuit' in example.features.feature
    assert count == num_circuits

def test_generate_dataset_command_line(tmp_path):
    # Test the command-line interface of generate_dataset.py
    output_file = str(tmp_path / "test_dataset_cli.tfrecord")
    # Run the script as a module using -m
    result = subprocess.run([
        "python", "-m", "src.data_generation.scripts.generate_dataset", # Use module path
        "--n_circuits", "5",
        "--min_gates", "1",
        "--max_gates", "3",
        "--n_qubits", "4",
        "--noise_level", "0.02",
        "--output_file", output_file
    ], capture_output=True, text=True)

    # Check that the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Verify that the TFRecord file was created
    assert os.path.exists(output_file)

    # Read the TFRecord file and verify its contents
    record_iterator = tf.data.TFRecordDataset(output_file).as_numpy_iterator()
    count = 0
    for record in record_iterator:
        count += 1
        example = tf.train.Example()
        example.ParseFromString(record)
        assert 'raw_circuit' in example.features.feature
    assert count == 5
