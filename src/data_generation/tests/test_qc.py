"""
Test suite for quantum computing core functionality.

This module contains comprehensive tests for quantum circuit generation, simulation,
and optimization functionality. It includes unit tests using pytest and hypothesis
for property-based testing to ensure the reliability and correctness of the
quantum circuit data generation pipeline.
"""

# tests/test_qc.py

import random
import os
import cirq
import pytest
import json # For JSONL tests
import subprocess
import subprocess  # Import subprocess
import hypothesis
from hypothesis import given
from hypothesis import strategies as st

from src.data_generation.qc.circuit_generation import (
    generate_random_circuit,
    gate_operation_to_data,
    circuit_to_operations_data,
    QuantumConfig,
    GateOperationData,
)
from src.data_generation.qc.optimization import optimize_circuit
from src.data_generation.qc.simulation import simulate_with_noise, simulate_statevector
from src.data_generation.scripts.generate_dataset import generate_dataset, generate_qc_dict # Import necessary functions

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

def test_gate_operation_to_data():
    """Tests converting a single gate operation to GateOperationData."""
    qubit = cirq.GridQubit(0, 0)
    op = cirq.X(qubit)
    op_data: GateOperationData = gate_operation_to_data(op)
    # Verify that the dataclass contains the gate type and qubit info as string
    assert isinstance(op_data, GateOperationData)
    assert op_data.gate_type == "_PauliX" # Cirq internal name for X gate type
    assert op_data.qubits_repr == (str(qubit),) # Check string representation
    assert op_data.exponent == 1.0 # X gate has exponent 1.0

    # Test a gate with no explicit exponent attribute (should be None)
    op_measure = cirq.measure(qubit, key='m')
    op_data_measure = gate_operation_to_data(op_measure)
    assert op_data_measure.gate_type == "MeasurementGate"
    assert op_data_measure.exponent is None

def test_circuit_to_operations_data():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Build a circuit: one gate per qubit followed by a measurement.
    ops = [cirq.X(q) for q in qubits] + [cirq.measure(*qubits, key='result')]
    circuit = cirq.Circuit(ops)
    operations_data, num_gates = circuit_to_operations_data(circuit)
    # For this project, num_gates counts non-measurement gates only.
    assert num_gates == len(qubits)
    # Each gate should map to a GateOperationData entry.
    for op_data in operations_data:
        assert isinstance(op_data, GateOperationData)

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

def test_generate_qc_dict():
    """ Test the creation of a single circuit data dictionary """
    n_qubits = 2
    n_gates = 3
    noise = 0.01
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    qc_dict = generate_qc_dict(qubits, n_gates, noise)

    assert isinstance(qc_dict, dict)
    expected_keys = {
        "raw_circuit",
        "operations", # List of dicts
        "num_gates", # int
        "simulation_counts", # dict
        "optimized_circuit",
    }
    assert set(qc_dict.keys()) == expected_keys

    # Check types of values
    assert isinstance(qc_dict["raw_circuit"], str)
    assert isinstance(qc_dict["operations"], list)
    if qc_dict["operations"]: # If not empty
        assert isinstance(qc_dict["operations"][0], dict) # operations are dicts
        assert "gate_type" in qc_dict["operations"][0]
        assert "qubits_repr" in qc_dict["operations"][0]
    assert isinstance(qc_dict["num_gates"], int)
    assert isinstance(qc_dict["simulation_counts"], dict)
    assert isinstance(qc_dict["optimized_circuit"], str)

def test_generate_dataset():
    """ Test generating a list of circuit dictionaries """
    n_qubits = 2
    n_circuits = 3
    min_g, max_g = 1, 3
    noise = 0.01
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    dataset = generate_dataset(n_circuits, qubits, min_g, max_g, noise)
    assert isinstance(dataset, list)
    assert len(dataset) == n_circuits
    # Check first item conforms
    if n_circuits > 0:
        assert isinstance(dataset[0], dict)
        expected_keys = {
            "raw_circuit", "operations", "num_gates",
            "simulation_counts", "optimized_circuit"
        }
        assert set(dataset[0].keys()) == expected_keys

def test_generate_dataset_jsonl(tmp_path):
    """ Test the generate_dataset function and saving to JSON Lines format """
    n_qubits = 2
    num_circuits = 2
    min_g, max_g = 1, 2
    noise = 0.01
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    dataset = generate_dataset(num_circuits, qubits, min_g, max_g, noise)
    output_file = str(tmp_path / "test_dataset.jsonl") # Use .jsonl extension

    # Write the dataset to JSON Lines
    try:
        with open(output_file, 'w') as f:
            for qc_dict in dataset:
                json.dump(qc_dict, f)
                f.write('\n')
    except Exception as e:
        pytest.fail(f"Writing to JSONL failed: {e}")

    # Verify that the JSONL file was created
    assert os.path.exists(output_file)

    # Read the JSONL file and verify its contents
    read_count = 0
    try:
        with open(output_file, 'r') as f:
            for line in f:
                # Skip empty lines if any
                if not line.strip():
                    continue
                read_count += 1
                try:
                    data = json.loads(line)
                    # Basic check for expected structure
                    assert 'raw_circuit' in data
                    assert 'operations' in data
                    assert 'num_gates' in data
                    assert 'simulation_counts' in data
                    assert 'optimized_circuit' in data
                except json.JSONDecodeError:
                    pytest.fail(f"Failed to decode JSON line: {line.strip()}")
                except AssertionError as e:
                    pytest.fail(f"Assertion failed for loaded data: {e}\nData: {data}")
    except Exception as e:
        pytest.fail(f"Reading or verifying JSONL failed: {e}")

    assert read_count == num_circuits, f"Expected {num_circuits} lines, found {read_count}"

def test_generate_dataset_command_line(tmp_path):
    """ Test the command-line interface of generate_dataset.py """
    import sys
    output_file = str(tmp_path / "test_dataset_cli.jsonl") # Use .jsonl extension
    num_circuits_cli = 5
    command = [
        sys.executable,
        "-m", "src.data_generation.scripts.generate_dataset",
        "--n_circuits", str(num_circuits_cli),
        "--min_gates", "1",
        "--max_gates", "3",
        "--n_qubits", "4",
        "--noise_level", "0.02",
        "--output_file", output_file
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert os.path.exists(output_file), f"Output file {output_file} not found."

    # Read the JSONL file and verify its contents
    read_count = 0
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                read_count += 1
                try:
                    data = json.loads(line)
                    assert 'raw_circuit' in data
                except json.JSONDecodeError:
                    pytest.fail(f"Failed to decode JSON line in CLI test: {line.strip()}")
    except Exception as e:
        pytest.fail(f"Reading or verifying JSONL from CLI test failed: {e}")

    assert read_count == num_circuits_cli, f"Expected {num_circuits_cli} circuits in CLI output file, found {read_count}"
