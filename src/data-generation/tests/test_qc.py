# tests/test_qc.py

import random
import os
import pickle
import cirq
import pytest

from qc.circuit_generation import (
    generate_random_circuit,
    gate_operation_to_dict,
    circuit_to_dict,
    QuantumConfig,
)
from qc.optimization import optimize_circuit
from qc.simulation import simulate_with_noise, simulate_statevector

# --- Tests for qc/circuit_generation.py ---

def test_generate_random_circuit():
    # Use a fixed seed for reproducibility.
    random.seed(42)
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    n_gates = 3
    circuit = generate_random_circuit(qubits, n_gates)
    # Check that the circuit contains at least one measurement operation.
    measurement_ops = [
        op for op in circuit.all_operations()
        if isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(measurement_ops) > 0, "Circuit must have a measurement operation."
    # Check that there are at least n_gates non-measurement operations.
    non_measurement_ops = [
        op for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(non_measurement_ops) >= n_gates, "Circuit should have at least as many gate operations as specified."

def test_gate_operation_to_dict():
    qubit = cirq.GridQubit(0, 0)
    op = cirq.X(qubit)
    op_dict = gate_operation_to_dict(op)
    # Verify that the dictionary contains the gate type and qubit info.
    assert "gate_type" in op_dict and isinstance(op_dict["gate_type"], str)
    assert "qubits" in op_dict and isinstance(op_dict["qubits"], list)
    assert str(qubit) in op_dict["qubits"]
    # If the gate has an exponent attribute, verify that it is captured.
    if hasattr(op.gate, 'exponent'):
        assert "exponent" in op_dict
        assert op_dict["exponent"] == op.gate.exponent

def test_circuit_to_dict():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Build a circuit: one gate per qubit followed by a measurement.
    ops = [cirq.X(q) for q in qubits] + [cirq.measure(*qubits, key='result')]
    circuit = cirq.Circuit(ops)
    gates_dict, num_gates = circuit_to_dict(circuit)
    # The total number of gate operations should equal the number in the circuit.
    assert num_gates == len(list(circuit.all_operations()))
    # Each gate should have a key of the form "gate_XX".
    for i in range(num_gates):
        key = f"gate_{i:02}"
        assert key in gates_dict

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

def test_simulate_statevector():
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Create a circuit of Hadamard gates (creates superposition).
    circuit = cirq.Circuit([cirq.H(q) for q in qubits])
    state_probs = simulate_statevector(circuit)
    # Sum of probabilities should be approximately 1.
    total_prob = sum(state_probs.values())
    assert abs(total_prob - 1) < 1e-6

# --- Tests for scripts/generate_dataset.py ---

def test_generate_dataset(tmp_path):
    # Import the generate_dataset function from the scripts module.
    from scripts.generate_dataset import generate_dataset
    qubits = [cirq.GridQubit(0, i) for i in range(QuantumConfig.N_QUBITS)]
    # Generate a small dataset (e.g. 3 circuits).
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
        assert expected_keys.issubset(qc_dict.keys())
    # Test saving and loading the dataset using pickle.
    dataset_path = tmp_path / "test_dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    with open(dataset_path, "rb") as f:
        loaded_dataset = pickle.load(f)
    assert loaded_dataset == dataset
