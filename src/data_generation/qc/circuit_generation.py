"""
Quantum circuit generation and representation module.

This module provides functionality for generating random quantum circuits using Cirq,
converting them to structured data formats, and extracting gate operation information.
It includes utilities for circuit creation, gate operation data conversion, and
circuit analysis for quantum circuit optimization workflows.
"""

import random
import cirq
from typing import List, Tuple, Dict, Any, Sequence, Optional
from dataclasses import dataclass
from typing import Final

class QuantumConfig:
    """
    Configuration parameters for quantum circuit generation.
    """
    N_QUBITS: Final[int] = 5
    MIN_GATES: Final[int] = 1
    MAX_GATES: Final[int] = 5
    DEFAULT_NOISE: Final[float] = 0.01
    INVALID_INDEX: Final[int] = -42

def generate_random_circuit(
    qubits: List[cirq.Qid],
    n_gates: int,
    gate_set: Optional[Sequence[cirq.Gate]] = None
) -> cirq.Circuit:
    """
    Generates a random quantum circuit.

    Args:
        qubits: A list of `cirq.Qid` objects representing the qubits in the circuit.
        n_gates: The number of gates to apply to the circuit.
        gate_set: An optional sequence of `cirq.Gate` objects to choose from.
            If None, a default set of single-qubit and two-qubit gates is used.

    Returns:
        A `cirq.Circuit` object representing the generated circuit.
    """
    circuit: cirq.Circuit = cirq.Circuit()
    if gate_set is None:
        # Use a default gate set if none is provided
        single_qubit_gates: List[cirq.Gate] = [cirq.X, cirq.Y, cirq.Z, cirq.H]
        two_qubit_gates: List[cirq.Gate] = [cirq.CNOT, cirq.CZ]
        gates: List[cirq.Gate] = single_qubit_gates + two_qubit_gates
    else:
        gates: Sequence[cirq.Gate] = gate_set

    for _ in range(n_gates):
        gate: cirq.Gate = random.choice(gates)
        if gate.num_qubits() == 1:
            qubit: cirq.Qid = random.choice(qubits)
            circuit.append(gate(qubit))
        elif gate.num_qubits() == 2 and len(qubits) >= 2:
            # Choose two distinct qubits for the two-qubit gate
            q1: cirq.Qid
            q2: cirq.Qid
            q1, q2 = random.sample(qubits, 2)
            circuit.append(gate(q1, q2))
        else:
            raise ValueError(f"Unsupported gate with {gate.num_qubits()} qubits or not enough qubits.")

    # Append a measurement for all qubits.
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

@dataclass(frozen=True)
class GateOperationData:
    """
    A dataclass representing a Cirq gate operation, ready for JSON serialization.

    Attributes:
        gate_type: The name of the gate type.
        qubits_repr: A tuple of strings representing the qubits the gate operates on.
        exponent: The exponent of the gate, if applicable.
    """
    gate_type: str
    qubits_repr: Tuple[str, ...]
    exponent: Optional[float] = None

def gate_operation_to_data(op: cirq.Operation) -> GateOperationData:
    """
    Converts a Cirq gate operation to a GateOperationData dataclass.

    Args:
        op: The `cirq.Operation` to convert.

    Returns:
        A `GateOperationData` object representing the operation.
    """
    gate_type: str
    # Handle potential PauliString gate representation
    if hasattr(op.gate, 'pauli_string'):
        gate_type = f"PauliStringGate_{op.gate.pauli_string}"
    else:
        gate_type = type(op.gate).__name__
    qubits_repr: Tuple[str, ...] = tuple(str(q) for q in op.qubits)
    exponent: Optional[float] = getattr(op.gate, 'exponent', None)
    # Handle special cases where exponent might be non-numeric (like ParameterizedValue)
    if not isinstance(exponent, (int, float)):
        exponent = None
    return GateOperationData(gate_type=gate_type, qubits_repr=qubits_repr, exponent=exponent)

def circuit_to_operations_data(circuit: cirq.Circuit) -> Tuple[List[GateOperationData], int]:
    """
    Converts a Cirq circuit into a list of its operations data and counts the total number of gates (excluding measurements).

    Args:
        circuit: The `cirq.Circuit` to convert.

    Returns:
        A tuple containing:
            - A list of GateOperationData objects representing the gate operations.
            - The total number of gates in the circuit (excluding measurements for this count).
    """
    operations_data: List[GateOperationData] = []
    gate_count: int = 0
    for moment in circuit:
        for op in moment.operations:
            op_data: GateOperationData = gate_operation_to_data(op)
            operations_data.append(op_data)
            if not isinstance(op.gate, cirq.MeasurementGate):
                gate_count += 1 # Only count non-measurement gates towards 'num_gates'
    return operations_data, gate_count
