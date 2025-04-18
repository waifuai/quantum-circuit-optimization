import random
import cirq
from typing import List, Tuple, Dict, Any, Sequence, Optional
from dataclasses import dataclass

class QuantumConfig:
    """
    Configuration parameters for quantum circuit generation.
    """
    N_QUBITS = 5
    MIN_GATES = 1
    MAX_GATES = 5
    DEFAULT_NOISE = 0.01
    INVALID_INDEX = -42

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
    A dataclass representing a Cirq gate operation.

    Attributes:
        gate_type: The name of the gate type.
        qubits: A tuple of `cirq.Qid` objects representing the qubits the gate operates on.
        exponent: The exponent of the gate, if applicable.
    """
    gate_type: str
    qubits: Tuple[cirq.Qid, ...]
    exponent: Optional[float] = None

def gate_operation_to_data(op: cirq.Operation) -> GateOperationData:
    """
    Converts a Cirq gate operation to a GateOperationData dataclass.

    Args:
        op: The `cirq.Operation` to convert.

    Returns:
        A `GateOperationData` object representing the operation.
    """
    gate_type: str = type(op.gate).__name__
    qubits_repr: Tuple[str, ...] = tuple(str(q) for q in op.qubits) # Use string representation
    exponent: Optional[float] = getattr(op.gate, 'exponent', None)
    return GateOperationData(gate_type=gate_type, qubits_repr=qubits_repr, exponent=exponent)

def circuit_to_operations_data(circuit: cirq.Circuit) -> Tuple[List[GateOperationData], int]:
    """
    Converts a Cirq circuit into a dictionary of its operations and counts the total number of gates.

    Args:
        circuit: The `cirq.Circuit` to convert.

    Returns:
        A tuple containing:
            - A dictionary where keys are gate identifiers (e.g., "gate_00") and values are dictionaries
              representing the gate operations.
            - The total number of gates in the circuit.
    """
    operations_data: List[GateOperationData] = []
    gate_number: int = 0
    for moment in circuit:
        for op in moment.operations:
            # Exclude measurement gates from the count and data list if desired,
            # but currently including them for a full operation list.
            op_data: GateOperationData = gate_operation_to_data(op)
            operations_data.append(op_data)
            gate_number += 1
    return operations_data, gate_number
