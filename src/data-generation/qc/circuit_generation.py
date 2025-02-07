# qc/circuit_generation.py
import random
import cirq
from typing import List, Tuple, Dict, Any, Sequence
from dataclasses import dataclass

class QuantumConfig:
    N_QUBITS = 5
    MIN_GATES = 1
    MAX_GATES = 5
    DEFAULT_NOISE = 0.01
    INVALID_INDEX = -42

def generate_random_circuit(qubits: List[cirq.Qid], n_gates: int, gate_set: Sequence[cirq.Gate] = None) -> cirq.Circuit:
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
    circuit = cirq.Circuit()
    if gate_set is None:
        single_qubit_gates = [cirq.X, cirq.Y, cirq.Z, cirq.H]
        two_qubit_gates = [cirq.CNOT, cirq.CZ]
        gates = single_qubit_gates + two_qubit_gates
    else:
        gates = gate_set

    for _ in range(n_gates):
        gate = random.choice(gates)
        if gate.num_qubits() == 1:
            qubit = random.choice(qubits)
            circuit.append(gate(qubit))
        elif gate.num_qubits() == 2:
            q1, q2 = random.sample(qubits, 2)
            circuit.append(gate(q1, q2))
        else:
            raise ValueError(f"Gate {gate} has an unsupported number of qubits: {gate.num_qubits()}")
    
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
    exponent: float = None

def gate_operation_to_dict(op: cirq.Operation) -> GateOperationData:
    """
    Converts a Cirq gate operation to a GateOperationData dataclass.

    Args:
        op: The `cirq.Operation` to convert.

    Returns:
        A `GateOperationData` object representing the operation.
    """
    gate_type = type(op.gate).__name__
    qubits = tuple(op.qubits)
    exponent = getattr(op.gate, 'exponent', None)
    return GateOperationData(gate_type=gate_type, qubits=qubits, exponent=exponent)

def circuit_to_dict(circuit: cirq.Circuit) -> Tuple[Dict[str, Any], int]:
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
    gates_dict = {}
    gate_number = 0
    for moment in circuit:
        for op in moment.operations:
            op_data = gate_operation_to_dict(op)
            gates_dict[f"gate_{gate_number:02}"] = op_data.__dict__
            gate_number += 1
    return gates_dict, gate_number
