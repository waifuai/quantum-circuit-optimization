# qc/circuit_generation.py
import random
import cirq
from typing import List, Tuple, Dict, Any

class QuantumConfig:
    N_QUBITS = 5
    MIN_GATES = 1
    MAX_GATES = 5
    DEFAULT_NOISE = 0.01
    INVALID_INDEX = -42

def generate_random_circuit(qubits: List[cirq.Qid], n_gates: int) -> cirq.Circuit:
    """
    Generates a random quantum circuit.
    
    Uses a mix of single-qubit and two-qubit gates and appends a measurement at the end.
    """
    circuit = cirq.Circuit()
    single_qubit_gates = [cirq.X, cirq.Y, cirq.Z, cirq.H]
    two_qubit_gates = [cirq.CNOT, cirq.CZ]

    for _ in range(n_gates):
        gate = random.choice(single_qubit_gates + two_qubit_gates)
        if gate in single_qubit_gates:
            qubit = random.choice(qubits)
            circuit.append(gate(qubit))
        else:
            q1, q2 = random.sample(qubits, 2)
            circuit.append(gate(q1, q2))
    
    # Append a measurement for all qubits.
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

def gate_operation_to_dict(op: cirq.Operation) -> Dict[str, Any]:
    """
    Converts a Cirq gate operation to a dictionary representation.
    
    Captures the gate type, qubits involved, and (if available) the gate’s exponent.
    """
    gate_type = type(op.gate).__name__
    op_dict = {
        "gate_type": gate_type,
        "qubits": [str(q) for q in op.qubits],
    }
    if hasattr(op.gate, 'exponent'):
        op_dict['exponent'] = op.gate.exponent
    return op_dict

def circuit_to_dict(circuit: cirq.Circuit) -> Tuple[Dict[str, Any], int]:
    """
    Converts a Cirq circuit into a dictionary of its operations and counts the total number of gates.
    """
    gates_dict = {}
    gate_number = 0
    for moment in circuit:
        for op in moment.operations:
            op_dict = gate_operation_to_dict(op)
            op_dict['gate_number'] = gate_number
            gates_dict[f"gate_{gate_number:02}"] = op_dict
            gate_number += 1
    return gates_dict, gate_number
