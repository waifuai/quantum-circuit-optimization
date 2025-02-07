# qc/optimization.py
import cirq
from typing import List, Optional
from qc.circuit_generation import QuantumConfig

def optimize_circuit(circuit: cirq.Circuit, qubits: List[cirq.Qid], gateset: Optional[cirq.GateSet] = None) -> cirq.Circuit:
    """
    Optimizes a quantum circuit using Cirq's optimize_for_target_gateset routine.

    Args:
        circuit: The quantum circuit to optimize.
        qubits: The list of qubits in the circuit.
        gateset: The target gateset to optimize for. If None, defaults to cirq.CZTargetGateset().

    Returns:
        The optimized quantum circuit.
    """
    if gateset is None:
        gateset = cirq.CZTargetGateset()

    optimized = cirq.optimize_for_target_gateset(
        circuit, 
        gateset=gateset,
    )
    return optimized
