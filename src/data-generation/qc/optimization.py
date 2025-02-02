# qc/optimization.py
import cirq
from typing import List
from qc.circuit_generation import QuantumConfig

def optimize_circuit(circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> cirq.Circuit:
    """
    Optimizes a quantum circuit using Cirq's optimize_for_target_gateset routine.
    """
    optimized = cirq.optimize_for_target_gateset(
        circuit, 
        gateset=cirq.CZTargetGateset(),
    )
    return optimized
