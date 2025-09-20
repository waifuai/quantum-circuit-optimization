"""
Quantum circuit optimization module.

This module provides functionality for optimizing quantum circuits using Cirq's
built-in optimization routines. It supports various target gatesets and provides
a unified interface for circuit optimization within the quantum circuit generation
and optimization pipeline.
"""

import cirq
from typing import List, Optional

def optimize_circuit(
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid],
    gateset: Optional[cirq.Gateset] = None,
) -> cirq.Circuit:
    """
    Optimizes a quantum circuit using Cirq's `optimize_for_target_gateset` routine.

    Args:
        circuit: The quantum circuit to optimize.
        qubits: The list of qubits in the circuit.
        gateset: The target gateset to optimize for. If None, defaults to `cirq.CZTargetGateset()`.

    Returns:
        The optimized quantum circuit.
    """
    # Use CZTargetGateset if no gateset is specified
    target_gateset: cirq.Gateset
    if gateset is None:
        target_gateset = cirq.CZTargetGateset()
    else:
        target_gateset = gateset

    # Optimize the circuit for the given gateset
    optimized: cirq.Circuit = cirq.optimize_for_target_gateset(
        circuit,
        gateset=target_gateset,
    )
    return optimized
