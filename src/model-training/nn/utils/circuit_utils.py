# utils/circuit_utils.py
import cirq
import tensorflow as tf
import numpy as np

def create_circuit(params, num_qubits=5, qubits=None):
    """Creates a parameterized quantum circuit.
    
    If qubits is None, uses a default 5-qubit register.
    """
    if qubits is None:
        qubits = cirq.LineQubit.range(num_qubits)
        qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()
    param_idx = 0
    for _ in range(5):  # 5 layers
        for j in range(num_qubits):
            circuit.append(cirq.rx(params[param_idx])(qubits[j]))
            param_idx += 1
        for j in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[j], qubits[j+1]))
    return circuit

def calculate_loss(circuit, target_state):
    """Calculates loss as (1 - fidelity) between the circuit output and target state."""
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    final_state = result.final_state_vector
    fidelity = cirq.fidelity(final_state, target_state)
    return 1 - fidelity  # Lower loss is better

def simulate_circuit(circuit):
    """Simulates the circuit and returns the final state vector."""
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state_vector = result.final_state_vector
    return state_vector

def calculate_fidelity(state_vector, target_state):
    """Calculates the fidelity between the state vector and the target state."""
    state_vector = state_vector / np.linalg.norm(state_vector)
    target_state = target_state / np.linalg.norm(target_state)
    fidelity = tf.abs(tf.tensordot(tf.cast(tf.math.conj(target_state), dtype=tf.complex128), tf.cast(state_vector, dtype=tf.complex128), axes=1))**2
    return fidelity
