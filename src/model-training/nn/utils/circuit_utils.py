# utils/circuit_utils.py
import cirq
import tensorflow as tf
import numpy as np

def create_circuit(params, num_qubits=5, qubits=None, circuit_type='default'):
    """Creates a parameterized quantum circuit.
    
    Args:
        params: Parameters for the circuit gates.
        num_qubits: Number of qubits in the circuit.
        qubits: Optional list of qubits. If not provided, uses a default 5-qubit register.
        circuit_type: Type of circuit to create ('default').

    Returns:
        A cirq.Circuit object.
    """
    if qubits is None:
        qubits = cirq.LineQubit.range(num_qubits)

    circuit = cirq.Circuit()
    param_idx = 0

    if circuit_type == 'default':
        for _ in range(5):  # 5 layers
            for j in range(num_qubits):
                circuit.append(cirq.rx(params[param_idx])(qubits[j]))
                param_idx += 1
            for j in range(num_qubits - 1):
                circuit.append(cirq.CNOT(qubits[j], qubits[j+1]))
    else:
        raise ValueError(f"Invalid circuit_type: {circuit_type}")

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

def calculate_fidelity_loss(y_true, y_pred, num_qubits, circuit_type):
    """
    Calculates the fidelity between the output statevector
    of the quantum circuit and the target state.
    """
    # Reshape y_pred to (batch_size, num_params)
    num_params = num_qubits * 5 #hardcoded layers
    y_pred = tf.reshape(y_pred, (-1, num_params))

    fidelities = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
    for i in tf.range(tf.shape(y_true)[0]):
        # Create the quantum circuit with predicted parameters
        circuit = create_circuit(y_pred[i], num_qubits=num_qubits, circuit_type=circuit_type)

        # Simulate the circuit and get the final state vector
        state_vector = simulate_circuit(circuit)

        # Calculate fidelity
        target_state = y_true[i]
        fidelity = calculate_fidelity(state_vector, target_state)
        fidelities = fidelities.write(i, fidelity)

    # Return the mean loss (1 - fidelities.stack())
    return tf.reduce_mean(1 - fidelities.stack())
