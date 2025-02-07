# utils/circuit_utils.py
import cirq

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
