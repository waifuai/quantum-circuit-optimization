# qc/simulation.py
import cirq
from typing import Dict
from qc.circuit_generation import QuantumConfig

def simulate_with_noise(circuit: cirq.Circuit, noise: float = QuantumConfig.DEFAULT_NOISE) -> Dict[int, int]:
    """
    Simulates the circuit under depolarizing noise and returns a histogram
    of measurement results (using 1000 repetitions).
    """
    noise_model = cirq.depolarize(noise)
    noisy_circuit = circuit.with_noise(noise_model)
    simulator = cirq.Simulator()
    result = simulator.run(noisy_circuit, repetitions=1000)
    return result.histogram(key='result')

def simulate_statevector(circuit: cirq.Circuit, repetitions: int = 1024) -> Dict[str, float]:
    """
    Simulates the circuit without noise and returns a dictionary mapping
    computational basis states (as bitstrings) to their probability.
    """
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    statevector = result.final_state_vector
    n_qubits = len(circuit.all_qubits())
    state_probs = {
        f"state_{bin(i)[2:].zfill(n_qubits)}": abs(ampl) ** 2 
        for i, ampl in enumerate(statevector)
    }
    return state_probs
