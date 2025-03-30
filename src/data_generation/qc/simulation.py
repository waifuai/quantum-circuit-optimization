import cirq
import numpy as np
from typing import Dict

from data_generation.qc.circuit_generation import QuantumConfig

def simulate_with_noise(
    circuit: cirq.Circuit,
    noise: float = QuantumConfig.DEFAULT_NOISE,
    noise_model: str = "depolarize",
) -> Dict[int, int]:
    """
    Simulates the circuit under noise and returns a histogram
    of measurement results (using 1000 repetitions).

    Args:
        circuit: The quantum circuit to simulate.
        noise: The noise level.
        noise_model: The type of noise model to use. Can be "depolarize", "bit_flip", or "phase_flip".

    Returns:
        A dictionary representing the histogram of measurement results.
    """
    # Apply the specified noise model
    if noise_model == "depolarize":
        noise_obj: cirq.NoiseModel = cirq.depolarize(noise)
    elif noise_model == "bit_flip":
        noise_obj = cirq.bit_flip(noise)
    elif noise_model == "phase_flip":
        noise_obj = cirq.phase_flip(noise)
    else:
        raise ValueError(f"Invalid noise model: {noise_model}")

    # Apply noise to the circuit
    noisy_circuit: cirq.Circuit = circuit.with_noise(noise_obj)
    simulator: cirq.Simulator = cirq.Simulator()
    result: cirq.Result = simulator.run(noisy_circuit, repetitions=1000)
    return result.histogram(key='result')


def simulate_statevector(circuit: cirq.Circuit, repetitions: int = 1024) -> Dict[str, float]:
    """
    Simulates the circuit without noise and returns a dictionary mapping
    computational basis states (as bitstrings) to their probability.
    """
    simulator: cirq.Simulator = cirq.Simulator()
    result: cirq.Result = simulator.simulate(circuit)
    statevector: np.ndarray = result.final_state_vector
    n_qubits: int = len(circuit.all_qubits())
    
    # Use NumPy for vectorized probability calculation
    probabilities: np.ndarray = np.abs(statevector)**2
    state_probs: Dict[str, float] = {
        f"state_{bin(i)[2:].zfill(n_qubits)}": prob
        for i, prob in enumerate(probabilities)
    }
    return state_probs
