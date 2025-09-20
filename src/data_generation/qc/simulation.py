"""
Quantum circuit simulation module.

This module provides functionality for simulating quantum circuits with and without noise
using Cirq's simulation capabilities. It supports various noise models and provides
utilities for extracting measurement results and statevector probabilities from
simulated quantum circuits.
"""

import cirq
import numpy as np
import logging
from typing import Dict, Optional

from data_generation.qc.circuit_generation import QuantumConfig

logger = logging.getLogger(__name__)

def simulate_with_noise(
    circuit: cirq.Circuit,
    noise: float = QuantumConfig.DEFAULT_NOISE,
    noise_model: str = "depolarize",
    repetitions: int = 1000,
) -> Dict[int, int]:
    """
    Simulates the circuit under noise and returns a histogram
    of measurement results.

    Args:
        circuit: The quantum circuit to simulate.
        noise: The noise level (must be between 0 and 1).
        noise_model: The type of noise model to use.
            Options: "depolarize", "bit_flip", "phase_flip".
        repetitions: Number of simulation repetitions.

    Returns:
        A dictionary representing the histogram of measurement results.

    Raises:
        ValueError: If noise level is invalid or noise model is unsupported.
        RuntimeError: If simulation fails.
    """
    if not 0 <= noise <= 1:
        raise ValueError(f"Noise level must be between 0 and 1, got {noise}")

    if repetitions <= 0:
        raise ValueError(f"Repetitions must be positive, got {repetitions}")

    # Validate noise model
    supported_models = {"depolarize", "bit_flip", "phase_flip"}
    if noise_model not in supported_models:
        raise ValueError(f"Unsupported noise model: {noise_model}. "
                        f"Supported: {supported_models}")

    logger.debug(f"Simulating circuit with {noise_model} noise (level: {noise})")

    try:
        # Apply the specified noise model
        if noise_model == "depolarize":
            noise_obj: cirq.NoiseModel = cirq.depolarize(noise)
        elif noise_model == "bit_flip":
            noise_obj = cirq.bit_flip(noise)
        elif noise_model == "phase_flip":
            noise_obj = cirq.phase_flip(noise)

        # Apply noise to the circuit
        noisy_circuit: cirq.Circuit = circuit.with_noise(noise_obj)
        simulator: cirq.Simulator = cirq.Simulator()
        result: cirq.Result = simulator.run(noisy_circuit, repetitions=repetitions)

        histogram = result.histogram(key='result')
        logger.debug(f"Simulation completed with {len(histogram)} unique outcomes")
        return histogram

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise RuntimeError(f"Circuit simulation failed: {e}") from e


def simulate_statevector(circuit: cirq.Circuit, repetitions: int = 1024) -> Dict[str, float]:
    """
    Simulates the circuit without noise and returns a dictionary mapping
    computational basis states (as bitstrings) to their probability.

    Args:
        circuit: The quantum circuit to simulate.
        repetitions: Number of simulation repetitions (for compatibility, not used in statevector simulation).

    Returns:
        A dictionary mapping state bitstrings to their probabilities.

    Raises:
        RuntimeError: If statevector simulation fails.
        ValueError: If circuit is invalid.
    """
    if not circuit:
        raise ValueError("Circuit cannot be empty")

    logger.debug("Starting statevector simulation")

    try:
        simulator: cirq.Simulator = cirq.Simulator()
        result: cirq.Result = simulator.simulate(circuit)
        statevector: np.ndarray = result.final_state_vector
        n_qubits: int = len(circuit.all_qubits())

        if n_qubits == 0:
            raise ValueError("Circuit must have at least one qubit")

        # Use NumPy for vectorized probability calculation
        probabilities: np.ndarray = np.abs(statevector)**2

        # Create state mapping with proper formatting
        state_probs: Dict[str, float] = {
            f"state_{bin(i)[2:].zfill(n_qubits)}": float(prob)
            for i, prob in enumerate(probabilities)
            if prob > 1e-10  # Filter out negligible probabilities
        }

        logger.debug(f"Statevector simulation completed for {n_qubits} qubits, "
                    f"{len(state_probs)} non-zero states")
        return state_probs

    except Exception as e:
        logger.error(f"Statevector simulation failed: {e}")
        raise RuntimeError(f"Statevector simulation failed: {e}") from e
