#!/usr/bin/env python
# scripts/generate_dataset.py
import random
import os
import argparse
from tqdm import tqdm
import cirq
import json # Use json for JSONL output
import sys
# Add project root to path to allow imports when run as script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import List, Dict, Any
from src.data_generation.qc.circuit_generation import generate_random_circuit, circuit_to_operations_data, QuantumConfig, GateOperationData
from src.data_generation.qc.simulation import simulate_with_noise
from src.data_generation.qc.optimization import optimize_circuit


from dataclasses import asdict

def generate_qc_dict(qubits: List[cirq.Qid], n_gates: int, noise_level: float) -> Dict[str, Any]:
    """
    Generates a dictionary containing the raw and optimized circuit representations,
    its operation list (as dicts), gate count, and simulation results.

    Args:
        qubits: A list of `cirq.Qid` objects representing the qubits in the circuit.
        n_gates: The number of non-measurement gates to attempt to apply to the circuit.
        noise_level: The noise level for simulation.

    Returns:
        A dictionary containing the circuit information, ready for JSON serialization.
    """
    # Generate a random circuit
    qc: cirq.Circuit = generate_random_circuit(qubits, n_gates)

    # Convert the circuit to a list of operation data objects and get gate count
    operations_data_list, num_gates = circuit_to_operations_data(qc)
    operations_list_dict = [asdict(op_data) for op_data in operations_data_list]

    # Simulate the circuit with noise
    simulation_counts: Dict[int, int] = simulate_with_noise(qc, noise=noise_level)

    # Optimize the circuit
    optimized: cirq.Circuit = optimize_circuit(qc, qubits)

    qc_dict: Dict[str, Any] = {
        "raw_circuit": str(qc),
        "operations": operations_list_dict, # Store list of dicts
        "num_gates": num_gates, # Store the gate count (excluding measurements)
        "simulation_counts": simulation_counts, # Store the actual dict
        "optimized_circuit": str(optimized)
    }
    return qc_dict


def generate_dataset(n_circuits: int, qubits: List[cirq.Qid], min_gates: int, max_gates: int, noise_level: float) -> List[Dict[str, Any]]:
    """
    Generates a list (dataset) of quantum circuit dictionaries.

    Args:
        n_circuits: The number of circuits to generate.
        qubits: The list of qubits to use in the circuits.
        min_gates: Minimum number of non-measurement gates per circuit.
        max_gates: Maximum number of non-measurement gates per circuit.
        noise_level: Noise level for simulation.

    Returns:
        A list of quantum circuit dictionaries.
    """
    dataset: List[Dict[str, Any]] = []
    for _ in tqdm(range(n_circuits), desc="Generating Circuits"):
        n_gates: int = random.randint(min_gates, max_gates)
        dataset.append(generate_qc_dict(qubits, n_gates, noise_level))
    return dataset




def main() -> None:
    """Main function to generate and save a quantum circuit dataset as JSON Lines."""
    parser = argparse.ArgumentParser(description="Generate a quantum circuit dataset.")
    parser.add_argument("--n_circuits", type=int, default=100, help="Number of circuits to generate.")
    parser.add_argument("--min_gates", type=int, default=QuantumConfig.MIN_GATES, help="Minimum number of non-measurement gates.")
    parser.add_argument("--max_gates", type=int, default=QuantumConfig.MAX_GATES, help="Maximum number of non-measurement gates.")
    parser.add_argument("--n_qubits", type=int, default=QuantumConfig.N_QUBITS, help="Number of qubits.")
    parser.add_argument("--noise_level", type=float, default=QuantumConfig.DEFAULT_NOISE, help="Noise level for simulation.")
    parser.add_argument("--output_file", type=str, default="cirq_dataset.jsonl", help="Output JSON Lines file name.")

    args = parser.parse_args()

    # Define qubits (using a 3x3 grid, taking the first N_QUBITS)
    all_qubits: List[cirq.GridQubit] = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    qubits: List[cirq.GridQubit] = all_qubits[:args.n_qubits]

    # Generate the dataset in memory
    dataset: List[Dict[str, Any]] = generate_dataset(
        args.n_circuits, qubits, args.min_gates, args.max_gates, args.noise_level
    )

    # Save the dataset to JSON Lines format
    try:
        with open(args.output_file, 'w') as f:
            for qc_dict in tqdm(dataset, desc="Writing JSONL"):
                json.dump(qc_dict, f) # Write JSON object
                f.write('\n') # Add newline to separate objects
        print(f"Dataset saved to {args.output_file}")
    except IOError as e:
        print(f"Error writing dataset to {args.output_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
