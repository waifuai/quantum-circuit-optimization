#!/usr/bin/env python
# scripts/generate_dataset.py
import random
import pickle
import os
from tqdm import tqdm
import cirq
from typing import List
from qc.circuit_generation import generate_random_circuit, circuit_to_dict, QuantumConfig
from qc.simulation import simulate_with_noise
from qc.optimization import optimize_circuit

def generate_qc_dict(qubits: List[cirq.Qid], n_gates: int) -> dict:
    """
    Generates a dictionary containing the raw and optimized circuit representations,
    its operation dictionary, and simulation results.
    """
    qc = generate_random_circuit(qubits, n_gates)
    qc_dict = {
        "raw_circuit": str(qc),
    }
    gates_dict, num_gates = circuit_to_dict(qc)
    qc_dict["gates"] = gates_dict
    qc_dict["num_gates"] = num_gates
    qc_dict["simulation_counts"] = simulate_with_noise(qc)
    optimized = optimize_circuit(qc, qubits)
    qc_dict["optimized_circuit"] = str(optimized)
    return qc_dict

def generate_dataset(n_circuits: int, qubits: List[cirq.Qid]) -> list:
    """
    Generates a list (dataset) of quantum circuit dictionaries.
    """
    dataset = []
    for _ in tqdm(range(n_circuits)):
        n_gates = random.randint(QuantumConfig.MIN_GATES, QuantumConfig.MAX_GATES)
        dataset.append(generate_qc_dict(qubits, n_gates))
    return dataset

def main():
    # Define qubits (using a 3x3 grid, taking the first N_QUBITS)
    all_qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    qubits = all_qubits[:QuantumConfig.N_QUBITS]
    
    dataset = generate_dataset(100, qubits)
    
    dataset_path = os.path.join('.', f'cirq_dataset_{random.randint(1, 10**9)}.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {dataset_path}")

if __name__ == "__main__":
    main()
