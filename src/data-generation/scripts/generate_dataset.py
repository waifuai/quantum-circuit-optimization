#!/usr/bin/env python
# scripts/generate_dataset.py
import random
import os
import argparse
from tqdm import tqdm
import cirq
import tensorflow as tf
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
    gates_dict, num_gates = circuit_to_dict(qc)
    simulation_counts = simulate_with_noise(qc)
    optimized = optimize_circuit(qc, qubits)

    qc_dict = {
        "raw_circuit": str(qc),
        "gates": [gate.__dict__ for gate in gates_dict.values()],  # Convert dataclasses to dicts
        "num_gates": num_gates,
        "simulation_counts": str(simulation_counts), # Serialize the dict to string
        "optimized_circuit": str(optimized)
    }
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

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def circuit_dict_to_tfrecord(qc_dict: dict) -> tf.train.Example:
    """
    Converts a quantum circuit dictionary to a TFRecord Example.
    """
    feature = {
        'raw_circuit': _bytes_feature(qc_dict['raw_circuit'].encode('utf-8')),
        'num_gates': _int64_feature(qc_dict['num_gates']),
        'simulation_counts': _bytes_feature(qc_dict['simulation_counts'].encode('utf-8')), # Serialize the dict to string
        'optimized_circuit': _bytes_feature(qc_dict['optimized_circuit'].encode('utf-8'))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main():
    parser = argparse.ArgumentParser(description="Generate a quantum circuit dataset.")
    parser.add_argument("--n_circuits", type=int, default=100, help="Number of circuits to generate.")
    parser.add_argument("--min_gates", type=int, default=QuantumConfig.MIN_GATES, help="Minimum number of gates.")
    parser.add_argument("--max_gates", type=int, default=QuantumConfig.MAX_GATES, help="Maximum number of gates.")
    parser.add_argument("--n_qubits", type=int, default=QuantumConfig.N_QUBITS, help="Number of qubits.")
    parser.add_argument("--noise_level", type=float, default=QuantumConfig.DEFAULT_NOISE, help="Noise level.")
    parser.add_argument("--output_file", type=str, default="cirq_dataset.tfrecord", help="Output file name.")

    args = parser.parse_args()

    # Define qubits (using a 3x3 grid, taking the first N_QUBITS)
    all_qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    qubits = all_qubits[:args.n_qubits]
    
    # Override QuantumConfig values with command-line arguments
    QuantumConfig.N_QUBITS = args.n_qubits
    QuantumConfig.MIN_GATES = args.min_gates
    QuantumConfig.MAX_GATES = args.max_gates

    dataset = generate_dataset(args.n_circuits, qubits)
    
    # Save the dataset to TFRecord format
    with tf.io.TFRecordWriter(args.output_file) as writer:
        for qc_dict in dataset:
            example = circuit_dict_to_tfrecord(qc_dict)
            writer.write(example.SerializeToString())

    print(f"Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
