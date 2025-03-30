"""
CSV Preprocessing

Normalizes features in CSV files using either shard-based or memory-based methods.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from src import config


def generate_statevector_keys(num_qubits: int) -> List[str]:
    """
    Generates a list of statevector keys.

    Args:
        num_qubits: The number of qubits.

    Returns:
        A list of statevector keys.
    """
    return [f"statevector_{bin(i)[2:].zfill(num_qubits)}" for i in range(2**num_qubits)]


def generate_gate_keys(num_gates: int) -> Dict[str, List[str]]:
    """
    Generates a dictionary of gate keys.

    Args:
        num_gates: The number of gates.

    Returns:
        A dictionary of gate keys.
    """
    gate_keys: Dict[str, List[str]] = {
        "type": [],
        "number": [],
        "control": [],
        "target": [],
        "angle1": [],
        "angle2": [],
        "angle3": [],
    }
    for i in range(num_gates + 1):
        prefix: str = f"gate_{i:02}_"
        gate_keys["type"].append(prefix + "Gate_Type")
        gate_keys["number"].append(prefix + "Gate_Number")
        gate_keys["control"].append(prefix + "Control")
        gate_keys["target"].append(prefix + "Target")
        gate_keys["angle1"].append(prefix + "Angle_1")
        gate_keys["angle2"].append(prefix + "Angle_2")
        gate_keys["angle3"].append(prefix + "Angle_3")
    return gate_keys


def normalize_gate_number(gate_number: int) -> int:
    """
    Normalizes gate number: returns 0 if gate_number is -1, else returns 1.

    Args:
        gate_number: The gate number.

    Returns:
        0 if gate_number is -1, else 1.
    """
    return 0 if gate_number == -1 else 1


def normalize_shard(data: pd.DataFrame, num_qubits: int, num_gates: int) -> pd.DataFrame:
    """
    Normalizes a single CSV shard.

    Args:
        data: The input DataFrame.
        num_qubits: The number of qubits.
        num_gates: The number of gates.

    Returns:
        The normalized DataFrame.
    """
    statevector_keys: List[str] = generate_statevector_keys(num_qubits)
    gate_keys: Dict[str, List[str]] = generate_gate_keys(num_gates)

    # Normalize statevector features
    data[statevector_keys] /= 1024

    # Normalize gate type using a fixed mapping
    type_mapping: Dict[str, float] = {'U3Gate': 0, 'CnotGate': 1/3, 'Measure': 2/3, 'BLANK': 1}
    data[gate_keys["type"]] = data[gate_keys["type"]].replace(type_mapping)

    # Normalize control and target qubits (assumed range adjustment)
    control_target_keys: List[str] = gate_keys["control"] + gate_keys["target"]
    data[control_target_keys] = (data[control_target_keys] + 1) / 5

    # Normalize angles (assumed range adjustment)
    angle_keys: List[str] = gate_keys["angle1"] + gate_keys["angle2"] + gate_keys["angle3"]
    data[angle_keys] = (data[angle_keys] + 1) / 3

    # Normalize gate numbers using the helper function
    data[gate_keys["number"]] = data[gate_keys["number"]].applymap(normalize_gate_number)
    
    return data


def normalize_memory(data: pd.DataFrame, num_gates: int) -> pd.DataFrame:
    """
    Normalizes numeric features in a CSV file using MinMaxScaler.

    Args:
        data: The input DataFrame.
        num_gates: The number of gates.

    Returns:
        The normalized DataFrame.
    """
    gate_mapping = {'U3Gate': 1, 'CnotGate': 2, 'Measure': 3, 'BLANK': 4}
    data = data.replace(gate_mapping)
    
    # Generate feature list and apply MinMax scaling
    dense_features = generate_feature_names(num_gates)
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])
    
    return data


def generate_feature_names(num_gates: int) -> list:
    """
    Generate list of feature names for normalization.

    Args:
        num_gates: The number of gates.

    Returns:
        A list of feature names.
    """
    features = []
    for gate_num in range(num_gates):
        prefix = f"gate_{gate_num:02}_"
        features.extend([
            f"{prefix}Gate_Type",
            f"{prefix}Gate_Number",
            f"{prefix}Control",
            f"{prefix}Target",
            f"{prefix}Angle_1",
            f"{prefix}Angle_2",
            f"{prefix}Angle_3"
        ])
    return features


def process_csv(input_path: Path, output_path: Path, method: str, num_qubits: int = config.NUM_QUBITS, num_gates: int = 40) -> None:
    """
    Normalizes a CSV file using either shard-based or memory-based methods.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
        method: The normalization method to use ('shard' or 'memory').
        num_qubits: The number of qubits.
        num_gates: The number of gates.
    """
    data: pd.DataFrame = pd.read_csv(input_path)
    
    if method == "shard":
        data = normalize_shard(data, num_qubits, num_gates)
    elif method == "memory":
        data = normalize_memory(data, num_gates)
    else:
        raise ValueError("Invalid normalization method. Choose 'shard' or 'memory'.")
    
    data.to_csv(output_path, index=False)
    print(f"Successfully processed {len(data)} records using {method} method. Output saved to {output_path}")


def main() -> None:
    """Main function to normalize CSV files."""
    parser = argparse.ArgumentParser(description="Normalize CSV files")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CSV file path")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["shard", "memory"],
        default="shard",
        help="Normalization method ('shard' for large files, 'memory' for smaller files)",
    )
    parser.add_argument(
        "-nq",
        "--num_qubits",
        type=int,
        default=config.NUM_QUBITS,
        help="Number of qubits",
    )
    parser.add_argument(
        "-ng",
        "--num_gates",
        type=int,
        default=40,
        help="Number of gates",
    )
    args = parser.parse_args()

    process_csv(args.input, args.output, args.method, args.num_qubits, args.num_gates)


if __name__ == "__main__":
    main()