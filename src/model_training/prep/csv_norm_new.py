"""
CSV preprocessing

Normalizes features in CSV shards using universal normalization constants.
Processes a directory of CSV shards, normalizing each shard based on predefined 
minimum and maximum values. This ensures consistency across datasets and allows 
for memory-efficient processing.

Normalization is performed on statevectors, gate types, control/target qubits, 
angles, and gate numbers. Shards are expected to have a header in the first line.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

N_QUBITS: int = 5
N_GATES: int = 40


def generate_statevector_keys(n_qubits: int) -> List[str]:
    """Generates a list of statevector keys."""
    return [f"statevector_{bin(i)[2:].zfill(n_qubits)}" for i in range(2**n_qubits)]


def generate_gate_keys(num_gates: int) -> Dict[str, List[str]]:
    """Generates a dictionary of gate keys."""
    keys: Dict[str, List[str]] = {
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
        keys["type"].append(prefix + "Gate_Type")
        keys["number"].append(prefix + "Gate_Number")
        keys["control"].append(prefix + "Control")
        keys["target"].append(prefix + "Target")
        keys["angle1"].append(prefix + "Angle_1")
        keys["angle2"].append(prefix + "Angle_2")
        keys["angle3"].append(prefix + "Angle_3")
    return keys


STATEVECTOR_KEYS: List[str] = generate_statevector_keys(N_QUBITS)
GATE_KEYS: Dict[str, List[str]] = generate_gate_keys(N_GATES)


def normalize_gate_number(n: int) -> int:
    """Normalizes gate number: returns 0 if n is -1, else returns 1."""
    return 0 if n == -1 else 1


def normalize_shard(input_filepath: Path) -> None:
    """Reads, normalizes, and writes a single CSV shard."""
    data: pd.DataFrame = pd.read_csv(input_filepath)

    # Normalize statevector features
    data[STATEVECTOR_KEYS] /= 1024

    # Normalize gate type using a fixed mapping
    type_mapping: Dict[str, float] = {'U3Gate': 0, 'CnotGate': 1/3, 'Measure': 2/3, 'BLANK': 1}
    data[GATE_KEYS["type"]] = data[GATE_KEYS["type"]].replace(type_mapping)

    # Normalize control and target qubits (assumed range adjustment)
    control_target_keys: List[str] = GATE_KEYS["control"] + GATE_KEYS["target"]
    data[control_target_keys] = (data[control_target_keys] + 1) / 5

    # Normalize angles (assumed range adjustment)
    angle_keys: List[str] = GATE_KEYS["angle1"] + GATE_KEYS["angle2"] + GATE_KEYS["angle3"]
    data[angle_keys] = (data[angle_keys] + 1) / 3

    # Normalize gate numbers using the helper function
    data[GATE_KEYS["number"]] = data[GATE_KEYS["number"]].applymap(normalize_gate_number)

    # Save output with _output suffix
    output_filepath: Path = input_filepath.with_name(input_filepath.stem + "_output.csv")
    data.to_csv(output_filepath, index=False)


def process_shards(directory: Path) -> None:
    """Processes all CSV shards in the given directory."""
    csv_files: List[Path] = sorted(directory.glob("*.csv"))
    for csv_file in tqdm(csv_files, desc="Processing shards"):
        normalize_shard(csv_file)


def main() -> None:
    """Main function to process CSV shards."""
    import argparse
    parser = argparse.ArgumentParser(description="Normalize CSV shards in a directory")
    parser.add_argument("-d", "--directory", type=Path, default=Path("./qc9/shards/"),
                        help="Directory containing CSV shards")
    args = parser.parse_args()

    process_shards(args.directory)


if __name__ == "__main__":
    main()
