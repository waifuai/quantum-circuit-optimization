# Quantum Circuit Dataset Generator

This package demonstrates generating random quantum circuits using Cirq, simulating them (with noise), and optimizing them.

## Directory Structure

- **qc/** – Contains modules:
  - `circuit_generation.py` (circuit creation and conversion to dictionary)
  - `simulation.py` (simulation functions)
  - `optimization.py` (optimization routines)
- **scripts/** – Contains main scripts:
  - `generate_dataset.py` – Generates a dataset of circuits and saves it as a pickle file.

## Usage

From the project root, run:

```bash
python scripts/generate_dataset.py
```

This will generate 100 circuits (adjustable via the script) and save them to a pickle file.

## Dependencies

- Python 3.10+
- Cirq
- tqdm

Install dependencies via:

```bash
pip install cirq tqdm
```
