# Quantum Circuit Dataset Generator

This package demonstrates generating random quantum circuits using Cirq, simulating them (with noise), optimizing them using basic Cirq routines, and saving the data.

## Directory Structure

- **qc/** – Contains modules:
  - `circuit_generation.py` (circuit creation and conversion to dictionary)
  - `simulation.py` (simulation functions)
  - `optimization.py` (optimization routines)
- **scripts/** – Contains main scripts:
  - `generate_dataset.py` – Generates a dataset of circuits and saves it as a TFRecord file.

## Usage

From the project root, run:

```bash
python src/data-generation/scripts/generate_dataset.py
```

This will generate 100 circuits (adjustable via script arguments) and save them to `cirq_dataset.tfrecord`.

## Dependencies

- Python 3.10+
- Cirq
- TensorFlow (only for TFRecord output)
- tqdm

Install dependencies via:

```bash
.venv/Scripts/python.exe -m uv pip install -r requirements.txt
