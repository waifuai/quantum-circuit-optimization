# model-training: Gemini API Quantum Circuit Optimization

This module contains all code for optimizing quantum circuits using the Google Gemini API (model: gemini-2.5-flash-preview-04-17) via in-context learning. All classical neural network and local model code has been removed.

## Contents

- `gemini_optimizer.py`: Core function to call the Gemini API for circuit optimization.
- `hf_transformer/predict.py`: Command-line interface for optimizing circuits with Gemini.

## Setting Up

1. **Install dependencies** (from the repo root):
   ```bash
   python -m uv pip install -r requirements.txt
   ```

2. **Set up your Gemini API key:**
   - Place your API key in a file at `~/.api-gemini` (one line, no extra whitespace).

## Usage

To optimize a quantum circuit using Gemini:

```bash
python src/model_training/hf_transformer/predict.py --input_circuit "H 0; CNOT 0 1; H 0"
```

The script will print the input and the optimized output returned by Gemini.

## Future Work

- Advanced prompt engineering for Gemini
- Automated evaluation and benchmarking of Gemini-optimized circuits
- Reinforcement learning and feedback via the Gemini API
- Large-scale dataset generation for Gemini prompt tuning