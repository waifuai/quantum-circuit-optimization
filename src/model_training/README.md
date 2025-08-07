# model-training: LLM-based Quantum Circuit Optimization

This module contains code for optimizing quantum circuits using:
- OpenRouter (default) with model `openrouter/horizon-beta`
- Google GenAI (Gemini) with model `gemini-2.5-pro`

## Contents

- `cli/predict.py`: Unified CLI for provider-agnostic optimization (default provider: openrouter)
- `gemini_cli/`: Deprecated historical CLI for Gemini (kept for compatibility)
- `gemini_optimizer.py`: Core Google GenAI SDK interaction function

## Setting Up

1. **Install dependencies** (from the repo root):
   ```bash
   .venv/Scripts/python.exe -m uv venv .venv ; .venv/Scripts/python.exe -m ensurepip ; .venv/Scripts/python.exe -m pip install uv ; .venv/Scripts/python.exe -m uv pip install -r requirements.txt
   ```

2. **Credentials:**
   - OpenRouter: set `OPENROUTER_API_KEY` or create `~/.api-openrouter` with the API key as a single line.
   - Gemini: set `GEMINI_API_KEY` or `GOOGLE_API_KEY` or create `~/.api-gemini` with the API key as a single line.

## Usage (Unified CLI)

To optimize a quantum circuit (default provider: OpenRouter):

```bash
python -m src.model_training.cli.predict --input_circuit "H 0 ; CNOT 0 1 ; H 0"
```

Use Gemini explicitly:

```bash
python -m src.model_training.cli.predict --provider gemini --input_circuit "H 0 ; CNOT 0 1 ; H 0"
```

### Model selection via dotfiles

- OpenRouter: `~/.model-openrouter` (first non-empty line), fallback `openrouter/horizon-beta`
- Gemini: `~/.model-gemini` (first non-empty line), fallback `gemini-2.5-pro`

## Future Work

- Advanced prompt engineering for both providers.
- Automated evaluation and benchmarking across providers.
- Optional retries/backoff and rate-limit handling for OpenRouter.
- Larger in-context example management.