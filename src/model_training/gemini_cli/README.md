# Gemini CLI for Quantum Circuit Optimization

**Note:** The `gemini_cli` directory name is historical. This module now provides a command-line interface (CLI) for optimizing quantum circuits using the Google Gemini API (model: gemini-2.5-flash-preview-04-17) via in-context learning.

---

## Overview

- `predict.py` uses `optimize_circuit_with_gemini` from `gemini_optimizer.py` to:
  1. Read your Gemini API key from `~/.api-gemini`.
  2. Build a prompt with example circuit pairs and your input circuit.
  3. Send the prompt to Gemini.
  4. Print the optimized circuit returned by Gemini.

No local models or tokenizers are used.

---

## 2. Dependencies

Ensure you have installed all required libraries, including the Gemini SDK:

```powershell
# Use uv inside your virtual environment
python -m uv pip install -r requirements.txt
```

The `requirements.txt` should include:
- cirq-core
- transformers
- google-generativeai
- (other existing dependencies…)

---

## 3. Setting Up Your Gemini API Key

**Update:** The current implementation in `gemini_optimizer.py` *only* checks for the key file. The environment variable method is not used.
- **Key File:**
   - Place your key in a plain text file at `~/.api-gemini` (one line).
   - The code will read this file and strip whitespace. **This is the only method currently implemented.**

If the key file is not found, the script raises an error.

---

## 4. In-Context Examples

At the top of `predict.py`, you can define any number of example pairs:

```python
EXAMPLES = [
    ("H 0; CNOT 0 1; H 0", "CNOT 0 1"),
    ("X 0; X 0; Y 1", "Y 1"),
    # Add more examples here...
]
```

These examples are concatenated into the prompt:

```
Optimize the following quantum circuits based on the provided examples:

Unoptimized: example1_input
Optimized:   example1_output

... (other examples) ...

Unoptimized: <your new circuit>
Optimized:
```

Gemini completes after the final `Optimized:` tag.

---

## 5. Usage

```powershell
python src/model_training/gemini_cli/predict.py --input_circuit "<circuit representation>"
```

> Example:
```bash
python src/model_training/gemini_cli/predict.py --input_circuit "H 0 ; CNOT 0 1 ; H 0"
```

The script prints the optimized circuit returned by Gemini.

---

## 6. Customization & Troubleshooting

- **Change Examples:** Modify the `EXAMPLES` list in `predict.py` to include domain-specific pairs.
- **Model Selection:** The model (`gemini-2.5-flash-preview-04-17`) is hardcoded in `gemini_optimizer.py`. To change it, modify the `GenerativeModel` instantiation:
  ```python
  model = genai.GenerativeModel("your-desired-gemini-model")
  ```
- **Error Handling:** API failures or parse errors raise `RuntimeError` with details.

Refer to `gemini_optimizer.py` for implementation details.
