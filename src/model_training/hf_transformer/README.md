# Quantum Circuit Optimization via Google Gemini (In-Context Learning)

This directory replaces the previous Hugging Face Transformer inference with a call to the Google Gemini API using *in-context learning*. Instead of loading a local model checkpoint, `predict.py` now constructs a prompt with example circuit pairs and streams it to Gemini for optimization.

---

## 1. Overview

- **Before:** `predict.py` loaded a local `EncoderDecoderModel` and custom tokenizer, then ran beam search generation.
- **Now:** `predict.py` calls `optimize_circuit_with_gemini` from `gemini_optimizer.py`, which:
  1. Reads your Gemini API key.
  2. Builds a prompt with user-defined examples and the new circuit.
  3. Sends the prompt to the Gemini model.
  4. Parses and returns the optimized circuit string.

This approach leverages Google Gemini's large language capabilities without requiring a local fine-tuned model.

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

`optimize_circuit_with_gemini` looks for your API key in two places:

1. **Environment Variable:**
   - Set `GEMINI_API_KEY` in your shell:
     ```bash
     export GEMINI_API_KEY="YOUR_API_KEY"
     ```
2. **Key File:**
   - Place your key in a plain text file at `~/.api-gemini` (one line).
   - The code will read and strip whitespace.

If neither is found, the script raises an error.

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
python predict.py --input_circuit "<circuit representation>"
```

> **Note:** The arguments `--model_dir` and `--tokenizer_dir` remain in the CLI but are no longer used for inference. You may leave them at default or pass dummy paths.

Example:
```bash
python predict.py --input_circuit "H 0; CNOT 0 1; H 0"
```

The script prints the optimized circuit returned by Gemini.

---

## 6. Customization & Troubleshooting

- **Change Examples:** Modify the `EXAMPLES` list in `predict.py` to include domain-specific pairs.
- **Model Selection:** By default `gemini-pro` is used; to change, adjust the call in `gemini_optimizer.py`:
  ```python
  response = genai.generate_text(model="gemini-pro", prompt=prompt)
  ```
- **Error Handling:** API failures or parse errors raise `RuntimeError` with details.

Refer to `gemini_optimizer.py` for implementation details.
