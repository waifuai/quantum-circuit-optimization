import os
from typing import List, Tuple, Optional
from google import genai
from pathlib import Path

# Default fallback if no ~/.model-gemini present
_DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
_MODEL_FILE = Path.home() / ".model-gemini"

def _resolve_gemini_model() -> str:
    """
    Resolve model name from ~/.model-gemini (first non-empty line), else fallback.
    """
    try:
        if _MODEL_FILE.is_file():
            for line in _MODEL_FILE.read_text(encoding="utf-8").splitlines():
                val = line.strip()
                if val:
                    return val
    except Exception:
        pass
    return _DEFAULT_GEMINI_MODEL

def _load_api_key_from_file() -> str:
    key_path = os.path.expanduser("~/.api-gemini")
    with open(key_path, "r") as f:
        return f.read().strip()

def optimize_circuit_with_gemini(unoptimized_circuit_string: str, examples: List[Tuple[str, str]], model: Optional[str] = None) -> str:
    """
    Optimize the given quantum circuit string using Google GenAI SDK with in-context learning.

    Auth:
      - If GEMINI_API_KEY/GOOGLE_API_KEY env var is set, the client will use it automatically.
      - Otherwise, if ~/.api-gemini exists, it will be read and passed explicitly to the client.

    Args:
        unoptimized_circuit_string: The circuit to optimize.
        examples: List of (unoptimized_example, optimized_example) tuples for in-context learning.
        model: Optional explicit Gemini model name. If None, resolved from ~/.model-gemini or fallback.

    Returns:
        The optimized circuit string returned by Gemini.
    """
    resolved_model = model or _resolve_gemini_model()

    # Prefer environment variable; fallback to key file if present.
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = None
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            # Fall back to ~/.api-gemini if present
            try:
                api_key = _load_api_key_from_file()
                client = genai.Client(api_key=api_key)
            except Exception:
                # As a last resort, try default which may pick up env in some environments.
                client = genai.Client()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GenAI client: {e}") from e

    prompt_lines = ["Optimize the following quantum circuits based on the provided examples:"]
    for inp, out in examples:
        prompt_lines.append(f"Unoptimized: {inp}")
        prompt_lines.append(f"Optimized: {out}")
        prompt_lines.append("")
    prompt_lines.append(f"Unoptimized: {unoptimized_circuit_string}")
    prompt_lines.append("Optimized:")
    prompt = "\n".join(prompt_lines)

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}") from e
