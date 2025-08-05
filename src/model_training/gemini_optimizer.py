import os
from typing import List, Tuple
from google import genai

GEMINI_MODEL_NAME = "gemini-2.5-pro"

def _load_api_key_from_file() -> str:
    key_path = os.path.expanduser("~/.api-gemini")
    with open(key_path, "r") as f:
        return f.read().strip()

def optimize_circuit_with_gemini(unoptimized_circuit_string: str, examples: List[Tuple[str, str]]) -> str:
    """
    Optimize the given quantum circuit string using Google GenAI SDK (model: gemini-2.5-pro) with in-context learning.

    Auth:
      - If GEMINI_API_KEY/GOOGLE_API_KEY env var is set, the client will use it automatically.
      - Otherwise, if ~/.api-gemini exists, it will be read and passed explicitly to the client.

    Args:
        unoptimized_circuit_string: The circuit to optimize.
        examples: List of (unoptimized_example, optimized_example) tuples for in-context learning.

    Returns:
        The optimized circuit string returned by Gemini.
    """
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
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}") from e
