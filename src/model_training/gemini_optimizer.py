import os
import google.generativeai as genai

def optimize_circuit_with_gemini(unoptimized_circuit_string: str, examples: list[tuple[str, str]]) -> str:
    """
    Optimize the given quantum circuit string using Google Gemini API with in-context learning.

    Args:
        unoptimized_circuit_string: The circuit to optimize.
        examples: List of (unoptimized_example, optimized_example) tuples for in-context learning.

    Returns:
        The optimized circuit string returned by Gemini.
    """
    # Load API key from environment variable or ~/.api-gemini file
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            with open(os.path.expanduser("~/.api-gemini"), "r") as f:
                api_key = f.read().strip()
        except Exception as e:
            raise RuntimeError("GEMINI_API_KEY not set and unable to read from ~/.api-gemini") from e
    # Configure the Gemini client
    genai.configure(api_key=api_key)

    # Build the prompt
    prompt_lines = ["Optimize the following quantum circuits based on the provided examples:"]
    for inp, out in examples:
        prompt_lines.append(f"Unoptimized: {inp}")
        prompt_lines.append(f"Optimized: {out}")
        prompt_lines.append("")  # blank line between examples
    prompt_lines.append(f"Unoptimized: {unoptimized_circuit_string}")
    prompt_lines.append("Optimized:")
    prompt = "\n".join(prompt_lines)

    # Call the Gemini API
    try:
        response = genai.generate_text(model="gemini-pro", prompt=prompt)
        text = response.text
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}") from e

    # Parse the optimized circuit from the response
    try:
        optimized = text.split("Optimized:")[-1].strip()
    except Exception as e:
        raise RuntimeError(f"Error parsing Gemini response: {e}") from e

    return optimized
