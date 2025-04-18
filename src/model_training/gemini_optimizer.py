import os
import google.generativeai as genai

def optimize_circuit_with_gemini(unoptimized_circuit_string: str, examples: list[tuple[str, str]]) -> str:
    """
    Optimize the given quantum circuit string using Google Gemini API (model: gemini-2.5-flash-preview-04-17) with in-context learning.

    Args:
        unoptimized_circuit_string: The circuit to optimize.
        examples: List of (unoptimized_example, optimized_example) tuples for in-context learning.

    Returns:
        The optimized circuit string returned by Gemini.

    Notes:
        - The Gemini API key is loaded exclusively from the file ~/.api-gemini. If the file is missing or unreadable, an error is raised.
        - Uses the model 'gemini-2.5-flash-preview-04-17'.
    """
    try:
        with open(os.path.expanduser("~/.api-gemini"), "r") as f:
            api_key = f.read().strip()
    except Exception as e:
        raise RuntimeError("Unable to read Gemini API key from ~/.api-gemini") from e
    genai.configure(api_key=api_key)

    prompt_lines = ["Optimize the following quantum circuits based on the provided examples:"]
    for inp, out in examples:
        prompt_lines.append(f"Unoptimized: {inp}")
        prompt_lines.append(f"Optimized: {out}")
        prompt_lines.append("")
    prompt_lines.append(f"Unoptimized: {unoptimized_circuit_string}")
    prompt_lines.append("Optimized:")
    prompt = "\n".join(prompt_lines)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        response = model.generate_content(prompt)
        text = response.text
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}") from e
    return text
