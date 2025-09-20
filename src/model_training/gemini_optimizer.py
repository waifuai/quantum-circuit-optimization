import os
import logging
from typing import List, Tuple, Optional
from google import genai
from src.model_training.config import resolve_gemini_model, resolve_gemini_api_key

# Configure logger
logger = logging.getLogger(__name__)

# Configuration functions are now imported from config module

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

    Raises:
        RuntimeError: If the API call fails or authentication is not properly configured.
        ValueError: If the input parameters are invalid.
    """
    if not unoptimized_circuit_string or not unoptimized_circuit_string.strip():
        raise ValueError("Input circuit string cannot be empty")
    if not examples:
        raise ValueError("Examples list cannot be empty")

    resolved_model = model or resolve_gemini_model()
    logger.info(f"Using Gemini model: {resolved_model}")
    logger.debug(f"Optimizing circuit: {unoptimized_circuit_string}")
    logger.debug(f"Using {len(examples)} examples for in-context learning")

    # Get API key using centralized configuration
    api_key = resolve_gemini_api_key()
    client = None

    try:
        if api_key:
            logger.debug("Using API key from configuration")
            client = genai.Client(api_key=api_key)
        else:
            logger.debug("No API key found, trying default client (may use environment)")
            client = genai.Client()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GenAI client: {e}") from e

    # Build prompt with examples
    prompt_lines = ["Optimize the following quantum circuits based on the provided examples:"]
    for inp, out in examples:
        prompt_lines.append(f"Unoptimized: {inp}")
        prompt_lines.append(f"Optimized: {out}")
        prompt_lines.append("")
    prompt_lines.append(f"Unoptimized: {unoptimized_circuit_string}")
    prompt_lines.append("Optimized:")
    prompt = "\n".join(prompt_lines)

    logger.debug(f"Generated prompt with {len(prompt_lines)} lines")

    try:
        logger.info("Calling Gemini API...")
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt
        )

        if not response.text or not response.text.strip():
            raise RuntimeError("Gemini API returned empty response")

        logger.info("Successfully received optimized circuit from Gemini")
        return response.text.strip()

    except AttributeError as e:
        raise RuntimeError(f"Invalid response structure from Gemini API: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}") from e
