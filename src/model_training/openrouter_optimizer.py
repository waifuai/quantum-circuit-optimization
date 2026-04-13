"""
OpenRouter quantum circuit optimization module.

This module provides integration with OpenRouter AI for quantum circuit optimization
using in-context learning. It handles API authentication, prompt construction with
examples, and response processing to generate optimized quantum circuit representations.
Migrated from Google Gemini to OpenRouter.
"""

import os
import logging
import requests
from typing import List, Tuple, Optional
from pathlib import Path
from src.model_training.config import resolve_openrouter_model, OPENROUTER_API_URL, DEFAULT_TIMEOUT

# Configure logger
logger = logging.getLogger(__name__)


def resolve_openrouter_api_key() -> Optional[str]:
    """Resolve OpenRouter API key from file or environment."""
    # Check environment variable first
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    # Fall back to file
    api_file = Path.home() / ".api-openrouter"
    try:
        if api_file.is_file():
            key = api_file.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    return None


def optimize_circuit_with_openrouter(unoptimized_circuit_string: str, examples: List[Tuple[str, str]], model: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Optimize the given quantum circuit string using OpenRouter API with in-context learning.

    Auth:
      - If OPENROUTER_API_KEY env var is set, it will be used.
      - Otherwise, if ~/.api-openrouter exists, it will be read.

    Args:
        unoptimized_circuit_string: The circuit to optimize.
        examples: List of (unoptimized_example, optimized_example) tuples for in-context learning.
        model: Optional explicit model name. If None, resolved from ~/.model-openrouter or fallback.
        timeout: HTTP timeout in seconds.

    Returns:
        The optimized circuit string returned by the model.

    Raises:
        RuntimeError: If the API call fails.
        ValueError: If the input parameters are invalid or API key is missing.
    """
    if not unoptimized_circuit_string or not unoptimized_circuit_string.strip():
        raise ValueError("Input circuit string cannot be empty")
    if not examples:
        raise ValueError("Examples list cannot be empty")

    api_key = resolve_openrouter_api_key()
    if not api_key:
        raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY env var or create ~/.api-openrouter file.")

    resolved_model = resolve_openrouter_model(model)
    logger.info(f"Using OpenRouter model: {resolved_model}")
    logger.debug(f"Optimizing circuit: {unoptimized_circuit_string}")
    logger.debug(f"Using {len(examples)} examples for in-context learning")

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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    try:
        logger.info("Calling OpenRouter API...")
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter API returned no choices in response")

        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("OpenRouter API returned empty response")

        logger.info("Successfully received optimized circuit from OpenRouter")
        return content

    except requests.exceptions.Timeout:
        raise RuntimeError(f"OpenRouter API call timed out after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter API: {e}")
    except Exception as e:
        raise RuntimeError(f"Error calling OpenRouter API: {e}")
