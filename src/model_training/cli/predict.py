import argparse
from typing import List, Tuple, Optional
import os
import sys
import logging
from pathlib import Path
import requests

# Package-local imports (keep 'src.' when running via -m as per project guidance)
from src.model_training.gemini_optimizer import optimize_circuit_with_gemini
from src.model_training.config import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    DEFAULT_TIMEOUT,
    OPENROUTER_API_URL,
    resolve_openrouter_model,
    resolve_gemini_model,
    resolve_openrouter_api_key,
)
from src.model_training.utils import (
    validate_circuit_string,
    parse_examples,
    format_circuit_for_display,
    setup_logging,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration functions are now imported from config module


def _optimize_prompt(examples: List[Tuple[str, str]], unoptimized: str) -> str:
    lines = ["Optimize the following quantum circuits based on the provided examples:"]
    for inp, out in examples:
        lines.append(f"Unoptimized: {inp}")
        lines.append(f"Optimized: {out}")
        lines.append("")
    lines.append(f"Unoptimized: {unoptimized}")
    lines.append("Optimized:")
    return "\n".join(lines)


def call_openrouter_optimize(unoptimized_circuit: str, examples: List[Tuple[str, str]], model: Optional[str], timeout: int = 60) -> Optional[str]:
    """Call OpenRouter API to optimize quantum circuit."""
    api_key = resolve_openrouter_api_key()
    if not api_key:
        logger.error("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or create ~/.api-openrouter file.")
        return None

    resolved_model = resolve_openrouter_model(model)
    logger.info(f"Using OpenRouter model: {resolved_model}")

    payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": _optimize_prompt(examples, unoptimized_circuit)}],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logger.info(f"Calling OpenRouter API with timeout: {timeout}s")
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)

        if resp.status_code != 200:
            logger.error(f"OpenRouter API returned status {resp.status_code}: {resp.text}")
            return None

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            logger.error("OpenRouter API returned no choices in response")
            return None

        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            logger.error("OpenRouter API returned empty content")
            return None

        logger.info("Successfully received optimized circuit from OpenRouter")
        return content

    except requests.exceptions.Timeout:
        logger.error(f"OpenRouter API call timed out after {timeout} seconds")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling OpenRouter API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling OpenRouter API: {e}")
        return None


def main() -> None:
    """Main entry point for the quantum circuit optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Unified CLI to optimize quantum circuits via providers (default: openrouter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS,
                        help="Which provider to use")
    parser.add_argument("--model", type=str, default=None, help="Override model name for the chosen provider")
    parser.add_argument("--input_circuit", type=str, required=True,
                        help="Input quantum circuit as a string (e.g., 'H 0; CNOT 0 1; H 0')")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP/API timeout seconds (for openrouter)")
    parser.add_argument("--example", action="append", default=[],
                        help="Add example pair as 'input||output'. Can be used multiple times")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level based on verbosity
    setup_logging(verbose=args.verbose)

    # Validate input circuit
    if not validate_circuit_string(args.input_circuit):
        logger.error("Invalid circuit string format")
        sys.exit(1)

    logger.info(f"Starting quantum circuit optimization using provider: {args.provider}")
    logger.info(f"Input circuit: {format_circuit_for_display(args.input_circuit)}")

    # Parse examples
    try:
        examples = parse_examples(args.example)
        for i, (inp, out) in enumerate(examples):
            logger.debug(f"Added example {i+1}: '{inp}' -> '{out}'")
    except ValueError as e:
        logger.error(f"Failed to parse examples: {e}")
        sys.exit(1)

    if not examples:
        examples = [
            ("H 0; CNOT 0 1; H 0", "CNOT 0 1"),
            ("X 0; X 0; Y 1", "Y 1"),
        ]
        logger.info("Using default examples")

    logger.info(f"Using {len(examples)} examples for in-context learning")

    if args.provider == "openrouter":
        logger.info("Calling OpenRouter API...")
        optimized = call_openrouter_optimize(args.input_circuit, examples, model=args.model, timeout=args.timeout)
        if optimized is None:
            logger.error("OpenRouter optimization failed")
            sys.exit(1)
        logger.info("Optimization completed successfully")
        print(optimized)
        return

    # Gemini path
    logger.info("Calling Gemini API...")
    resolved_model = _resolve_gemini_model(args.model)
    try:
        optimized = optimize_circuit_with_gemini(args.input_circuit, examples, model=resolved_model)
        logger.info("Optimization completed successfully")
        print(optimized)
    except Exception as e:
        logger.error(f"Gemini optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()