"""
Quantum circuit optimization command-line interface.

This module provides a unified command-line interface for optimizing quantum circuits
using the OpenRouter AI provider. It supports in-context learning with custom examples,
multiple model configurations, and comprehensive logging for debugging and monitoring
optimization performance.
"""

import argparse
from typing import List, Tuple, Optional
import os
import sys
import logging
from pathlib import Path
import requests

# Package-local imports (keep 'src.' when running via -m as per project guidance)
from src.model_training.openrouter_optimizer import optimize_circuit_with_openrouter
from src.model_training.config import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    DEFAULT_TIMEOUT,
    resolve_openrouter_model,
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


def main() -> None:
    """Main entry point for the quantum circuit optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Unified CLI to optimize quantum circuits via OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS,
                        help="Which provider to use")
    parser.add_argument("--model", type=str, default=None, help="Override model name for the chosen provider")
    parser.add_argument("--input_circuit", type=str, required=True,
                        help="Input quantum circuit as a string (e.g., 'H 0; CNOT 0 1; H 0')")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP/API timeout seconds")
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

    # Optimize via OpenRouter
    try:
        optimized = optimize_circuit_with_openrouter(
            args.input_circuit, examples, model=args.model, timeout=args.timeout
        )
        logger.info("Optimization completed successfully")
        print(optimized)
    except (RuntimeError, ValueError) as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
