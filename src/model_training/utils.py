"""
Utility functions for quantum circuit optimization.

This module provides common functionality used across different
components of the quantum circuit optimization system.
"""
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def validate_circuit_string(circuit_str: str) -> bool:
    """
    Validate that a circuit string is in the expected format.

    Args:
        circuit_str: The circuit string to validate.

    Returns:
        True if the circuit string appears valid, False otherwise.
    """
    if not circuit_str or not circuit_str.strip():
        return False

    # Basic validation - check for common gate patterns
    circuit_str = circuit_str.strip().upper()

    # Check for common quantum gates
    valid_gates = {'H', 'X', 'Y', 'Z', 'CNOT', 'CX', 'CZ', 'S', 'T', 'RX', 'RY', 'RZ'}
    tokens = circuit_str.replace(';', ' ').split()

    for token in tokens:
        if not any(gate in token for gate in valid_gates):
            continue  # Skip non-gate tokens (like qubit indices)
        # Check if token starts with a valid gate
        if not any(token.startswith(gate) for gate in valid_gates):
            logger.warning(f"Potentially invalid gate token: {token}")
            return False

    return True


def parse_examples(examples_str: List[str]) -> List[Tuple[str, str]]:
    """
    Parse example strings into (input, output) tuples.

    Args:
        examples_str: List of example strings in "input||output" format.

    Returns:
        List of (input, output) tuples.

    Raises:
        ValueError: If example format is invalid.
    """
    examples = []
    for i, ex in enumerate(examples_str):
        try:
            inp, out = ex.split("||", 1)
            examples.append((inp.strip(), out.strip()))
        except ValueError:
            raise ValueError(f"Invalid example format at index {i}: {ex}. "
                           "Expected format: 'input||output'")

    return examples


def format_circuit_for_display(circuit_str: str, max_length: int = 50) -> str:
    """
    Format a circuit string for display, truncating if necessary.

    Args:
        circuit_str: The circuit string to format.
        max_length: Maximum length before truncation.

    Returns:
        Formatted circuit string.
    """
    circuit_str = circuit_str.strip()
    if len(circuit_str) <= max_length:
        return circuit_str

    return circuit_str[:max_length-3] + "..."


def setup_logging(verbose: bool = False, level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging.
        level: Logging level as string.
    """
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )