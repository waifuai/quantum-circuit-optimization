import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemini_optimizer import optimize_circuit_with_gemini

# In-context examples for Gemini API (replace with domain-specific examples)
EXAMPLES = [
    ("H 0; CNOT 0 1; H 0", "CNOT 0 1"),
    ("X 0; X 0; Y 1", "Y 1"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize a quantum circuit using Google Gemini API (gemini-2.5-pro-preview-03-25) via in-context learning."
    )
    parser.add_argument(
        "--input_circuit", type=str, required=True,
        help="Input quantum circuit as a string (e.g., 'H 0; CNOT 0 1; H 0')."
    )
    args = parser.parse_args()

    try:
        optimized_circuit = optimize_circuit_with_gemini(args.input_circuit, EXAMPLES)
        print("\n--- Prediction Result ---")
        print(f"Input:  '{args.input_circuit}'")
        print(f"Output: '{optimized_circuit}'")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)