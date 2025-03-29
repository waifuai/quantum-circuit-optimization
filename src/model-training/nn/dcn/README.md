# Quantum Circuit Optimization with Cirq

This directory contains code for optimizing quantum circuits using Cirq. It focuses on adjusting the parameters of a quantum circuit to minimize a loss function based on a target state.

## Repository Structure

```
src/model-training/nn/dcn/
├── cirq_circuit_optimizer.py  # Quantum circuit optimizer using Cirq
└── test_suite.py              # Unit tests (may need updating)
```

## `cirq_circuit_optimizer.py`

### Description

This script optimizes the parameters of a quantum circuit defined in Cirq. The goal is to find the parameters that minimize the difference between the circuit's output statevector and a target statevector. The optimization is performed using SciPy's minimize function (BFGS algorithm by default).

### Key Features

*   **Circuit Creation**: Creates a quantum circuit using Cirq based on input parameters (using `utils.circuit_utils.create_circuit`).
*   **Fidelity Calculation**: Calculates the fidelity between the circuit's output statevector and a target state.
*   **Loss Calculation**: Defines a loss function as `1 - fidelity`.
*   **Optimization**: Uses `scipy.optimize.minimize` to find parameters that minimize the average loss over a dataset.

### Usage

1.  **Prepare your data**: Ensure you have a CSV file (e.g., `./qc5f_1k_2.csv` as used in the script) containing features and target statevector data. Update the data loading logic and feature/target definitions as needed.
2.  **Define your target state**: The script currently uses the `statevector_00000` column from the CSV as the target. Modify this if needed.
3.  **Run the script**: Execute the script using `python cirq_circuit_optimizer.py`.

### Dependencies

*   Cirq
*   NumPy
*   pandas
*   scikit-learn
*   SciPy

## `test_suite.py`

### Description

This script contains unit tests. **Note:** The tests related to the removed `legacy_cpu_dcn.py` need to be removed or updated. The tests for `cirq_circuit_optimizer.py` should still be relevant.

### Tests for `cirq_circuit_optimizer.py`

*   **`test_optimize_circuit_returns_array`**: Verifies that the `optimize_circuit` function returns a NumPy array of parameters with the expected shape.
*   **`test_loss_function_calls`**: Checks that the `calculate_fidelity` and `create_circuit` functions are called within the optimization loop as expected.

### Usage

Run the test suite using:

```bash
python test_suite.py
```
(After updating the tests)

## Notes

*   The code assumes a specific format for the input CSV file. Adapt the data loading and preparation steps based on your data.
*   The quantum circuit structure is defined in `src/model-training/nn/utils/circuit_utils.py`. Modify this if needed.
*   The optimizer uses SciPy's BFGS algorithm. Other methods supported by `scipy.optimize.minimize` could be explored.
*   Ensure that all dependencies are installed before running the scripts.

## Contact

For questions or feedback, please open an issue on the GitHub repository.
