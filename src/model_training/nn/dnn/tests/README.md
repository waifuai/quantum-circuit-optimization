# README for `test_suite.py`

This file contains a comprehensive suite of unit tests for the Deep Neural Network (DNN) training and optimization scripts used in this project. It utilizes the `pytest` framework for testing various aspects of the code, including data loading and preprocessing, model building, and interactions between the DNN and Cirq components in the hybrid optimizer scripts.

## Overview of Tests

The tests are organized into sections based on the specific scripts or functionalities they target:

**1. Tests for `legacy_dnn_regression.py`:**

   - `test_create_save_paths`: Verifies that the `create_save_paths` function generates correct log and model directory paths.
   - `test_load_and_preprocess_data`: Checks the data loading and preprocessing steps for the 1s regression script, ensuring the correct shape of input features (X) and target (y).
   - `test_build_dnn_model`: Validates that the `build_dnn_model` function creates a compiled Keras model with the expected input and output dimensions.

**2. Tests for `legacy_dnn_regression_32s.py`:**

   - `test_load_and_preprocess_data_32s`: Tests the data loading and preprocessing for the 32s regression script, ensuring that all dense features are present in the loaded data.
   - `test_build_dnn_model_32s`: Confirms that the `build_dnn_model` function for the 32s version constructs a Keras model with the correct input and output shapes (32 output units).

**3. Tests for `legacy_dnn_regression_32s_resume.py`:**

   - `test_preprocess_data_resume`: Checks the `preprocess_data` function, specifically verifying that the input and output shapes are correct and that the target column names are as expected.
   - `test_load_or_build_model`: Tests the `load_or_build_model` function, ensuring it can build a new model when no pretrained model exists and that the model has the expected output shape.

**4. Tests for Hybrid Optimizer Scripts (1s and 32s):**

   - `test_hybrid_optimizer_csv_not_found_1s`: Simulates a scenario where the input CSV file is missing and checks that the 1s hybrid optimizer script exits gracefully with an appropriate error message.
   - `test_hybrid_optimizer_circuit_1s`: Uses monkeypatching to replace the `create_circuit` function with a dummy implementation and verifies that the 1s script correctly uses this dummy function to generate the circuit.
   - `test_hybrid_optimizer_circuit_32s`: Similar to `test_hybrid_optimizer_circuit_1s`, but for the 32s hybrid optimizer script.

## Helper Functions

The test suite includes several helper functions:

- `create_dense_feature_names`: Generates a list of dense feature names according to the specific format used in the project.
- `create_dummy_csv`: Creates a dummy CSV file with random data that conforms to the expected format for either the 1s or 32s versions. This function can include target columns (for 1s) or a full set of statevector columns (for 32s).
- `import_module_from_path`: A utility to import a Python module given its file path. This is useful when the module under test is not directly in the Python path.

## Fixtures

- `dummy_csv_1s`: A pytest fixture that creates a temporary dummy CSV file suitable for the 1s optimizer tests.
- `dummy_csv_32s`: A pytest fixture that generates a temporary dummy CSV file designed for the 32s optimizer tests.

## Running the Tests

To run these tests, you will need:

- Python 3
- `pytest`
- `tensorflow`
- `numpy`
- `pandas`

1. **Install dependencies:**
    ```bash
    pip install pytest tensorflow numpy pandas
    ```

2. **Navigate to the directory** containing `test_suite.py`.

3. **Execute pytest:**
    ```bash
    pytest test_suite.py
    ```

## Notes

-   The tests for the hybrid optimizer scripts assume that the scripts output some information about the circuit generation to standard output. The `capsys` fixture in `pytest` is used to capture this output and verify its contents.
-   The tests that involve monkeypatching (replacing functions with dummy implementations) are designed to isolate specific parts of the code and test their behavior without needing to run the entire pipeline.
-   Adjust the relative paths in the `import_module_from_path` calls if your project's directory structure is different from what is assumed in these tests.

This comprehensive test suite ensures the robustness of the DNN training and optimization code, covering a wide range of scenarios and edge cases. By systematically testing individual components and their interactions, these tests help maintain code quality and prevent regressions as the project evolves.
