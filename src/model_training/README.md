# model training: Quantum Circuit Optimization with Deep Learning

This repository explores the use of deep learning techniques, specifically neural networks, to optimize quantum circuits. It leverages TensorFlow for classical machine learning, Cirq for quantum circuit manipulation, and Trax for advanced sequence-to-sequence modeling with Transformers. The project is divided into several modules, each focusing on a different aspect of the problem, including data preprocessing, model training, and prediction.

## Project Structure

This directory now focuses on inference via Google Gemini:

- `gemini_optimizer.py`: Function `optimize_circuit_with_gemini` for calling the Gemini API.
- `hf_transformer/`: Example CLI (`predict.py`) and documentation showing how to use Gemini for circuit optimization.

## Modules

- **Gemini Inference**:
  - `gemini_optimizer.py` (see above)
  - `hf_transformer/` (Gemini-based prediction example)

## Getting Started

### Installation

To use this project, clone the repository and install the required libraries:

```bash
# Navigate to the specific module you want to work with, e.g.,
cd src/model-training/
pip install --user cirq numpy pandas scikit-learn tensorflow
```

### Data

- **Gemini Inference**: Ensure that you have access to the Gemini API and have set up the necessary credentials.

### Running the Scripts

Each module contains scripts that can be run independently. Refer to the README files within each module for specific instructions.

#### Examples:

- **Optimize Circuit with Gemini**:
  ```bash
  python src/model-training/gemini_optimizer.py
  ```
- **Run Gemini-based Prediction**:
  ```bash
  python src/model-training/hf_transformer/predict.py
  ```

## Notes

- The project assumes a 5-qubit quantum circuit structure in many places. Adjust the code if your data has a different structure or if you want to use a different circuit.
- Training Transformer models can be computationally intensive. Consider using a GPU for faster training.
- Performance depends on the quality and quantity of training data.
- Refer to the README files within each module for more detailed information.
- Be sure to update the dataset paths in scripts to match your local file structure.

## Future Work

- Develop a custom loss function for integrating quantum circuit output into DNN training.
- Explore techniques for backpropagation through quantum circuits.
- Experiment with different DNN and Transformer architectures and hyperparameters.
- Investigate more advanced optimization algorithms.
- Apply these techniques to larger and more complex quantum circuits.
- Explore the use of quantum hardware for circuit simulation and optimization.
- Develop more comprehensive testing and validation procedures.
- Improve the Trax Transformer implementation by exploring different decoding methods (e.g. beam search in `trax-train`) and potentially using a more robust data representation.
- Incorporate error mitigation techniques into the optimization process.
- Investigate the use of reinforcement learning for quantum circuit optimization.