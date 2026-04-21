# Quantum Circuit Optimization

A comprehensive toolkit for quantum circuit optimization using large language models (LLMs). This project provides tools for generating quantum circuit datasets and optimizing them using the OpenRouter AI provider.

## Features

- **Dataset Generation**: Create random quantum circuits using Cirq with configurable parameters
- **Circuit Simulation**: Simulate circuits with optional noise models
- **AI-Powered Optimization**: Use LLMs (OpenRouter) for circuit optimization via in-context learning
- **Unified CLI**: Simple command-line interface
- **Extensible Architecture**: Easy to add new optimization providers and circuit types

## Project Structure

```
quantum-circuit-optimization/
├── src/
│   ├── data_generation/          # Quantum circuit dataset generation
│   │   ├── qc/                   # Core quantum computing modules
│   │   └── scripts/              # Data generation scripts
│   └── model_training/           # AI-powered circuit optimization
│       ├── cli/                  # Unified command-line interface
│       ├── config.py             # Configuration management
│       └── optimizer.py          # OpenRouter API integration
├── tests/                        # Test suites
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.10+
- Virtual environment support (venv, uv, conda, etc.)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd quantum-circuit-optimization
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Using uv (recommended)
   .venv/Scripts/python.exe -m uv venv .venv
   .venv/Scripts/python.exe -m ensurepip
   .venv/Scripts/python.exe -m pip install uv
   .venv/Scripts/python.exe -m uv pip install -r requirements.txt

   # Or using standard venv
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## Data Generation

Generate quantum circuit datasets for training and testing:

```bash
# Generate 100 circuits with default parameters
python src/data_generation/scripts/generate_dataset.py

# Generate custom dataset
python src/data_generation/scripts/generate_dataset.py \
    --n_circuits 500 \
    --min_gates 2 \
    --max_gates 10 \
    --n_qubits 3 \
    --output_file custom_dataset.jsonl
```

### Configuration Options
- `--n_circuits`: Number of circuits to generate (default: 100)
- `--min_gates`/`--max_gates`: Gate count range per circuit
- `--n_qubits`: Number of qubits (default: 5)
- `--noise_level`: Simulation noise level (default: 0.01)

## Circuit Optimization

Optimize quantum circuits using OpenRouter AI:

### Quick Start

1. **Set up API credentials:**
   ```bash
   # For OpenRouter (default)
   export OPENROUTER_API_KEY="***"
   # Or create ~/.api-openrouter file with your key
   ```

2. **Optimize a circuit:**
   ```bash
   python -m src.model_training.cli.predict \
       --input_circuit "H 0 ; CNOT 0 1 ; H 0"
   ```

### Advanced Usage

```bash
# Custom examples for in-context learning
python -m src.model_training.cli.predict \
    --input_circuit "X 0 ; X 0 ; Y 1" \
    --example "H 0 ; CNOT 0 1 ; H 0||CNOT 0 1" \
    --example "X 0 ; X 0 ; Y 1||Y 1" \
    --timeout 120 \
    --verbose

# Override model
python -m src.model_training.cli.predict \
    --model "deepseek/deepseek-chat-v3-0324" \
    --input_circuit "H 0 ; CNOT 0 1 ; H 0"
```

### Configuration Files

Create configuration files for model selection:

- `~/.model-openrouter`: Specify OpenRouter model (one line)
- `~/.api-openrouter`: OpenRouter API key

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Install test dependencies
pip install -e .[test]

# Run all tests
pytest

# Run specific test suites
pytest src/data_generation/tests/
pytest src/model_training/tests/
```

## Development

### Adding New Providers

1. Add provider configuration to `src/model_training/config.py`
2. Implement optimization function following the pattern in the existing optimizer
3. Add provider to the CLI in `src/model_training/cli/predict.py`

### Code Quality

- All code follows PEP 8 style guidelines
- Type hints are used throughout the codebase
- Comprehensive error handling and logging
- Extensive test coverage

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Future Work

### Data Generation
- Support for custom gate sets
- Advanced noise models
- Circuit complexity metrics
- Integration with other quantum frameworks

### Model Training
- Support for additional AI providers (Claude, GPT-4, etc.)
- Automated evaluation metrics
- Reinforcement learning approaches
- Error mitigation strategies

### Performance
- Batch processing capabilities
- Caching mechanisms
- Async API calls
- Resource optimization

## License

This project is licensed under the MIT-0 License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions:
1. Check the [Issues](http://github.com/waifuai/quantum-circuit-optimization/issues) page
2. Create a new issue with detailed information
3. Provide sample inputs and expected outputs
