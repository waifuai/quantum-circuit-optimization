# Model Training: AI-Powered Quantum Circuit Optimization

This module provides AI-powered quantum circuit optimization using large language models through multiple providers.

## Features

- **Multi-Provider Support**: OpenRouter (default) and Google Gemini
- **Unified CLI**: Single interface for all optimization providers
- **In-Context Learning**: Uses examples to guide optimization
- **Configurable Models**: Easy model selection via configuration files
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Architecture

- `cli/predict.py`: Unified command-line interface for circuit optimization
- `config.py`: Centralized configuration management for all providers
- `gemini_optimizer.py`: Google Gemini API integration and optimization logic

## Quick Start

1. **Set up credentials:**
   ```bash
   # For OpenRouter
   export OPENROUTER_API_KEY="your-api-key"
   # Or create ~/.api-openrouter file

   # For Google Gemini
   export GEMINI_API_KEY="your-api-key"
   # Or create ~/.api-gemini file
   ```

2. **Optimize a circuit:**
   ```bash
   # Using OpenRouter (default)
   python -m src.model_training.cli.predict \
       --input_circuit "H 0 ; CNOT 0 1 ; H 0"

   # Using Google Gemini
   python -m src.model_training.cli.predict \
       --provider gemini \
       --input_circuit "H 0 ; CNOT 0 1 ; H 0"
   ```

## Configuration

### Model Selection
Configure default models using dotfiles:

- `~/.model-openrouter`: OpenRouter model name (fallback: `deepseek/deepseek-chat-v3-0324:free`)
- `~/.model-gemini`: Gemini model name (fallback: `gemini-2.5-pro`)

### API Keys
Set API keys via environment variables or files:

- `OPENROUTER_API_KEY` or `~/.api-openrouter`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` or `~/.api-gemini`

## Advanced Usage

### Custom Examples
Provide domain-specific examples for better optimization:

```bash
python -m src.model_training.cli.predict \
    --input_circuit "X 0 ; X 0 ; Y 1" \
    --example "H 0 ; CNOT 0 1 ; H 0||CNOT 0 1" \
    --example "X 0 ; X 0 ; Y 1||Y 1" \
    --provider gemini
```

### Model Override
Use specific models for optimization:

```bash
python -m src.model_training.cli.predict \
    --provider openrouter \
    --model "openrouter/anthropic/claude-3-sonnet" \
    --input_circuit "H 0 ; CNOT 0 1 ; H 0"
```

### Verbose Logging
Enable detailed logging for debugging:

```bash
python -m src.model_training.cli.predict \
    --input_circuit "H 0 ; CNOT 0 1 ; H 0" \
    --verbose
```

## Adding New Providers

To add a new optimization provider:

1. **Update Configuration** (`config.py`):
   ```python
   # Add new provider constants
   NEW_PROVIDER_MODEL = "new-provider/model"
   ```

2. **Add Provider Function**:
   ```python
   def call_new_provider_optimize(unoptimized_circuit, examples, model, timeout):
       # Implementation here
       pass
   ```

3. **Update CLI** (`cli/predict.py`):
   ```python
   # Add to choices and main logic
   parser.add_argument("--provider", choices=["openrouter", "gemini", "new_provider"])
   ```

## Error Handling

The module includes comprehensive error handling:

- **API Failures**: Graceful degradation with informative messages
- **Network Issues**: Timeout handling and retry logic
- **Configuration Errors**: Clear guidance on setup requirements
- **Validation**: Input validation for circuits and examples

## Development

### Testing
```bash
# Run model training tests
pytest src/model_training/tests/

# Test specific functionality
pytest src/model_training/tests/test_cli.py
```

### Code Quality
- Full type hints throughout
- PEP 8 compliance
- Comprehensive docstrings
- Logging for all operations

## Future Enhancements

- **Additional Providers**: Support for Claude, GPT-4, and other LLMs
- **Batch Processing**: Optimize multiple circuits simultaneously
- **Caching**: Response caching to reduce API calls
- **Metrics**: Automated evaluation of optimization quality
- **Async Support**: Non-blocking API calls for better performance