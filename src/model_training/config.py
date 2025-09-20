"""
Configuration module for quantum circuit optimization.

This module centralizes configuration settings for different providers
and provides utilities for configuration management.
"""
from pathlib import Path
from typing import Optional
import os

# Default model configurations
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

# Configuration file paths
OPENROUTER_MODEL_FILE = Path.home() / ".model-openrouter"
GEMINI_MODEL_FILE = Path.home() / ".model-gemini"
OPENROUTER_API_KEY_FILE = Path.home() / ".api-openrouter"
GEMINI_API_KEY_FILE = Path.home() / ".api-gemini"

# Environment variable names
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
GEMINI_API_KEY_ENV_VARS = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]

# API endpoints
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default timeouts (seconds)
DEFAULT_TIMEOUT = 60

# Default provider
DEFAULT_PROVIDER = "openrouter"

# Supported providers
SUPPORTED_PROVIDERS = ["openrouter", "gemini"]


def resolve_model_from_file(file_path: Path, fallback: str) -> str:
    """
    Resolve model name from a configuration file.

    Args:
        file_path: Path to the configuration file
        fallback: Fallback model name if file doesn't exist or is empty

    Returns:
        The resolved model name
    """
    try:
        if file_path.is_file():
            for line in file_path.read_text(encoding="utf-8").splitlines():
                model = line.strip()
                if model:
                    return model
    except Exception:
        pass
    return fallback


def resolve_api_key_from_file(file_path: Path) -> Optional[str]:
    """
    Resolve API key from a configuration file.

    Args:
        file_path: Path to the API key file

    Returns:
        The API key if found, None otherwise
    """
    try:
        if file_path.is_file():
            return file_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def resolve_openrouter_model(explicit: Optional[str] = None) -> str:
    """Resolve OpenRouter model name."""
    if explicit and explicit.strip():
        return explicit.strip()
    return resolve_model_from_file(OPENROUTER_MODEL_FILE, DEFAULT_OPENROUTER_MODEL)


def resolve_gemini_model(explicit: Optional[str] = None) -> str:
    """Resolve Gemini model name."""
    if explicit and explicit.strip():
        return explicit.strip()
    return resolve_model_from_file(GEMINI_MODEL_FILE, DEFAULT_GEMINI_MODEL)


def resolve_openrouter_api_key() -> Optional[str]:
    """Resolve OpenRouter API key."""
    # Check environment variable first
    env_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if env_key and env_key.strip():
        return env_key.strip()

    # Fall back to file
    return resolve_api_key_from_file(OPENROUTER_API_KEY_FILE)


def resolve_gemini_api_key() -> Optional[str]:
    """Resolve Gemini API key."""
    # Check environment variables in order
    for env_var in GEMINI_API_KEY_ENV_VARS:
        key = os.getenv(env_var)
        if key and key.strip():
            return key.strip()

    # Fall back to file
    return resolve_api_key_from_file(GEMINI_API_KEY_FILE)