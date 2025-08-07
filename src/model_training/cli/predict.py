import argparse
from typing import List, Tuple, Optional
import os
from pathlib import Path
import sys
import requests

# Package-local imports (keep 'src.' when running via -m as per project guidance)
from src.model_training.gemini_optimizer import optimize_circuit_with_gemini

# Dotfiles for model resolution per provider
OPENROUTER_MODEL_FILE = Path.home() / ".model-openrouter"
GEMINI_MODEL_FILE = Path.home() / ".model-gemini"

# Defaults
DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_MODEL = "openrouter/horizon-beta"

# OpenRouter credentials and endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY_FILE = Path.home() / ".api-openrouter"


def _resolve_first_nonempty_line(p: Path, fallback: str) -> str:
    try:
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                v = line.strip()
                if v:
                    return v
    except Exception:
        pass
    return fallback


def _resolve_openrouter_model(explicit: Optional[str]) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return _resolve_first_nonempty_line(OPENROUTER_MODEL_FILE, DEFAULT_OPENROUTER_MODEL)


def _resolve_gemini_model(explicit: Optional[str]) -> str:
    # Reuse resolver implemented in gemini_optimizer
    from src.model_training.gemini_optimizer import _resolve_gemini_model as _rgm  # type: ignore
    if explicit and explicit.strip():
        return explicit.strip()
    return _rgm()


def _resolve_openrouter_api_key() -> Optional[str]:
    env_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        if OPENROUTER_API_KEY_FILE.is_file():
            return OPENROUTER_API_KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


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
    api_key = _resolve_openrouter_api_key()
    if not api_key:
        return None
    payload = {
        "model": _resolve_openrouter_model(model),
        "messages": [{"role": "user", "content": _optimize_prompt(examples, unoptimized_circuit)}],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        content = (choices[0].get("message", {}).get("content") or "").strip()
        return content or None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified CLI to optimize quantum circuits via providers (default: openrouter)."
    )
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=["openrouter", "gemini"],
                        help="Which provider to use. Default: openrouter")
    parser.add_argument("--model", type=str, default=None, help="Override model name for the chosen provider")
    parser.add_argument("--input_circuit", type=str, required=True,
                        help="Input quantum circuit as a string (e.g., 'H 0; CNOT 0 1; H 0').")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP/API timeout seconds (for openrouter)")
    parser.add_argument("--example", action="append", default=[],
                        help="Add example pair as 'input||output'. Can be used multiple times.")
    args = parser.parse_args()

    examples: List[Tuple[str, str]] = []
    for ex in args.example:
        try:
            inp, out = ex.split("||", 1)
            examples.append((inp.strip(), out.strip()))
        except Exception:
            # Ignore malformed example entries
            pass
    if not examples:
        examples = [
            ("H 0; CNOT 0 1; H 0", "CNOT 0 1"),
            ("X 0; X 0; Y 1", "Y 1"),
        ]

    if args.provider == "openrouter":
        optimized = call_openrouter_optimize(args.input_circuit, examples, model=args.model, timeout=args.timeout)
        if optimized is None:
            print("OpenRouter call failed or returned empty content.", file=sys.stderr)
            sys.exit(1)
        print(optimized)
        return

    # Gemini path
    resolved_model = _resolve_gemini_model(args.model)
    try:
        optimized = optimize_circuit_with_gemini(args.input_circuit, examples, model=resolved_model)  # type: ignore
        print(optimized)
    except Exception as e:
        print(f"Gemini error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()