from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

# Benchmark defaults: a standard thread (~10 short turns) stays below the threshold,
# while the 16-turn stress thread (~2300 user tokens) is compacted several times.
DEFAULT_COMPACT_THRESHOLD_TOKENS = 600
DEFAULT_COMPACT_KEEP_MESSAGES = 4

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Environment variable holding the API key / base URL for each provider.
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths for the repo root, dataset directory, and state directory.
    - Compact-memory settings: token threshold and number of recent messages to keep.
    - Provider settings for the main model and the judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Live LLM calls are opt-in (LLM_MODE=live) so benchmarks stay deterministic by default.
    live: bool = False


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Build a ProviderConfig from `<prefix>_PROVIDER`, `<prefix>_MODEL`, `<prefix>_TEMPERATURE`.

    The judge (prefix JUDGE) falls back to the main model's settings when unset.
    """

    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else DEFAULT_PROVIDER)
    provider = normalize_provider(raw_provider)

    model_name = os.getenv(f"{prefix}_MODEL")
    if not model_name:
        model_name = fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]

    key_env = API_KEY_ENV[provider]
    url_env = BASE_URL_ENV.get(provider)
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=_env_float(f"{prefix}_TEMPERATURE", fallback.temperature if fallback else 0.0),
        api_key=os.getenv(key_env) if key_env else None,
        base_url=os.getenv(url_env) if url_env else None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a populated LabConfig.

    Environment variables (all optional; offline mode needs none of them):
    - LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE
    - JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_TEMPERATURE (default: same as LLM_*)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    - CUSTOM_BASE_URL, CUSTOM_API_KEY, OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES
    - STATE_DIR (override where User.md and other state is written)
    - LLM_MODE=live to call the real model (default: offline, deterministic)
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = Path(os.getenv("STATE_DIR") or (root / "state")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    threshold = _env_int("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS)
    keep = _env_int("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES)
    if threshold <= 0 or keep <= 0:
        raise ValueError("COMPACT_THRESHOLD_TOKENS and COMPACT_KEEP_MESSAGES must be positive.")

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=model,
        judge_model=judge_model,
        live=os.getenv("LLM_MODE", "offline").strip().lower() == "live",
    )
