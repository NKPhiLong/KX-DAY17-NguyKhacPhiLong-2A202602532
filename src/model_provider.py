from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

# Common aliases and typos -> canonical provider name.
_PROVIDER_ALIASES = {
    "openai": "openai",
    "open_ai": "openai",
    "open-ai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "anthropics": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "olama": "ollama",
    "openrouter": "openrouter",
    "open_router": "openrouter",
    "open-router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_live_ready(self) -> bool:
        """True when enough settings exist to call a real model.

        Ollama runs locally and needs no key; every other provider needs one.
        """

        if not self.model_name:
            return False
        if self.provider == "ollama":
            return True
        if self.provider == "custom":
            return bool(self.api_key and self.base_url)
        return bool(self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; reject unknown providers early."""

    key = (value or "").strip().lower().replace(" ", "_")
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    raise ValueError(
        f"Unsupported LLM provider {value!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    SDK imports are lazy so the offline path never needs provider packages or keys.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature, "api_key": config.api_key}
        if provider == "custom":
            if not config.base_url:
                raise ValueError("Provider 'custom' requires CUSTOM_BASE_URL.")
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name, temperature=config.temperature, google_api_key=config.api_key
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    raise ValueError(f"Unsupported LLM provider {provider!r}.")
