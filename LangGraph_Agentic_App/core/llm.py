from __future__ import annotations

import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config import (
    DEFAULT_LLM_PROVIDER,
    LLM_MODEL_ANTHROPIC,
    LLM_MODEL_OPENAI,
    LLM_PROVIDER_ANTHROPIC,
    LLM_PROVIDER_OPENAI,
)


LLMProvider = Literal["openai", "anthropic"]

_PROVIDER_ALIASES: dict[str, LLMProvider] = {
    "openai": LLM_PROVIDER_OPENAI,
    "gpt": LLM_PROVIDER_OPENAI,
    "anthropic": LLM_PROVIDER_ANTHROPIC,
    "claude": LLM_PROVIDER_ANTHROPIC,
}


def normalise_llm_provider(provider: str | None) -> LLMProvider:
    text = str(provider or "").strip().lower()
    if text in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[text]
    return LLM_PROVIDER_OPENAI if DEFAULT_LLM_PROVIDER == LLM_PROVIDER_OPENAI else LLM_PROVIDER_ANTHROPIC


def get_default_llm_model(provider: str | None) -> str:
    normalised_provider = normalise_llm_provider(provider)
    if normalised_provider == LLM_PROVIDER_ANTHROPIC:
        return LLM_MODEL_ANTHROPIC
    return LLM_MODEL_OPENAI


def resolve_llm_config(provider: str | None = None, model: str | None = None) -> tuple[LLMProvider, str]:
    normalised_provider = normalise_llm_provider(provider)
    model_name = str(model or "").strip() or get_default_llm_model(normalised_provider)
    return normalised_provider, model_name


def get_api_key_env_var(provider: str | None) -> str:
    normalised_provider = normalise_llm_provider(provider)
    if normalised_provider == LLM_PROVIDER_ANTHROPIC:
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    temperature: float = 0.1,
):
    normalised_provider, model_name = resolve_llm_config(provider, model)
    api_key = os.getenv(get_api_key_env_var(normalised_provider), "")

    if normalised_provider == LLM_PROVIDER_ANTHROPIC:
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=api_key)

    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)