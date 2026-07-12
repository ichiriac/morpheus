"""Couche LLM : une interface unique, plusieurs backends.

- stub      : déterministe, sans réseau — pour smoke tests et CI.
- openai    : Qwen local via endpoint OpenAI-compatible (vLLM / llama.cpp / TGI).
- anthropic : ligne de référence supérieure (API Sonnet 4.6).
"""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMClient, Message


def build_llm(cfg: LLMConfig) -> LLMClient:
    if cfg.kind == "stub":
        from .stub import StubLLM

        return StubLLM(cfg)
    if cfg.kind == "openai":
        from .openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(cfg)
    if cfg.kind == "anthropic":
        from .anthropic_client import AnthropicLLM

        return AnthropicLLM(cfg)
    raise ValueError(f"backend LLM inconnu : {cfg.kind!r}")


__all__ = ["LLMClient", "Message", "build_llm"]
