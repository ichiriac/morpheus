"""Backend Anthropic — ligne de référence supérieure sur τ²-bench (API Sonnet 4.6).

config :
    kind: anthropic
    model: claude-sonnet-4-6            # cf. specs/02 pour les IDs à jour
    api_key_env: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os

from ..config import LLMConfig
from .base import Message


class AnthropicLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover - dépend de l'extra `anthropic`
            raise ImportError(
                "backend 'anthropic' requis : `pip install morpheus[anthropic]`"
            ) from e

        key = os.environ.get(cfg.api_key_env or "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                f"clé API absente : variable d'env {cfg.api_key_env or 'ANTHROPIC_API_KEY'}"
            )
        self.cfg = cfg
        self._client = Anthropic(api_key=key)

    def complete(self, messages: list[Message], **kwargs) -> str:
        system = "\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        resp = self._client.messages.create(
            model=self.cfg.model,
            system=system or None,
            messages=turns,
            temperature=kwargs.get("temperature", self.cfg.temperature),
            max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
        )
        return "".join(block.text for block in resp.content if block.type == "text")
