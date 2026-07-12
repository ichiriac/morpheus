"""Backend Qwen local via endpoint OpenAI-compatible (vLLM / llama.cpp server / TGI).

Exemple de lancement d'un serveur vLLM :
    vllm serve Qwen/Qwen3-32B --port 8000 --max-model-len 32768
puis config :
    kind: openai
    model: Qwen/Qwen3-32B
    base_url: http://localhost:8000/v1
    api_key_env: null            # llama.cpp/vLLM local n'exigent pas de clé
"""

from __future__ import annotations

import os

from ..config import LLMConfig
from .base import Message


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - dépend de l'extra `openai`
            raise ImportError(
                "backend 'openai' requis : `pip install morpheus[openai]`"
            ) from e

        api_key = "not-needed"
        if cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env, "not-needed")
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=api_key)

    def complete(self, messages: list[Message], **kwargs) -> str:
        extra = self.cfg.extra_body or None
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.cfg.temperature),
            max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
            extra_body=extra,
        )
        return resp.choices[0].message.content or ""
