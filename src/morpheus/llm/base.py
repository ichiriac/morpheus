"""Interface LLM commune à tous les backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Contrat minimal attendu par la politique et le world-model."""

    def complete(self, messages: list[Message], **kwargs) -> str:
        """Renvoie le texte de la complétion pour une liste de messages."""
        ...


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)
