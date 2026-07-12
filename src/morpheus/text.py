"""Petits utilitaires de texte partagés (nettoyage des sorties de vrais LLM)."""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^```[\w]*\n?|\n?```$", re.M)


def strip_reasoning(text: str) -> str:
    """Retire les blocs de raisonnement (`<think>…</think>` de Qwen3) et les fences ```…```.

    Les modèles « thinking » émettent un préambule de raisonnement avant la réponse ; on le
    coupe pour que le parseur de la politique ne le confonde pas avec des actions."""
    text = _THINK.sub("", text)
    # si un <think> reste ouvert sans fermeture, on garde ce qui suit la dernière balise
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = _FENCE.sub("", text)
    return text.strip()


def tokens(s: str) -> set[str]:
    # découpe aussi sur underscore : "lookup_order" -> {"lookup", "order"}
    return set(re.findall(r"[a-z]+", s.lower()))


def snap_to_whitelist(name: str, allowed: list[str]) -> str | None:
    """Ramène un nom d'outil (possiblement halluciné/mal orthographié) vers la liste
    autorisée. Exact > insensible à la casse > meilleur recouvrement de tokens. None si
    aucun candidat plausible."""
    if name in allowed:
        return name
    low = name.lower()
    for a in allowed:
        if a.lower() == low:
            return a
    nt = tokens(name)
    if not nt:
        return None
    best, best_overlap = None, 0
    for a in allowed:
        ov = len(nt & tokens(a))
        if ov > best_overlap:
            best, best_overlap = a, ov
    return best if best_overlap > 0 else None
