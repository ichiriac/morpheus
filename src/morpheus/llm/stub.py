"""Backend LLM déterministe, sans réseau.

But : faire tourner toute la boucle (policy + world-model + orchestrateur) en CI et en
smoke test, sans GPU ni clé API. Il n'a AUCUNE vocation de performance — il expose juste
un comportement stable et lisible pour valider la plomberie et le calcul des métriques.

Contrat spécial : la politique lui envoie, dans le dernier message `user`, un bloc
``[CANDIDATE_TOOLS] a, b, c`` listant les outils disponibles + le texte de l'état. Le stub
répond en choisissant l'outil dont le nom partage le plus de mots avec l'objectif/état.
Pour le world-model, il renvoie une prédiction textuelle triviale.
"""

from __future__ import annotations

import re

from ..config import LLMConfig
from .base import Message


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]+", s.lower()))


class StubLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    def complete(self, messages: list[Message], **kwargs) -> str:
        last = messages[-1].content if messages else ""

        # --- Mode world-model : on demande une prédiction d'état ---
        if "[PREDICT_NEXT_STATE]" in last:
            return "PRED: l'action progresse d'un cran vers l'objectif (prédiction stub)."

        # --- Mode juge de proximité au but ---
        if "[SCORE_GOAL_DISTANCE]" in last:
            # score heuristique : recouvrement de tokens état/but
            goal = _extract(last, "GOAL")
            state = _extract(last, "STATE")
            overlap = len(_tokens(goal) & _tokens(state))
            return f"SCORE: {overlap}"

        # --- Mode politique : proposer une action parmi les outils candidats ---
        tools = _extract_list(last, "CANDIDATE_TOOLS")
        if tools:
            state = _extract(last, "STATE")
            # 1) indice explicite d'état ("prochaine étape attendue : X") — compréhension
            #    de lecture qu'un vrai LLM ferait aussi.
            hint = re.search(r"\battendue\s*:\s*([\w.\-]+)", state)
            expected = hint.group(1).rstrip(".") if hint else None
            if expected and expected in tools:
                others = [t for t in tools if t != expected][:2]
                lines = [f"ACTION: {expected} | ARGS: {{}}"]
                lines += [f"ACTION: {t} | ARGS: {{}}" for t in others]
                return "\n".join(lines)
            # 2) sinon : recouvrement de tokens avec l'état
            best = max(tools, key=lambda t: len(_tokens(t) & _tokens(state)))
            return f"ACTION: {best} | ARGS: {{}}"

        return "ACTION: noop | ARGS: {}"


def _extract(text: str, tag: str) -> str:
    m = re.search(rf"\[{tag}\](.*?)(?:\[/{tag}\]|\n\n|$)", text, re.S)
    return m.group(1).strip() if m else ""


def _extract_list(text: str, tag: str) -> list[str]:
    raw = _extract(text, tag)
    if not raw:
        return []
    return [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
