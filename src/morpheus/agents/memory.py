"""Mémoire épisodique de faits atomiques (style LWM-Planner, cf. specs/01 §RAG).

Le RAG sur la *policy* seule est REDONDANT : cette connaissance est déjà dans le system_context
de la politique (elle y tient). La mémoire épisodique sort de ce régime : elle accumule, au fil
de l'épisode, des **faits atomiques extraits des observations RÉELLES de l'agent** — donc une
connaissance qui N'EST PAS dans le prompt système, et qui excède la fenêtre de récence du
transcript ReAct. Sur surprise, on récupère par pertinence (BM25) les faits passés utiles, y
compris ceux tombés hors de la fenêtre du transcript (mémoire long-horizon pour 10+ tours).

Non-contamination : les faits proviennent de ce que l'agent a lui-même OBSERVÉ (résultats
d'outils, messages utilisateur) — pas du scénario caché, pas de la DB, pas des critères d'éval.
C'est de la mémoire, pas de la fuite.
"""

from __future__ import annotations

import json
import re

from ..orchestrator.types import Observation
from .knowledge import KnowledgeBase, Rule

_JSON_HINT = re.compile(r"[{\[]")
_PREFIX = re.compile(r"^(tool|user|system)\s*:\s*", re.IGNORECASE)


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def extract_facts(action: str, obs_text: str, max_facts: int = 8) -> list[str]:
    """Faits atomiques tirés d'une observation. JSON → un fait par paire clé/valeur scalaire
    (recall ciblé : « user_id = … » retrouvable longtemps après) ; sinon un fait distillé
    `action → résultat`. Déterministe, sans LLM."""
    text = (obs_text or "").strip()
    if not text:
        return []
    tool = action.split("(")[0].strip() or "action"
    body = _PREFIX.sub("", text)

    facts: list[str] = []
    if _JSON_HINT.search(body[:2]):
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (str, int, float, bool)) and str(v).strip():
                    facts.append(f"{tool}: {k} = {v}")
            if facts:
                return facts[:max_facts]

    return [f"{action} → {' '.join(body.split())[:180]}"]


class FactMemory:
    """Faits atomiques accumulés sur l'épisode, dédupliqués, interrogeables par BM25.
    Une instance PAR épisode (créée dans loop.run) — pas de fuite inter-tâches."""

    def __init__(self) -> None:
        self._rules: list[Rule] = []
        self._seen: set[str] = set()

    def __len__(self) -> int:
        return len(self._rules)

    def add(self, fact: str, source: str = "") -> None:
        key = _norm(fact)
        if key and key not in self._seen:
            self._seen.add(key)
            self._rules.append(Rule(domain="memory", section=source[:48], text=fact))

    def observe(self, action: str, obs: Observation) -> None:
        """Extrait et mémorise les faits d'une observation réelle (appelé après chaque pas)."""
        for f in extract_facts(action, obs.text):
            self.add(f, source=action.split("(")[0])

    def retrieve(self, query: str, k: int = 3) -> list[Rule]:
        """Les `k` faits mémorisés les plus pertinents pour `query` (vide si mémoire vide)."""
        if not self._rules:
            return []
        return KnowledgeBase(self._rules).retrieve(query, k)
