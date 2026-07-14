"""Recompute HORS-LIGNE des signaux du routeur depuis une trace journalisée (Phase 4).

Les trajectoires annotées (`data/annotations`) datent d'AVANT l'instrumentation de
`TraceStep.signals` : on recalcule ici les signaux déterministes exactement comme
`loop.py` les collecte en ligne — même détecteur d'erreur que l'adaptateur τ²
(`_looks_like_error`), même requête KB/mémoire (`f"{chosen} {real_state}"`), même replay
`FactMemory` (la mémoire n'intègre un pas qu'APRÈS la collecte de ses signaux). Quand un
pas porte des `signals` journalisés (traces post-instrumentation), les valeurs
journalisées PRIMENT : c'est ce que la boucle a réellement vu.

Limites assumées (encodées en None → indicateur « non sondé » dans `as_vector`) :
- l'observation d'OUVERTURE n'est pas journalisée dans la trace → au pas 1,
  `familiarity` = 0.0 et `score_before` = None ;
- `delta` = la divergence journalisée : 0.0 sur les pas SANS lookahead (baseline,
  candidat unique) — ce 0 signifie « non mesuré », pas « prédiction parfaite » ;
- `reducibility` (sonde LLM) n'est pas rejouable hors-ligne ;
- `score_before/after` exigent `scores` (score_to_goal par état, ex. JepaWorldModel via
  `--checkpoint` dans scripts/build_router_dataset.py) — sinon None.

Torch-free ET numpy-free : importable partout (même contrainte que agents/surprise.py).
"""

from __future__ import annotations

from typing import Any

from ..agents.knowledge import KnowledgeBase
from ..agents.memory import FactMemory
from ..agents.surprise import SurpriseSignals, familiarity
from ..envs.tau2_adapter import _looks_like_error
from ..orchestrator.types import Observation


def _tool_name(chosen: str) -> str:
    """Nom d'outil depuis la forme journalisée `str(Action)` : "tool(arg=…)" → "tool"."""
    return (chosen or "").split("(", 1)[0].strip()


def signals_for_episode(
    steps: list[dict[str, Any]],
    kb: KnowledgeBase | None = None,
    rag_top_k: int = 3,
    use_memory: bool = True,
    memory_top_k: int = 3,
    scores: list[float] | None = None,
) -> list[SurpriseSignals]:
    """Un `SurpriseSignals` par pas de la trace (dicts `TraceStep`, anciens ou nouveaux).

    `use_memory=True` par défaut (≠ loop, off par défaut) : pour un DATASET, plus de
    signal vaut mieux — le replay est déterministe et l'indicateur « sondé » garde la
    distinction à l'inférence. `scores[i]` = score_to_goal de `real_state[i]`."""
    out: list[SurpriseSignals] = []
    mem = FactMemory() if use_memory else None
    past_states: list[str] = []
    prev_tool: str | None = None
    prev_tool_error = False

    for i, step in enumerate(steps):
        chosen = str(step.get("chosen", ""))
        real = str(step.get("real_state", "") or "")
        rec: dict[str, Any] = step.get("signals") or {}

        # signature de l'outil : journalisée (traces récentes) sinon MÊME détecteur que τ²
        tool_error = (bool(step["tool_error"]) if "tool_error" in step
                      else _looks_like_error(real))

        # requête IDENTIQUE à loop.py étape 5 : l'état vrai + l'action qui a surpris
        query = f"{chosen} {real}"
        kb_top = rec.get("kb_top_score")
        kb_hits = rec.get("kb_hits")
        if kb_hits is None and kb is not None:
            top = [(s, r) for s, r in kb.score(query) if s > 0.0][:rag_top_k]
            kb_top = top[0][0] if top else 0.0
            kb_hits = len(top)

        mem_hits = rec.get("memory_hits")
        if mem_hits is None and mem is not None:
            mem_hits = len(mem.retrieve(query, memory_top_k))

        score_after = rec.get("score_after")
        if score_after is None and scores is not None:
            score_after = scores[i]
        score_before = rec.get("score_before")
        if score_before is None:
            score_before = step.get("score_before")
        if score_before is None and scores is not None and i > 0:
            score_before = scores[i - 1]

        out.append(SurpriseSignals(
            delta=float(step.get("divergence", 0.0) or 0.0),
            tool_error=tool_error,
            score_before=score_before,
            score_after=score_after,
            kb_top_score=kb_top,
            kb_hits=kb_hits,
            memory_hits=mem_hits,
            familiarity=familiarity(real, past_states),
            repeated_tool=(prev_tool is not None and _tool_name(chosen) == prev_tool
                           and not prev_tool_error),
            is_user_turn=_tool_name(chosen) == "respond_to_user",
            reducibility=rec.get("reducibility"),
        ))

        # miroir de l'étape 6 de loop.py : historique/mémoire n'intègrent le pas
        # qu'APRÈS la collecte de ses signaux
        past_states.append(real)
        prev_tool, prev_tool_error = _tool_name(chosen), tool_error
        if mem is not None:
            mem.observe(chosen, Observation(text=real, tool_error=tool_error))

    return out
