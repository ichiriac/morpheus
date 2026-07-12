"""Politique = Qwen. Propose K actions candidates au tour courant (étape PROPOSER)."""

from __future__ import annotations

import json
import re

from ..llm.base import LLMClient, system, user
from ..orchestrator.types import Action, State
from ..text import snap_to_whitelist, strip_reasoning

_SYS = (
    "Tu es la politique d'un agent outillé. À partir de l'objectif et de l'état courant, "
    "tu proposes plusieurs actions candidates PLAUSIBLES et DISTINCTES pour le prochain "
    "pas — pas le plan entier, juste le prochain coup.\n"
    "Contraintes STRICTES :\n"
    "- N'utilise QUE des outils de la liste fournie (noms exacts).\n"
    "- Une action par ligne, format exact : "
    "`ACTION: <nom_outil> | ARGS: {\"clef\": \"valeur\"}`.\n"
    "- Pas de texte autour, pas de numérotation, pas de commentaire."
)


class Policy:
    def __init__(self, llm: LLMClient, k: int = 4) -> None:
        self.llm = llm
        self.k = k

    def build_prompt(self, state: State, tools: list[str]) -> str:
        return (
            f"[GOAL]{state.goal}[/GOAL]\n"
            f"[STATE]{state.text}[/STATE]\n"
            f"Historique récent : {state.history[-5:]}\n"
            f"[CANDIDATE_TOOLS]{', '.join(tools)}[/CANDIDATE_TOOLS]\n"
            f"Propose jusqu'à {self.k} actions candidates distinctes."
        )

    def propose(self, state: State, tools: list[str]) -> list[Action]:
        raw = self.llm.complete([system(_SYS), user(self.build_prompt(state, tools))])
        actions = _parse_actions(raw, tools)
        if not actions:  # filet de sécurité : au moins une action valide
            actions = [Action(tool=tools[0])] if tools else [Action(tool="noop")]
        return actions[: self.k]


def _parse_actions(raw: str, tools: list[str]) -> list[Action]:
    """Parse tolérant pour de vrais LLM : retire le raisonnement, lit les lignes ACTION,
    et, à défaut, repère les noms d'outils autorisés apparaissant dans le texte. Tout nom
    est ramené (snap) sur la whitelist ; les hallucinations non résolubles sont écartées."""
    text = strip_reasoning(raw)
    actions: list[Action] = []
    seen: set[str] = set()

    def add(tool_raw: str, args: dict) -> None:
        tool = snap_to_whitelist(tool_raw, tools)
        if tool is None:
            return
        key = f"{tool}:{sorted(args.items())}"
        if key in seen:
            return
        seen.add(key)
        actions.append(Action(tool=tool, args=args))

    # 1) format attendu : lignes `ACTION: <tool> | ARGS: {…}`
    for line in text.splitlines():
        m = re.search(r"ACTION:\s*([\w.\-]+)", line)
        if not m:
            continue
        args: dict = {}
        am = re.search(r"ARGS:\s*(\{.*\})", line)
        if am:
            try:
                args = json.loads(am.group(1))
            except json.JSONDecodeError:
                args = {}
        add(m.group(1), args)

    # 2) filet : si le modèle n'a pas suivi le format, repérer les outils cités dans l'ordre
    if not actions:
        for tok in re.findall(r"[\w.\-]+", text):
            if tok in tools:
                add(tok, {})

    return actions
