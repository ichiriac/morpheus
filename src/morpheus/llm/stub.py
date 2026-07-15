"""Backend LLM déterministe, sans réseau.

But : faire tourner toute la boucle (policy + world-model + orchestrateur) en CI et en
smoke test, sans GPU ni clé API. Il n'a AUCUNE vocation de performance — il expose juste
un comportement stable et lisible pour valider la plomberie et le calcul des métriques.

Contrat spécial : la politique lui envoie, dans le dernier message `user`, un bloc
``[CANDIDATE_TOOLS] a, b, c`` listant les outils disponibles + le texte de l'état. Si l'état
annonce l'étape attendue, le stub la propose EN PREMIER suivie de deux autres outils — K > 1
candidats, ce dont dépend le lookahead de `loop.py` (gardé par `len(candidates) > 1`) ; sinon
il retombe sur un seul outil, celui dont le nom recoupe le plus l'état. Pour le world-model,
il renvoie une prédiction textuelle triviale.

Les balises des blocs viennent de `morpheus.prompt_tags` — partagées avec les producteurs de
prompts. Ne PAS réécrire un littéral ici : ce lecteur et `agents/policy.py` ont déjà divergé
une fois (`[STATE]` vs `[ÉTAT COURANT]`), ce qui a désactivé le MPC sans rien casser.
"""

from __future__ import annotations

import re

from .. import prompt_tags as T
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
        if T.PREDICT_NEXT_STATE in last:
            return "PRED: l'action progresse d'un cran vers l'objectif (prédiction stub)."

        # --- Mode sonde « réductibilité » (signal du routeur de surprise) ---
        if T.EXPLAIN_GAP in last:
            # heuristique stable : recouvrement prédit/réel → écart d'autant plus réductible
            pred, real = _tokens(_extract(last, T.PREDICTED)), _tokens(_extract(last, T.REAL))
            union = len(pred | real) or 1
            return f"REDUCTIBLE: {round(10 * len(pred & real) / union)}"

        # --- Mode juge de proximité au but ---
        if T.SCORE_GOAL_DISTANCE in last:
            # score heuristique : recouvrement de tokens état/but
            goal = _extract(last, T.GOAL)
            state = _extract(last, T.STATE)          # bloc écrit par le world-model
            overlap = len(_tokens(goal) & _tokens(state))
            return f"SCORE: {overlap}"

        # --- Mode politique : proposer une action parmi les outils candidats ---
        tools = _extract_list(last, T.CANDIDATE_TOOLS)
        if tools:
            # `POLICY_STATE`, pas `STATE` : c'est la balise que `Policy.build_prompt` écrit.
            state = _extract(last, T.POLICY_STATE)
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
    """Lit le bloc `[tag]…[/tag]` écrit par `prompt_tags.block` (fermeture manquante tolérée :
    on s'arrête au paragraphe). `tag` vient de `prompt_tags` et peut contenir un espace ou un
    accent (« ÉTAT COURANT ») : on l'échappe plutôt que de l'interpoler en brut dans la regex."""
    t = re.escape(tag)
    m = re.search(rf"\[{t}\](.*?)(?:\[/{t}\]|\n\n|$)", text, re.S)
    return m.group(1).strip() if m else ""


def _extract_list(text: str, tag: str) -> list[str]:
    raw = _extract(text, tag)
    if not raw:
        return []
    return [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
