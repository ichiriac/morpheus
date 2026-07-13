"""World-model.

Phase 1 (ici) : LLM-as-world-model. Le LLM lui-même simule l'état résultant d'une action
et juge la distance au but — aucun entraînement, aucun JEPA. C'est la baseline « boucle
fermée + lookahead » à battre par la version latente.

Phase 2+ : sous-classer / remplacer `predict` et `score` par le prédicteur JEPA
`ŝ' = P(E_state(s), E_action(a))` et une distance dans l'espace latent. L'interface ne
change pas — seule l'implémentation devient latente.
"""

from __future__ import annotations

import re

from ..llm.base import LLMClient, system, user
from ..orchestrator.types import Action, State
from ..text import strip_reasoning

_SYS_PRED = (
    "Tu es un modèle du monde. On te donne un état et une action d'outil. Tu prédis, en une "
    "ou deux phrases, l'état résultant PLAUSIBLE — sans l'exécuter réellement."
)
_SYS_SCORE = (
    "Tu évalues à quel point un état est proche d'un objectif. Réponds strictement "
    "`SCORE: <entier 0-10>` où 10 = objectif atteint, 0 = très loin."
)


class WorldModel:
    """LLM-as-world-model (Phase 1). Sert au lookahead MPC et à la prédiction pour δ."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def predict(self, state: State, action: Action) -> str:
        """ŝ' textuel : l'état latent/texte prédit après `action`."""
        prompt = (
            "[PREDICT_NEXT_STATE]\n"
            f"[STATE]{state.text}[/STATE]\n"
            f"[ACTION]{action}[/ACTION]\n"
            "Prédis l'état résultant."
        )
        return strip_reasoning(self.llm.complete([system(_SYS_PRED), user(prompt)]))

    def score_to_goal(self, goal: str, state_text: str) -> float:
        """Proximité au but ∈ [0, 1] (1 = atteint). Judge LLM en Phase 1."""
        prompt = (
            "[SCORE_GOAL_DISTANCE]\n"
            f"[GOAL]{goal}[/GOAL]\n"
            f"[STATE]{state_text}[/STATE]"
        )
        raw = strip_reasoning(self.llm.complete([system(_SYS_SCORE), user(prompt)]))
        m = re.search(r"SCORE:\s*(\d+)", raw)
        val = int(m.group(1)) if m else 0
        return max(0.0, min(1.0, val / 10.0))

    def rollout(self, policy, state: State, first: Action, tools: list[str],
                horizon: int) -> tuple[float, str]:
        """Lookahead texte (MPC, boucle OUVERTE en imagination) : déroule `first` puis
        `horizon-1` pas gloutons imaginés. Renvoie `(meilleure proximité au but, ŝ' du 1er pas)`.
        Le ŝ' du 1er pas est réutilisé par la boucle (divergence) → évite un `predict` en double.
        En Phase 2 ce rollout se fera dans le latent JEPA."""
        first_pred = self.predict(state, first)
        best = self.score_to_goal(state.goal, first_pred)
        cur = State(goal=state.goal,
                    observation=type(state.observation)(text=first_pred),
                    turn=state.turn + 1,
                    history=state.history + [str(first)])
        for _ in range(max(0, horizon - 1)):
            cands = policy.propose(cur, tools)
            if not cands:
                break
            nxt = cands[0]
            imagined_text = self.predict(cur, nxt)
            best = max(best, self.score_to_goal(state.goal, imagined_text))
            cur = State(goal=state.goal,
                        observation=type(state.observation)(text=imagined_text),
                        turn=cur.turn + 1,
                        history=cur.history + [str(nxt)])
        return best, first_pred
