"""Le PROPOSER voit le tour courant et le budget — sinon il ne peut pas se rythmer.

Régression gardée : jusqu'au 2026-07-17 `state.turn` existait mais n'était JAMAIS rendu dans le
prompt. Mesuré sur `retail74_baseline_run2` (74 tâches) : 59 % des épisodes finissaient au plafond
de 16 tours, et les tâches à ≥10 actions de référence faisaient 0/9 — arithmétiquement hors
d'atteinte à ~2.67 tours consommés par pas expert. Un agent qui ignore son budget ne peut pas le
gérer ; ces tests empêchent que le bloc [BUDGET] disparaisse en silence.
"""

from __future__ import annotations

from morpheus.agents.policy import Policy
from morpheus.agents.world_model import WorldModel
from morpheus.llm.base import Message
from morpheus.orchestrator.types import Observation, State


class _CapturingLLM:
    def __init__(self) -> None:
        self.users: list[str] = []

    def complete(self, messages: list[Message], **kwargs) -> str:
        self.users.append(next((m.content for m in messages if m.role == "user"), ""))
        return "ACTION: a | ARGS: {}"


def _st(turn: int = 0, max_turns: int = 0) -> State:
    return State(goal="but", observation=Observation(text="état"), turn=turn, max_turns=max_turns)


def test_budget_rendu_avec_tour_et_restant():
    p = Policy(_CapturingLLM())
    out = p.build_prompt(_st(turn=12, max_turns=40), ["a"])
    assert "[BUDGET]" in out
    assert "Tour 12/40" in out
    assert "28 tours" in out          # 40 - 12 : c'est le RESTANT qui pilote le rythme


def test_budget_absent_si_tour_inconnu():
    # turn=0 = hors boucle (tests, appels directs) : ne pas fabriquer un budget imaginaire.
    assert "[BUDGET]" not in Policy(_CapturingLLM()).build_prompt(_st(), ["a"])


def test_budget_degrade_si_max_turns_inconnu():
    # turn connu mais budget inconnu : annoncer le tour, ne PAS inventer un restant.
    out = Policy(_CapturingLLM()).build_prompt(_st(turn=5), ["a"])
    assert "[BUDGET]" in out and "Tour 5." in out and "il reste" not in out


def test_restant_jamais_negatif():
    # Le cap peut être atteint ou dépassé ; un « il reste -1 tours » serait absurde.
    out = Policy(_CapturingLLM()).build_prompt(_st(turn=41, max_turns=40), ["a"])
    assert "il reste 0 tours" in out


def test_budget_atteint_le_llm_via_propose():
    # Le prompt CONSTRUIT ne sert à rien s'il n'est pas celui ENVOYÉ.
    llm = _CapturingLLM()
    Policy(llm).propose(_st(turn=3, max_turns=40), ["a"])
    assert "Tour 3/40" in llm.users[-1]


def test_rollout_propage_le_budget_aux_etats_imagines():
    # Sans propagation, les pas gloutons du lookahead verraient un budget vide alors que le vrai
    # PROPOSER en voit un : le WM simulerait un régime différent de celui qu'il doit anticiper.
    llm = _CapturingLLM()
    wm = WorldModel(llm)
    wm.rollout(Policy(llm), _st(turn=3, max_turns=40), "a()", ["a"], horizon=2)
    assert any("Tour 4/40" in u for u in llm.users), llm.users
