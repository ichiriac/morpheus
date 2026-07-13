"""Le contexte système (policy du domaine) va au VRAI PROPOSER, pas aux rollouts world-model."""

from __future__ import annotations

from morpheus.agents.policy import Policy
from morpheus.agents.world_model import WorldModel
from morpheus.llm.base import Message
from morpheus.orchestrator.types import Action, Observation, State

_POLICY_MARKER = "RÈGLE-DOMAINE-XYZ : toujours authentifier avant tout remboursement."


class _CapturingLLM:
    """Capture les messages `system` et `user` vus, renvoie une action bidon parsable."""

    def __init__(self) -> None:
        self.systems: list[str] = []
        self.users: list[str] = []

    def complete(self, messages: list[Message], **kwargs) -> str:
        self.systems.append(next((m.content for m in messages if m.role == "system"), ""))
        self.users.append(next((m.content for m in messages if m.role == "user"), ""))
        return "ACTION: authenticate_user | ARGS: {}"


def _state() -> State:
    return State(goal="but", observation=Observation(text="état"))


def test_propose_injects_system_context():
    llm = _CapturingLLM()
    Policy(llm).propose(_state(), ["authenticate_user"], system_context=_POLICY_MARKER)
    assert _POLICY_MARKER in llm.systems[-1]


def test_propose_without_context_stays_lean():
    llm = _CapturingLLM()
    Policy(llm).propose(_state(), ["authenticate_user"])
    assert _POLICY_MARKER not in llm.systems[-1]


def test_world_model_rollout_does_not_leak_policy_into_prompts():
    """rollout appelle policy.propose SANS system_context → les prompts imaginés restent légers."""
    pol_llm = _CapturingLLM()
    wm_llm = _CapturingLLM()
    policy = Policy(pol_llm)
    wm = WorldModel(wm_llm)
    wm.rollout(policy, _state(), Action(tool="authenticate_user"),
               ["authenticate_user"], horizon=2)
    # aucun prompt (politique imaginée OU world-model) ne doit contenir la policy du domaine
    assert all(_POLICY_MARKER not in s for s in pol_llm.systems + wm_llm.systems)


_RESULT_MARKER = "ORDERS: #A123 mechanical keyboard"


def test_propose_renders_transcript_results():
    """Le scratchpad (action→résultat) doit apparaître dans le prompt du vrai PROPOSER."""
    llm = _CapturingLLM()
    transcript = [("get_user_details(user_id='u1')", _RESULT_MARKER)]
    Policy(llm).propose(_state(), ["get_user_details"], transcript=transcript)
    assert _RESULT_MARKER in llm.users[-1]


def test_rollout_does_not_carry_transcript():
    """Les rollouts imaginés appellent propose() sans transcript → mémoire hors world-model."""
    pol_llm = _CapturingLLM()
    policy = Policy(pol_llm)
    WorldModel(_CapturingLLM()).rollout(
        policy, _state(), Action(tool="get_user_details"), ["get_user_details"], horizon=2
    )
    assert all(_RESULT_MARKER not in u for u in pol_llm.users)
