"""Mémoire épisodique de faits atomiques (LWM-Planner) : extraction, dédup, récupération,
et intégration dans la boucle (accumulation par pas + récupération sur surprise)."""

from __future__ import annotations

from morpheus.agents.memory import FactMemory, extract_facts
from morpheus.agents.policy import Policy
from morpheus.agents.surprise import SurpriseRouter
from morpheus.agents.world_model import WorldModel
from morpheus.config import LLMConfig, OrchestratorConfig
from morpheus.llm import build_llm
from morpheus.orchestrator.loop import Orchestrator
from morpheus.orchestrator.types import Action, Observation, StepResult


def test_extract_facts_json_keys():
    obs = 'tool: {"user_id": "yusuf_rossi_9620", "status": "pending", "address": {"city": "X"}}'
    facts = extract_facts("get_user_details(user_id='u')", obs)
    assert "get_user_details: user_id = yusuf_rossi_9620" in facts
    assert "get_user_details: status = pending" in facts
    assert all("address" not in f for f in facts)          # dict imbriqué ignoré (scalaires only)


def test_extract_facts_plain_text():
    facts = extract_facts("find_user_id_by_email(email='x')", "tool: yusuf_rossi_9620")
    assert facts == ["find_user_id_by_email(email='x') → yusuf_rossi_9620"]
    assert extract_facts("noop()", "") == []               # observation vide → rien


def test_fact_memory_dedup_and_retrieve():
    m = FactMemory()
    m.observe("get_user_details(user_id='u')", Observation(text='tool: {"user_id": "abc123"}'))
    m.observe("get_user_details(user_id='u')", Observation(text='tool: {"user_id": "abc123"}'))
    assert len(m) == 1                                     # dédupliqué
    m.observe("check_status()", Observation(text="tool: airplane mode is on"))
    hits = m.retrieve("what is the user_id abc123", 2)
    assert hits and "abc123" in hits[0].text
    assert m.retrieve("totally unrelated zzz", 3) == []    # aucun token partagé → rien


# --- intégration boucle -------------------------------------------------------

class _TwoCand(Policy):
    """2 candidats fixes (→ branche world-model active) — pas d'appel LLM réel nécessaire."""

    def propose(self, state, tools, system_context=None, transcript=None, facts=None, route=None):
        return [Action(tool=tools[0]), Action(tool=tools[min(1, len(tools) - 1)])]


class _EchoEnv:
    """Env déterministe : observations à vocabulaire stable → la mémoire accumule des faits
    dont les tokens recouvrent les requêtes suivantes (récupération mémoire testable)."""

    def __init__(self):
        self.t = 0

    def reset(self):
        return Observation(text="session start alpha beta")

    def step(self, action):
        self.t += 1
        return StepResult(Observation(text=f"alpha beta gamma result {self.t}"),
                          0.0, self.t >= 5, {"success": False})

    def goal(self):
        return "reach alpha beta gamma"

    def tool_names(self):
        return ["toolA", "toolB"]

    def required_turns(self):
        return 5


def _orch(use_memory):
    llm = build_llm(LLMConfig(kind="stub"))
    cfg = OrchestratorConfig(k_candidates=2, horizon=1, max_turns=6, use_world_model=True,
                             surprise_threshold=-1.0, use_memory=use_memory, memory_top_k=3)
    return Orchestrator(_TwoCand(llm, k=2), WorldModel(llm), cfg, SurpriseRouter())


def test_loop_accumulates_and_retrieves_memory():
    result = _orch(use_memory=True).run(_EchoEnv())
    mem_facts = [f for s in result.trace for f in s.retrieved_facts if f.startswith("[memory:")]
    assert mem_facts, "la mémoire épisodique doit être récupérée sur surprise"
    # jamais au tour 1 : la mémoire est vide avant la 1re observation mémorisée.
    assert not any(f.startswith("[memory:") for f in result.trace[0].retrieved_facts)


def test_loop_memory_off_by_default():
    result = _orch(use_memory=False).run(_EchoEnv())
    assert not any(f.startswith("[memory:") for s in result.trace for f in s.retrieved_facts)
