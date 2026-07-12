"""Smoke tests : la boucle complète tourne avec stub + mock, sans réseau."""

from __future__ import annotations

from morpheus.agents.policy import Policy
from morpheus.agents.surprise import ERROR, NOVELTY, SurpriseRouter, divergence
from morpheus.agents.world_model import WorldModel
from morpheus.config import Config, LLMConfig, OrchestratorConfig
from morpheus.envs.mock_env import MockRetailEnv, make_mock_env
from morpheus.llm import build_llm
from morpheus.orchestrator.loop import Orchestrator
from morpheus.orchestrator.types import Action


def _orch(use_wm: bool = True) -> Orchestrator:
    llm = build_llm(LLMConfig(kind="stub"))
    cfg = OrchestratorConfig(k_candidates=4, horizon=2, max_turns=12, use_world_model=use_wm)
    return Orchestrator(Policy(llm, k=4), WorldModel(llm), cfg, SurpriseRouter())


def test_mock_env_solves_with_correct_chain():
    env = MockRetailEnv(length=3)
    assert env.reset().text
    r1 = env.step(Action("authenticate_user"))
    assert not r1.done and r1.reward > 0
    env.step(Action("lookup_order"))
    r3 = env.step(Action("check_refund_policy"))
    assert r3.done and r3.info["success"]


def test_mock_env_wrong_tool_flags_error():
    env = MockRetailEnv(length=3)
    env.reset()
    r = env.step(Action("issue_refund"))  # mauvais outil au 1er pas
    assert r.observation.tool_error and not r.done


def test_orchestrator_runs_end_to_end():
    orch = _orch(use_wm=True)
    env = make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    result = orch.run(env)
    assert result.turns >= 1
    assert len(result.trace) == result.turns
    assert env.required_turns() == 4


def test_baseline_and_worldmodel_both_run():
    for use_wm in (True, False):
        env = make_mock_env(task_index=1, seed=0, buckets=[4, 8, 12])
        result = _orch(use_wm=use_wm).run(env)
        assert result.turns >= 1


def test_divergence_bounds():
    assert divergence("", "") == 0.0
    assert divergence("a b c", "a b c") == 0.0
    assert divergence("a b c", "x y z") == 1.0


def test_surprise_router_rules():
    r = SurpriseRouter()
    assert r.route(delta=0.9, tool_error=True, score_before=0.5, score_after=0.5) == ERROR
    assert r.route(delta=0.9, tool_error=False, score_before=0.6, score_after=0.4) == ERROR
    assert r.route(delta=0.9, tool_error=False, score_before=0.4, score_after=0.6) == NOVELTY


def test_config_loads_defaults():
    cfg = Config()
    assert cfg.orchestrator.max_turns == 12
    assert cfg.eval.turn_buckets == [4, 8, 12]


def test_strip_reasoning_removes_think_and_fences():
    from morpheus.text import strip_reasoning, snap_to_whitelist

    assert strip_reasoning("<think>bla bla</think>\nACTION: x") == "ACTION: x"
    assert strip_reasoning("```json\nACTION: x\n```") == "ACTION: x"
    # snap : exact, casse, recouvrement, échec
    tools = ["lookup_order", "issue_refund"]
    assert snap_to_whitelist("lookup_order", tools) == "lookup_order"
    assert snap_to_whitelist("Lookup_Order", tools) == "lookup_order"
    assert snap_to_whitelist("order_lookup", tools) == "lookup_order"
    assert snap_to_whitelist("zzz", tools) is None


def test_parse_actions_robust_to_real_llm_output():
    from morpheus.agents.policy import _parse_actions

    tools = ["authenticate_user", "lookup_order", "issue_refund"]
    # bruit typique d'un vrai LLM : think-block, prose, mauvaise casse, outil halluciné
    raw = (
        "<think>je dois d'abord authentifier</think>\n"
        "Voici mes propositions :\n"
        "ACTION: Authenticate_User | ARGS: {}\n"
        "ACTION: lookup_order | ARGS: {\"id\": 42}\n"
        "ACTION: fly_to_moon | ARGS: {}\n"      # halluciné → écarté
    )
    actions = _parse_actions(raw, tools)
    tools_out = [a.tool for a in actions]
    assert "authenticate_user" in tools_out           # snap de la casse
    assert "lookup_order" in tools_out
    assert "fly_to_moon" not in tools_out             # hallucination filtrée


def test_check_llm_runs_with_stub():
    from morpheus.diagnostics import check_llm

    cfg = Config()  # stub + mock par défaut
    assert check_llm(cfg) == 0
