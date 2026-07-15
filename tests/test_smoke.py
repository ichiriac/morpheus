"""Smoke tests : la boucle complète tourne avec stub + mock, sans réseau."""

from __future__ import annotations

from morpheus.agents.policy import Policy
from morpheus.agents.surprise import ERROR, NOVELTY, SurpriseRouter, SurpriseSignals, divergence
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


def _orch_conc(concurrency: int) -> Orchestrator:
    llm = build_llm(LLMConfig(kind="stub"))
    cfg = OrchestratorConfig(k_candidates=4, horizon=2, max_turns=12,
                             use_world_model=True, concurrency=concurrency)
    return Orchestrator(Policy(llm, k=4), WorldModel(llm), cfg, SurpriseRouter())


def test_parallel_rollouts_match_sequential():
    # concurrency>1 (threads) doit donner EXACTEMENT le même résultat que séquentiel
    # (stub déterministe + executor.map préserve l'ordre → départage des ex æquo identique).
    env_seq = make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    env_par = make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    r_seq = _orch_conc(1).run(env_seq)
    r_par = _orch_conc(4).run(env_par)
    assert r_seq.success == r_par.success and r_seq.turns == r_par.turns
    assert [s.chosen for s in r_seq.trace] == [s.chosen for s in r_par.trace]


def test_divergence_bounds():
    assert divergence("", "") == 0.0
    assert divergence("a b c", "a b c") == 0.0
    assert divergence("a b c", "x y z") == 1.0


def test_surprise_router_rules():
    r = SurpriseRouter()
    assert r.route(SurpriseSignals(delta=0.9, tool_error=True,
                                   score_before=0.5, score_after=0.5)) == ERROR
    assert r.route(SurpriseSignals(delta=0.9, tool_error=False,
                                   score_before=0.6, score_after=0.4)) == ERROR
    assert r.route(SurpriseSignals(delta=0.9, tool_error=False,
                                   score_before=0.4, score_after=0.6)) == NOVELTY
    # direction non sondée (pas de scores) : pas de preuve de faute ⇒ NOVELTY, comme avant
    assert r.route(SurpriseSignals(delta=0.9, tool_error=False)) == NOVELTY


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


# --------------------------------------------------------------------------- #
# Contrat politique ⇄ stub : le stub doit RELIRE le prompt que la politique écrit.
# Régression déjà vécue : la politique émettait `[ÉTAT COURANT]`, le stub lisait `[STATE]`.
# L'état lu était vide, donc UNE seule action, donc `len(candidates) > 1` faux dans loop.py,
# donc AUCUN lookahead — silencieusement, sur toute config à politique stub. Rien n'échouait :
# ces deux tests sont là pour que ça échoue.
# --------------------------------------------------------------------------- #

def test_stub_reads_state_written_by_policy():
    """La balise d'état de `Policy.build_prompt` est bien celle que le stub relit."""
    from morpheus.llm.stub import _extract
    from morpheus import prompt_tags as T
    from morpheus.orchestrator.types import Observation, State

    pol = Policy(build_llm(LLMConfig(kind="stub")), k=3)
    state = State(goal="but", observation=Observation(text="Prochaine étape attendue : lookup_order."))
    prompt = pol.build_prompt(state, ["authenticate_user", "lookup_order"])
    assert _extract(prompt, T.POLICY_STATE) == "Prochaine étape attendue : lookup_order."


def test_stub_policy_proposes_k_candidates_from_state_hint():
    """L'indice d'état étant lu, le stub propose K candidats (l'attendu d'abord) — sans quoi
    le lookahead de loop.py, gardé par `len(candidates) > 1`, ne s'exécuterait jamais."""
    from morpheus.orchestrator.types import Observation, State

    tools = ["authenticate_user", "lookup_order", "check_refund_policy"]
    pol = Policy(build_llm(LLMConfig(kind="stub")), k=3)
    state = State(goal="but", observation=Observation(text="Prochaine étape attendue : lookup_order."))
    actions = pol.propose(state, tools)
    assert len(actions) > 1, "un seul candidat ⇒ loop.py sauterait le lookahead"
    assert actions[0].tool == "lookup_order", "l'étape attendue doit être proposée en premier"


def test_stub_policy_actually_exercises_lookahead_end_to_end():
    """Intégration : avec la Policy+stub STANDARD (celle des configs), la boucle doit vraiment
    faire du MPC — >1 candidats, ŝ' prédit, divergence calculée. C'est ce qu'aucun test
    n'attrapait : les tests du chemin MPC injectaient des politiques factices multi-candidats,
    ce qui masquait le fait que la vraie Policy+stub tournait à K=1."""
    result = _orch(use_wm=True).run(make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12]))
    fired = [s for s in result.trace if len(s.candidates) > 1]
    assert fired, "aucun tour à >1 candidats : la Policy+stub n'exerce jamais le lookahead"
    for s in fired:
        assert s.predicted_state is not None      # le world-model a bien prédit
        assert 0.0 <= s.divergence <= 1.0
