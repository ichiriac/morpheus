"""Tests d'intégration de l'adaptateur τ²-bench + du bucketing réussite-vs-tours.

Les tests τ² sont ignorés (skip) si le paquet `tau2` n'est pas installé, pour garder la
suite verte hors du pod GPU. Le test de bucketing, lui, ne dépend pas de τ².
"""

from __future__ import annotations

import pytest

from morpheus.eval.metrics import SuccessVsTurns
from morpheus.envs.tau2_adapter import RESPOND_TOOL, _build_goal, _extract_text
from morpheus.orchestrator.types import Action


class _FakeScenario:
    def __str__(self) -> str:  # simule str(UserScenario) : contient le brief PRIVÉ
        return "reason_for_call: SECRET · unknown_info: à découvrir · task_instructions: ..."


class _FakeTask:
    ticket = None
    user_scenario = _FakeScenario()


def test_nonsolo_goal_does_not_leak_user_scenario():
    """Non-solo : le but ne doit contenir AUCUNE info privée du scénario utilisateur."""
    g = _build_goal(_FakeTask(), solo=False, domain="retail")
    assert "SECRET" not in g and "unknown_info" not in g and "task_instructions" not in g
    assert "retail" in g and "respond_to_user" in g   # instruction générique attendue


def test_solo_goal_uses_ticket():
    class _T:
        ticket = "Faire X puis Y"
        user_scenario = _FakeScenario()
    assert _build_goal(_T(), solo=True, domain="telecom") == "Faire X puis Y"


def test_extract_text_reads_common_keys():
    assert _extract_text(Action(tool=RESPOND_TOOL, args={"text": "Quel est l'id ?"})) == "Quel est l'id ?"
    assert _extract_text(Action(tool=RESPOND_TOOL, args={"message": "Bonjour"})) == "Bonjour"


def test_extract_text_fallbacks_and_never_empty():
    # pas de clé texte → rationale, puis valeurs d'args, sinon message par défaut non vide
    assert _extract_text(Action(tool=RESPOND_TOOL, args={}, rationale="Demander l'email")) == "Demander l'email"
    assert _extract_text(Action(tool=RESPOND_TOOL, args={"foo": "bar"})) == "bar"
    assert _extract_text(Action(tool=RESPOND_TOOL, args={})).strip()  # jamais vide


def test_extract_text_disguises_toolcall_shaped_content():
    # un texte en forme d'appel d'outil ne doit pas être re-parsé comme tool call par τ²
    out = _extract_text(Action(tool=RESPOND_TOOL, args={"text": "cancel_order(id=7)"}))
    assert out.startswith("Message :")


def test_success_vs_turns_buckets_by_required_turns():
    """La métrique agrège par `required_turns` (le bucket = longueur de tâche τ²)."""
    m = SuccessVsTurns()
    for rt, ok in [(4, True), (4, False), (8, True), (8, True), (12, False)]:
        m.add(rt, ok)
    curve = {b: (rate, n) for b, rate, n in m.curve()}
    assert curve[4] == (0.5, 2)
    assert curve[8] == (1.0, 2)
    assert curve[12] == (0.0, 1)
    assert [b for b, _, _ in m.curve()] == [4, 8, 12]  # trié par longueur croissante
    assert m.overall == pytest.approx(3 / 5)


# --- Intégration τ² (skip si non installé) ---
tau2 = pytest.importorskip("tau2", reason="τ²-bench non installé (hors pod GPU)")
pytest.importorskip("gymnasium", reason="gymnasium requis par tau2.gym")

from morpheus.config import EvalConfig  # noqa: E402
from morpheus.envs import build_env_factory  # noqa: E402
from morpheus.orchestrator.types import Action  # noqa: E402


def test_tau2_solo_factory_env_contract():
    """Solo telecom : la fabrique câble reset/step/goal/tool_names/required_turns."""
    cfg = EvalConfig(env="tau2", domain="telecom", tau2_solo=True, tasks=2, tau2_max_steps=8)
    make, n = build_env_factory(cfg)
    assert n == 2

    env = make(0)
    obs = env.reset()
    assert isinstance(obs.text, str) and obs.text          # observation initiale non vide
    assert env.goal()                                       # ticket présent (tâche solo)
    tools = env.tool_names()
    assert "done" in tools                                  # outil d'arrêt exposé à la politique
    assert RESPOND_TOOL not in tools                        # pas de user en solo → pas d'outil réponse
    assert isinstance(env.required_turns(), int)

    res = env.step(Action(tool="done"))                     # done → épisode terminé
    assert res.done is True
    assert "success" in res.info
    env.close()


def test_tau2_solo_rejects_ticketless_domain():
    """Retail n'a pas de tickets → solo doit échouer explicitement (pas silencieusement)."""
    cfg = EvalConfig(env="tau2", domain="retail", tau2_solo=True, tasks=2)
    with pytest.raises(ValueError):
        build_env_factory(cfg)


def test_tau2_nonsolo_exposes_respond_to_user():
    """Non-solo (retail) : l'outil respond_to_user est exposé à la politique.
    Vérifié sans reset() → pas besoin du user-sim LLM."""
    cfg = EvalConfig(env="tau2", domain="retail", tau2_solo=False, tasks=1)
    make, _n = build_env_factory(cfg)
    env = make(0)                                   # construit sans démarrer l'orchestrateur
    assert RESPOND_TOOL in env.tool_names()
