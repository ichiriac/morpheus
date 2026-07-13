"""Tests d'intégration de l'adaptateur τ²-bench + du bucketing réussite-vs-tours.

Les tests τ² sont ignorés (skip) si le paquet `tau2` n'est pas installé, pour garder la
suite verte hors du pod GPU. Le test de bucketing, lui, ne dépend pas de τ².
"""

from __future__ import annotations

import pytest

from morpheus.eval.metrics import SuccessVsTurns


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
