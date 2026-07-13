"""JepaWorldModel (Étape 4) : world-model latent branché sur jepa.pt.

Guardé par `importorskip("torch")` → skip si torch absent (comme la partie torch de
test_jepa). Vérifie le contrat WorldModel (predict/score_to_goal/divergence/rollout) et le
fait qu'il est **drop-in** dans l'orchestrateur, sans rien changer à la boucle.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from morpheus.agents.policy import Policy
from morpheus.agents.surprise import SurpriseRouter
from morpheus.config import Config, LLMConfig, OrchestratorConfig
from morpheus.envs.mock_env import make_mock_env
from morpheus.llm import build_llm
from morpheus.orchestrator.loop import Orchestrator

LATENT_DIM = 32


@pytest.fixture(scope="module")
def jepa_ckpt(tmp_path_factory):
    """Entraîne un mini-JEPA (hashing, 2 epochs) et renvoie le chemin de jepa.pt."""
    pytest.importorskip("torch")
    from morpheus.jepa.train import JepaConfig, train

    out = tmp_path_factory.mktemp("jepa_wm")
    cfg = JepaConfig(source="synthetic", n_episodes=40, encoder="hashing", enc_dim=64,
                     latent_dim=LATENT_DIM, action_dim=16, hidden=64, epochs=2,
                     batch_size=32, device="cpu", out_dir=str(out))
    train(cfg)
    return str(out / "jepa.pt")


def _wm(jepa_ckpt):
    from morpheus.agents.jepa_world_model import JepaWorldModel

    return JepaWorldModel(jepa_ckpt, device="cpu")


def test_jepa_wm_contract(jepa_ckpt):
    from morpheus.orchestrator.types import Action, Observation, State

    wm = _wm(jepa_ckpt)
    state = State(goal="issue a refund", observation=Observation(text="user authenticated"))

    pred = wm.predict(state, Action("lookup_order"))
    assert isinstance(pred, list) and len(pred) == LATENT_DIM   # ŝ' latent, JSON-sérialisable

    score = wm.score_to_goal(state.goal, state.text)
    assert 0.0 <= score <= 1.0

    delta = wm.divergence(pred, "order #42 found, status pending")
    assert 0.0 <= delta <= 1.0

    best, first_pred = wm.rollout(Policy(build_llm(LLMConfig(kind="stub")), k=4),
                                  state, Action("lookup_order"), ["lookup_order"], horizon=3)
    assert 0.0 <= best <= 1.0 and len(first_pred) == LATENT_DIM


def test_jepa_wm_is_drop_in_orchestrator(jepa_ckpt):
    """La boucle tourne à l'identique avec le WM latent ; la trace reste JSON-sérialisable."""
    llm = build_llm(LLMConfig(kind="stub"))
    cfg = OrchestratorConfig(k_candidates=4, horizon=2, max_turns=8, use_world_model=True,
                             surprise_threshold=0.0)
    orch = Orchestrator(Policy(llm, k=4), _wm(jepa_ckpt), cfg, SurpriseRouter())
    result = orch.run(make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12]))
    assert result.turns >= 1 and len(result.trace) == result.turns
    # predicted_state est un latent (liste) : doit sérialiser sans erreur
    json.dumps([asdict(s) for s in result.trace])


def test_jepa_wm_disabled_by_default():
    """Intégration OPTIONNELLE : off par défaut → le runner garde le LLM WM, torch non requis."""
    assert Config().jepa_wm.enabled is False
