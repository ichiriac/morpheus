"""Routeur appris (Phase 4) : recompute hors-ligne des signaux, logreg numpy, e2e."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from morpheus.agents.surprise import ERROR, NOVELTY, Router, SurpriseRouter, SurpriseSignals
from morpheus.router.features import signals_for_episode
from morpheus.router.model import RouterModel
from morpheus.router.train import (confusion, cross_validate, fit_logreg, standardize,
                                   stratified_folds, train_router)


def _step(turn: int, chosen: str, real: str, divergence: float = 0.5, **extra) -> dict:
    return {"turn": turn, "chosen": chosen, "real_state": real,
            "divergence": divergence, **extra}


# --------------------------------------------------------------------------- #
# features : recompute fidèle à loop.py
# --------------------------------------------------------------------------- #

def test_signals_recompute_from_old_trace():
    steps = [
        _step(1, "lookup_order(order_id='W1')", "tool: Error: Order not found", 0.8),
        _step(2, "lookup_order(order_id='W1')", "order W1 found status delivered", 0.5),
        _step(3, "respond_to_user(text='ok')", "user: please refund order W1", 0.4),
    ]
    sigs = signals_for_episode(steps)
    # signature de l'outil : même détecteur que l'adaptateur τ²
    assert sigs[0].tool_error is True and sigs[1].tool_error is False
    # rubrique loop_no_progress : répéter compte SEULEMENT si le pas précédent n'a pas erré
    assert sigs[0].repeated_tool is False               # pas de pas précédent
    assert sigs[1].repeated_tool is False               # le pas 1 AVAIT erré → retry légitime
    assert sigs[2].is_user_turn is True
    # localité : l'ouverture n'est pas journalisée → 0.0 au pas 1 ; ensuite du déjà-vu
    assert sigs[0].familiarity == 0.0
    assert sigs[2].familiarity > 0.0                    # "order W1" déjà observé
    # mémoire épisodique rejouée : vide au pas 1, puis les faits des pas passés remontent
    assert sigs[0].memory_hits == 0
    assert sigs[1].memory_hits >= 1
    # scores non sondés sans checkpoint : None (pas 0)
    assert sigs[0].score_before is None and sigs[0].score_after is None


def test_signals_repeated_tool_after_success():
    steps = [
        _step(1, "lookup_order(order_id='W1')", "order W1 found"),
        _step(2, "lookup_order(order_id='W1')", "order W1 found again"),
    ]
    sigs = signals_for_episode(steps, use_memory=False)
    assert sigs[1].repeated_tool is True                # même outil, le pas 1 avait RÉUSSI


def test_repeated_dialogue_is_not_a_loop():
    # revue 2026-07-14 : des respond_to_user consécutifs = conversation normale, PAS une boucle
    steps = [
        _step(1, "respond_to_user(text='votre nom ?')", "user: Mei Kovacs"),
        _step(2, "respond_to_user(text='votre zip ?')", "user: 28236"),
    ]
    sigs = signals_for_episode(steps, use_memory=False)
    assert sigs[1].repeated_tool is False               # dialogue exclu du signal boucle
    assert sigs[1].is_user_turn is True


def test_recorded_signals_take_precedence():
    steps = [_step(1, "toolA()", "résultat", 0.6,
                   signals={"score_before": 0.2, "score_after": 0.9, "kb_top_score": 3.3,
                            "kb_hits": 2, "memory_hits": 1, "reducibility": 0.7})]
    sig = signals_for_episode(steps)[0]
    assert sig.score_after == 0.9 and sig.score_before == 0.2
    assert sig.kb_top_score == 3.3 and sig.kb_hits == 2
    assert sig.reducibility == 0.7                      # non rejouable, repris de la trace


def test_scores_argument_fills_before_after():
    steps = [_step(1, "a()", "x"), _step(2, "b()", "y")]
    sigs = signals_for_episode(steps, use_memory=False, scores=[0.3, 0.6])
    assert sigs[0].score_before is None                 # ouverture non journalisée
    assert sigs[0].score_after == 0.3
    assert sigs[1].score_before == 0.3 and sigs[1].score_after == 0.6


# --------------------------------------------------------------------------- #
# modèle + entraînement (numpy pur)
# --------------------------------------------------------------------------- #

def test_logreg_learns_separable_toy():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2.0, 0.4, (30, 3)), rng.normal(2.0, 0.4, (30, 3))])
    y = np.concatenate([np.zeros(30), np.ones(30)])
    mu, sigma = standardize(X)
    w, b = fit_logreg((X - mu) / sigma, y, epochs=500, lr=0.5, l2=1e-3)
    from morpheus.router.model import sigmoid
    pred = (sigmoid(((X - mu) / sigma) @ w + b) >= 0.5).astype(float)
    assert confusion(y, pred)["balanced_accuracy"] == 1.0


def test_stratified_folds_keep_both_classes():
    y = np.array([1.0] * 6 + [0.0] * 14)
    for fold in stratified_folds(y, k=3, seed=0):
        assert (y[fold] == 1.0).any() and (y[fold] == 0.0).any()


def test_cross_validate_shrinks_folds_to_rare_class():
    y = np.array([1.0] * 2 + [0.0] * 10)                # 2 ERREUR seulement
    X = np.asarray([[float(v)] for v in y])             # feature = le label (séparable)
    preds, k = cross_validate(X, y, folds=5, epochs=200, lr=0.5, l2=1e-3, seed=0)
    assert k == 2                                       # clampé à l'effectif de la classe rare
    assert confusion(y, preds)["balanced_accuracy"] == 1.0


def test_router_model_roundtrip_and_route(tmp_path):
    # ERREUR ⟺ tool_error : le modèle doit l'apprendre et router comme la règle
    sigs = [SurpriseSignals(delta=0.9, tool_error=(i % 2 == 0)) for i in range(20)]
    X = np.asarray([s.as_vector() for s in sigs])
    y = np.asarray([1.0 if s.tool_error else 0.0 for s in sigs])
    mu, sigma = standardize(X)
    w, b = fit_logreg((X - mu) / sigma, y, epochs=500, lr=0.5, l2=1e-3)
    model = RouterModel(feature_names=SurpriseSignals.VECTOR_FIELDS, mu=mu, sigma=sigma, w=w, b=b)
    assert model.route(SurpriseSignals(delta=0.9, tool_error=True)) == ERROR
    assert model.route(SurpriseSignals(delta=0.9, tool_error=False)) == NOVELTY
    path = tmp_path / "router.json"
    model.save(path)
    loaded = RouterModel.load(path)
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert loaded.feature_names == tuple(SurpriseSignals.VECTOR_FIELDS)


def test_train_router_end_to_end(tmp_path):
    # mini-dataset séparable : label = ERROR ssi tool_error (12 pas, 2 classes)
    rows = []
    for i in range(12):
        err = i % 3 == 0
        sig = SurpriseSignals(delta=0.8 if err else 0.3, tool_error=err)
        rows.append({"episode": f"run#{i}", "turn": 1,
                     "label": "ERROR" if err else "NOVELTY", "rationale": "test",
                     "chosen": "toolA()", "signals": sig.to_dict(),
                     "vector": sig.as_vector(),
                     "vector_fields": list(SurpriseSignals.VECTOR_FIELDS)})
    ds = tmp_path / "dataset.jsonl"
    ds.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert train_router(str(ds), str(tmp_path / "out"), folds=4, epochs=300, seed=0) == 0
    saved = RouterModel.load(tmp_path / "out" / "router.json")
    assert saved.meta["n"] == 12 and len(saved.w) == len(SurpriseSignals.VECTOR_FIELDS)


# --------------------------------------------------------------------------- #
# branchement dans la boucle (Phase 4) : flag + garde de régime
# --------------------------------------------------------------------------- #

def _fitted(sigs, labels, **meta):
    X = np.asarray([s.as_vector() for s in sigs])
    y = np.asarray([1.0 if e else 0.0 for e in labels])
    mu, sigma = standardize(X)
    w, b = fit_logreg((X - mu) / sigma, y, epochs=500, lr=0.5, l2=1e-3)
    return RouterModel(feature_names=SurpriseSignals.VECTOR_FIELDS, mu=mu, sigma=sigma,
                       w=w, b=b, meta=meta)


def test_build_router_defaults_to_phase1_rule():
    from morpheus.config import OrchestratorConfig
    from morpheus.eval.runner import build_router

    router = build_router(OrchestratorConfig())          # pas de router_checkpoint
    assert isinstance(router, SurpriseRouter)


def test_build_router_loads_learned_and_honours_contract(tmp_path):
    from morpheus.config import OrchestratorConfig
    from morpheus.eval.runner import build_router

    # entraîné SANS KB ni mémoire → régime = celui des défauts de OrchestratorConfig
    sigs = [SurpriseSignals(delta=0.9, tool_error=(i % 2 == 0)) for i in range(20)]
    path = tmp_path / "router.json"
    _fitted(sigs, [s.tool_error for s in sigs]).save(path)

    cfg = OrchestratorConfig(router_checkpoint=str(path))
    router = build_router(cfg)
    assert isinstance(router, RouterModel)
    assert isinstance(router, Router)                    # contrat de la boucle rempli
    assert router.route(SurpriseSignals(delta=0.9, tool_error=True)) == ERROR


def test_build_router_skips_regime_guard_when_no_world_model(tmp_path):
    """Baseline nue (--no-world-model) : route() n'est jamais appelé (predicted=None). Charger
    un checkpoint hors-régime ne doit PAS échouer — sinon la mesure baseline-vs-JEPA-WM, qui
    passe le MÊME YAML aux deux runs, casserait sur le run baseline pour un routeur inutilisé."""
    from morpheus.config import OrchestratorConfig
    from morpheus.eval.runner import build_router

    # checkpoint volontairement hors-régime (dérive massive) — rejeté SI world-model actif
    sigs = [SurpriseSignals(delta=0.5, tool_error=False, kb_top_score=(30.0 if i % 2 else 1.0),
                            kb_hits=3) for i in range(20)]
    path = tmp_path / "router.json"
    _fitted(sigs, [i % 2 == 1 for i in range(20)]).save(path)

    with pytest.raises(ValueError):                      # world-model on ⇒ garde active
        build_router(OrchestratorConfig(router_checkpoint=str(path),
                                        use_rag=False, use_world_model=True))
    r = build_router(OrchestratorConfig(router_checkpoint=str(path),
                                        use_rag=False, use_world_model=False))
    assert isinstance(r, RouterModel)                    # baseline : chargé, inerte, pas de garde


def test_probe_rate_reads_training_regime_from_checkpoint():
    # sondé sur 100 % des exemples ⇒ mu de l'indicateur = 1.0 ; jamais sondé ⇒ 0.0
    sigs = [SurpriseSignals(delta=0.5, tool_error=(i % 2 == 0), kb_top_score=3.0, kb_hits=2)
            for i in range(10)]
    m = _fitted(sigs, [s.tool_error for s in sigs])
    assert m.probe_rate("kb_probed") == 1.0
    assert m.probe_rate("reducibility_probed") == 0.0


def test_regime_drift_rejects_run_when_gated_signal_not_probed_live(tmp_path):
    """Un signal sondé au train mais pas en live est épinglé à 0 par as_vector() : le modèle
    lit une valeur hors-distribution ⇒ dérive constante. Le runner doit refuser AVANT le run."""
    from morpheus.config import OrchestratorConfig
    from morpheus.eval.runner import build_router

    # kb_top_score porte le signal ET est sondé partout au train
    sigs = [SurpriseSignals(delta=0.5, tool_error=False, kb_top_score=(20.0 if i % 2 else 1.0),
                            kb_hits=3) for i in range(20)]
    m = _fitted(sigs, [i % 2 == 1 for i in range(20)])
    drift, detail = m.regime_drift(use_rag=False, use_memory=False, use_world_model=False)
    assert abs(drift) >= 1.0 and "kb_top_score" in detail

    path = tmp_path / "router.json"
    m.save(path)
    cfg = OrchestratorConfig(router_checkpoint=str(path), use_rag=False)
    with pytest.raises(ValueError, match="régime de sondage"):
        build_router(cfg)

    # régime rétabli (use_rag=True) ⇒ plus de dérive, le run passe
    assert m.regime_drift(use_rag=True, use_memory=False, use_world_model=False)[0] == 0.0
    assert isinstance(build_router(OrchestratorConfig(router_checkpoint=str(path),
                                                      use_rag=True)), RouterModel)


def test_signal_probed_live_but_never_at_train_is_inert_and_reported():
    """direction : disponible dans la boucle (world-model on), jamais sondée au train ⇒
    colonne constante, gradient nul, poids 0. Aucune dérive — mais le routeur est AVEUGLE."""
    sigs = [SurpriseSignals(delta=0.5, tool_error=(i % 2 == 0)) for i in range(10)]
    m = _fitted(sigs, [s.tool_error for s in sigs])
    i = m.feature_names.index("direction")
    assert m.w[i] == 0.0                                 # jamais appris
    assert m.regime_drift(use_world_model=True)[0] == 0.0        # inerte
    assert "direction" in m.unused_live_signals(use_world_model=True)


def test_shipped_checkpoint_matches_its_training_regime():
    """Garde-fou sur les artefacts VERSIONNÉS : le checkpoint livré a été entraîné avec KB,
    mémoire ET direction sondées (dataset direction-sondé, build_router_dataset.py --checkpoint).
    Les configs τ² laissent use_rag/use_memory à False ⇒ ce couple DOIT être rejeté, pas subi
    silencieusement. Verrouille aussi la reproductibilité checkpoint↔dataset (même régime)."""
    ckpt = Path("checkpoints/router/router.json")
    if not ckpt.exists():
        pytest.skip("checkpoint du routeur non présent")
    m = RouterModel.load(ckpt)
    assert m.probe_rate("kb_probed") == 1.0 and m.probe_rate("memory_probed") == 1.0
    assert m.probe_rate("direction_probed") > 0.5        # entraîné AVEC direction (99/108)

    # régime des configs τ² (use_rag/use_memory False) ⇒ dérive décisive ⇒ run refusé
    off, _ = m.regime_drift(use_rag=False, use_memory=False, use_world_model=True)
    assert off < -1.0                                    # dérive vers NOUVEAUTÉ, décisive
    # régime d'entraînement retrouvé ⇒ aucune dérive, aucun signal aveugle
    on, _ = m.regime_drift(use_rag=True, use_memory=True, use_world_model=True)
    assert on == 0.0
    assert m.unused_live_signals(use_rag=True, use_memory=True, use_world_model=True) == []
