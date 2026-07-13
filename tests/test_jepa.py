"""Tests JEPA.

Partie SANS torch (data + encoder) : tourne partout, y compris ici sans GPU.
Partie torch (model/losses/train) : guardée par importorskip → skip si torch absent,
s'exécute sur le pod RunPod.
"""

from __future__ import annotations

import numpy as np
import pytest

from morpheus.jepa.data import (
    Transition, from_alfworld_steps, from_messages, synthetic_transitions,
)
from morpheus.jepa.encoders import HashingEncoder, build_encoder


# --------------------------- data (torch-free) --------------------------- #

def test_synthetic_transitions_are_valid_and_chained():
    trans = synthetic_transitions(n_episodes=10, seed=0)
    assert len(trans) > 10
    assert all(t.is_valid() for t in trans)
    # au moins une transition terminale
    assert any(t.done for t in trans)


def test_synthetic_transitions_carry_goal_progress():
    """Signal goal-relative : chaque pas porte un but, un progress ∈]0,1] croissant, un traj_id."""
    trans = synthetic_transitions(n_episodes=5, seed=0)
    by_traj: dict[int, list] = {}
    for t in trans:
        assert t.goal.strip()                       # but non vide
        assert 0.0 < t.progress <= 1.0
        by_traj.setdefault(t.traj_id, []).append(t)
    for tid, steps in by_traj.items():
        progs = [s.progress for s in steps]
        assert progs == sorted(progs)               # progress croissant dans la trajectoire
        assert progs[-1] == pytest.approx(1.0)      # état résolu = progress 1.0
        assert steps[-1].done and {s.goal for s in steps} == {steps[0].goal}


def test_from_messages_sets_goal_and_progress():
    msgs = [
        {"role": "user", "content": "rembourse ma commande 42"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "lookup_order", "arguments": "{}"}}]},
        {"role": "tool", "content": "commande trouvée"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "issue_refund", "arguments": "{}"}}]},
        {"role": "tool", "content": "remboursement émis"},
    ]
    trans = from_messages(msgs, traj_id=7)
    assert [t.goal for t in trans] == ["rembourse ma commande 42"] * 2
    assert [t.traj_id for t in trans] == [7, 7]
    assert trans[0].progress == pytest.approx(0.5) and trans[1].progress == pytest.approx(1.0)
    assert trans[-1].done


def test_from_messages_openai_toolcall_shape():
    msgs = [
        {"role": "system", "content": "tu es un agent"},
        {"role": "user", "content": "rembourse ma commande 42"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "lookup_order",
                                      "arguments": "{\"id\": 42}"}}]},
        {"role": "tool", "content": "commande 42 : payée, éligible remboursement"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "issue_refund", "arguments": "{\"id\": 42}"}}]},
        {"role": "tool", "content": "remboursement émis"},
    ]
    trans = from_messages(msgs)
    assert len(trans) == 2
    assert "lookup_order" in trans[0].action
    assert "remboursement" in trans[1].next_obs
    # l'état de la 1re transition contient le contexte utilisateur
    assert "rembourse" in trans[0].obs


def test_from_messages_sharegpt_shape():
    conv = [
        {"from": "human", "value": "quel temps fait-il ?"},
        {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {}}"},
        {"from": "observation", "value": "22°C ensoleillé"},
    ]
    trans = from_messages(conv)
    assert len(trans) == 1
    assert "get_weather" in trans[0].action
    assert "22" in trans[0].next_obs


def test_from_alfworld_steps():
    steps = [
        {"observation": "tu es dans la cuisine", "action": "go to fridge"},
        {"observation": "tu es devant le frigo", "action": "open fridge"},
        {"observation": "le frigo est ouvert", "action": "take milk"},
    ]
    trans = from_alfworld_steps(steps)
    assert len(trans) == 2
    assert trans[0].obs.startswith("tu es dans la cuisine")
    assert trans[0].next_obs.startswith("tu es devant le frigo")


# --------------------------- encoder (numpy) ----------------------------- #

def test_hashing_encoder_deterministic_and_normalized():
    enc = HashingEncoder(dim=64, seed=0)
    a = enc.encode(["issue_refund pour la commande 42"])
    b = enc.encode(["issue_refund pour la commande 42"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)                     # déterministe
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-5)  # L2-normalisé
    # textes différents → vecteurs différents
    c = enc.encode(["texte totalement autre"])
    assert not np.allclose(a, c)


def test_build_encoder_factory():
    enc = build_encoder("hashing", dim=32)
    assert enc.encode(["x"]).shape == (1, 32)


# --------------------------- torch (RunPod) ------------------------------ #

def test_jepa_forward_and_loss_step():
    torch = pytest.importorskip("torch")
    from morpheus.jepa.losses import jepa_loss
    from morpheus.jepa.model import JEPA

    enc_dim, n = 64, 128
    model = JEPA(enc_dim=enc_dim, latent_dim=48, action_dim=24, hidden=64)
    s = torch.randn(n, enc_dim)
    a = torch.randn(n, enc_dim)
    sn = torch.randn(n, enc_dim)
    pred, target, z = model(s, a, sn)
    assert pred.shape == (n, 48) and target.shape == (n, 48)
    loss, logs = jepa_loss(pred, target, z)
    loss.backward()
    assert "pred" in logs and np.isfinite(logs["total"])


def test_goal_alignment_loss_rewards_progress_alignment():
    """cos(proj(s),proj(g))≈2·progress−1 : latents alignés sur le progress ⇒ perte ~0 ;
    latents anti-alignés ⇒ perte franchement > 0. Le mask `valid` neutralise les buts vides."""
    torch = pytest.importorskip("torch")
    from morpheus.jepa.losses import goal_alignment_loss

    import torch.nn.functional as F

    n, d = 8, 16
    torch.manual_seed(0)
    progress = torch.linspace(0.1, 1.0, n)
    traj_id = torch.arange(n)                       # trajectoires distinctes → tous négatifs
    z_goal = F.normalize(torch.randn(n, d), dim=-1)
    # composante orthogonale au but (Gram-Schmidt) pour fabriquer un cos EXACT = 2·progress−1
    perp = torch.randn(n, d)
    perp = F.normalize(perp - (perp * z_goal).sum(-1, keepdim=True) * z_goal, dim=-1)
    c = (2 * progress - 1).unsqueeze(1)             # cible de cos
    z_aligned = c * z_goal + torch.sqrt((1 - c ** 2).clamp_min(0)) * perp  # cos(.,z_goal)=2p−1
    good, glog = goal_alignment_loss(z_aligned, z_goal, progress, traj_id, temp=0.1)
    bad, _ = goal_alignment_loss(-z_aligned, z_goal, progress, traj_id, temp=0.1)
    assert glog["g_align"] < 0.05 and good < bad

    # buts tous vides (valid=0) → perte nulle, pas de NaN (dégrade vers JEPA nu)
    valid0 = torch.zeros(n)
    empty, elog = goal_alignment_loss(z_aligned, z_goal, progress, traj_id, valid=valid0)
    assert float(empty) == pytest.approx(0.0) and np.isfinite(elog["g_align"])


def test_jepa_train_smoke_synthetic():
    pytest.importorskip("torch")
    from morpheus.jepa.train import JepaConfig, train

    cfg = JepaConfig(source="synthetic", n_episodes=30, encoder="hashing", enc_dim=64,
                     latent_dim=48, action_dim=24, hidden=64, epochs=2, batch_size=32,
                     device="cpu", out_dir="runs/jepa_smoke")
    stats = train(cfg)
    assert stats["n_transitions"] > 0
    assert np.isfinite(stats["best_val_pred"])
