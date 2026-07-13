"""World-model LATENT (Phase 2/Étape 4) : le prédicteur JEPA remplace le LLM-as-world-model.

Même contrat que `agents/world_model.py` (predict / score_to_goal / rollout / divergence) →
**drop-in** dans `orchestrator/loop.py`, qui ne connaît que l'interface. La différence est
que tout se passe dans l'espace latent appris :

  ŝ'  = P(proj(E_state(s)), enc_action(E_state(a)))     # predict_next, SANS exécuter
  score_to_goal = cos(proj(E_state(s)), proj(E_state(goal)))  ramené à [0, 1]
  divergence    = (1 - cos(ŝ', proj(E_state(obs_réelle)))) / 2   ∈ [0, 1]

L'encodeur `E_state` reste GELÉ (rechargé depuis le checkpoint). Import torch PARESSEUX :
la classe ne se construit que si `jepa_wm.enabled` — les tests par défaut (LLM/stub) ne
touchent jamais torch.

Limite v0 assumée : le lookahead latent est à **1 pas** (on ne sait pas décoder un latent
prédit en texte pour re-proposer via la politique Qwen). `horizon>1` nécessiterait une
politique en espace latent → chantier ultérieur. Le score du 1er pas prédit reste le signal
MPC utile, et la boucle fermée ré-ancre de toute façon sur l'état vrai à chaque tour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..orchestrator.types import Action, State


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class JepaWorldModel:
    """Prédicteur JEPA chargé depuis `jepa.pt`, exposé via le contrat WorldModel."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        import torch  # import paresseux (n'impacte pas les tests LLM/stub)

        from ..jepa.encoders import build_encoder
        from ..jepa.model import JEPA

        self._torch = torch
        self._device = ("cuda" if (device == "auto" and torch.cuda.is_available())
                        else ("cpu" if device == "auto" else device))

        ckpt = torch.load(str(checkpoint), map_location=self._device, weights_only=False)
        cfg = ckpt["config"]
        enc_dim = int(ckpt["enc_dim"])
        # E_state gelé : reconstruit à l'identique de l'entraînement (même kind + dims).
        self._enc = build_encoder(
            cfg.get("encoder", "hashing"),
            dim=enc_dim,
            model_name=cfg.get("encoder_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        self._model = JEPA(
            enc_dim, cfg.get("latent_dim", 256), cfg.get("action_dim", 128),
            cfg.get("hidden", 512),
        )
        self._model.load_state_dict(ckpt["model"])
        self._model.eval().to(self._device)

    # --- helpers latents ---

    def _emb(self, text: str) -> "np.ndarray":  # E_state gelé (numpy, L2-normalisé)
        return self._enc.encode([text or ""])[0]

    def _tensor(self, arr: np.ndarray):
        t = self._torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
        return t.to(self._device)

    def _proj(self, text: str) -> np.ndarray:
        """proj(E_state(text)) — état projeté dans le latent de travail."""
        with self._torch.no_grad():
            z = self._model.encode_state(self._tensor(self._emb(text)))
        return z.squeeze(0).cpu().numpy()

    def _predict_latent(self, state_text: str, action_text: str) -> np.ndarray:
        """ŝ' = P(proj(E_state(s)), enc_action(E_state(a)))."""
        with self._torch.no_grad():
            z = self._model.predict_next(
                self._tensor(self._emb(state_text)),
                self._tensor(self._emb(action_text)),
            )
        return z.squeeze(0).cpu().numpy()

    # --- contrat WorldModel ---

    def predict(self, state: State, action: Action) -> list[float]:
        """ŝ' latent (liste JSON-sérialisable pour la trace)."""
        return self._predict_latent(state.text, str(action)).tolist()

    def score_to_goal(self, goal: str, state_text: str) -> float:
        """Proximité au but ∈ [0, 1] : cos latent (proj état, proj but) ramené à [0, 1].

        VALIDÉ goal-relative quand `proj` est entraîné avec le terme d'alignement but↔état
        (`jepa/losses.py::goal_alignment_loss`) sur la distribution cible. Gate
        `scripts/validate_goal_signal.py` sur held-out τ²-retail (checkpoint
        `jepa_tau2_align`) : **H1 PASS** (monotonie, rho +0.635, p≈0) + **H2 PASS** (séparation
        succès/échec par NIVEAU, length-robust, p≈0). ⚠️ La validité est **par distribution** :
        un checkpoint entraîné hors-domaine (ex. APIGen nu) reste un PROXY sur τ² (cf. mémoire
        `goal-signal-distribution-mismatch`). Réentraîner l'alignement en-domaine avant de s'y fier."""
        return max(0.0, min(1.0, (_cos(self._proj(state_text), self._proj(goal)) + 1.0) / 2.0))

    def divergence(self, predicted, real_text: str) -> float:
        """δ ∈ [0, 1] : (1 - cos(ŝ', proj(E_state(obs_réelle)))) / 2. `predicted` = ŝ' latent."""
        z_pred = np.asarray(predicted, dtype=np.float32)
        target = self._proj(real_text)
        return max(0.0, min(1.0, (1.0 - _cos(z_pred, target)) / 2.0))

    def rollout(self, policy, state: State, first: Action, tools: list[str],
                horizon: int) -> tuple[float, list[float]]:
        """Lookahead latent à 1 pas : prédit ŝ' de `first`, score sa proximité au but latent.
        Renvoie `(proximité, ŝ')` ; ŝ' est réutilisé par la boucle pour δ (pas de re-predict).
        `horizon` est accepté pour respecter le contrat mais ignoré au-delà de 1 (cf. module)."""
        z_goal = self._proj(state.goal)
        z_pred = self._predict_latent(state.text, str(first))
        score = max(0.0, min(1.0, (_cos(z_pred, z_goal) + 1.0) / 2.0))
        return score, z_pred.tolist()
