"""Modèle JEPA (torch). Prédit l'embedding de l'état résultant dans un espace latent.

Recette (Phase 2 v0, encodeur GELÉ) :
  s   = E_state(obs)              # embedding pré-entraîné, figé (numpy → tensor)
  s'  = E_state(next_obs)         # idem
  z   = proj(s)                   # projection apprise vers le latent de travail
  a   = enc_action(E_state(act))  # action encodée dans le latent
  ẑ'  = P(z, a)                   # PRÉDICTION de l'état résultant
  cible = proj(s').detach()       # cible (stop-grad, esprit JEPA)
  perte = 1 - cos(ẑ', cible)  + VICReg (anti-collapse)

`proj` est partagé entre état et cible ; le stop-grad + VICReg empêchent l'effondrement.
En passant `freeze_encoder=False` + un encodeur torch, on pourrait entraîner E_state
conjointement (vrai H-JEPA) — hors périmètre v0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(sizes: list[int], act=nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class JEPA(nn.Module):
    def __init__(self, enc_dim: int, latent_dim: int = 256, action_dim: int = 128,
                 hidden: int = 512) -> None:
        super().__init__()
        self.enc_dim = enc_dim
        self.latent_dim = latent_dim
        # projection état (partagée état / cible)
        self.proj = _mlp([enc_dim, hidden, latent_dim])
        # encodeur d'action (l'action est un texte, encodé par E_state puis projeté)
        self.enc_action = _mlp([enc_dim, hidden, action_dim])
        # prédicteur : (z, a) -> ẑ'
        self.predictor = _mlp([latent_dim + action_dim, hidden, hidden, latent_dim])

    def encode_state(self, s_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(s_emb)

    def forward(self, s_emb: torch.Tensor, a_emb: torch.Tensor,
                s_next_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.proj(s_emb)
        a = self.enc_action(a_emb)
        pred = self.predictor(torch.cat([z, a], dim=-1))
        with torch.no_grad():
            target = self.proj(s_next_emb)     # cible stop-grad (esprit JEPA)
        return pred, target, z

    @torch.no_grad()
    def predict_next(self, s_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """ŝ' latent — utilisé à l'inférence (lookahead MPC, calcul de divergence)."""
        return self.predictor(torch.cat([self.proj(s_emb), self.enc_action(a_emb)], dim=-1))
