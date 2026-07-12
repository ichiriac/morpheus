"""Pertes JEPA (torch) : prédiction + VICReg (anti-collapse)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosinus moyen entre ẑ' prédit et la cible (déjà stop-grad dans le modèle)."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def vicreg(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Régularisateurs VICReg sur un batch de latents (n, d) :
    - variance : pousse chaque dimension à garder de la variance (>= gamma) → anti-collapse ;
    - covariance : décorrèle les dimensions.
    Retourne (perte_variance, perte_covariance)."""
    n, d = z.shape
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


def jepa_loss(pred: torch.Tensor, target: torch.Tensor, z: torch.Tensor,
              w_pred: float = 1.0, w_var: float = 1.0, w_cov: float = 0.04
              ) -> tuple[torch.Tensor, dict[str, float]]:
    p = prediction_loss(pred, target)
    var_loss, cov_loss = vicreg(torch.cat([z, pred], dim=0))
    total = w_pred * p + w_var * var_loss + w_cov * cov_loss
    return total, {
        "pred": float(p.detach()),
        "var": float(var_loss.detach()),
        "cov": float(cov_loss.detach()),
        "total": float(total.detach()),
    }
