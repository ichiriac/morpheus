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


def goal_alignment_loss(z_state: torch.Tensor, z_goal: torch.Tensor,
                        progress: torch.Tensor, traj_id: torch.Tensor,
                        valid: torch.Tensor | None = None,
                        temp: float = 0.1, w_align: float = 1.0, w_nce: float = 1.0
                        ) -> tuple[torch.Tensor, dict[str, float]]:
    """Rend le latent GOAL-RELATIVE — exactement la quantité que `score_to_goal` lit :
    `cos(proj(état), proj(but))`. Deux termes complémentaires :

    - **alignement (régression)** : `cos(proj(s), proj(g)) ≈ 2·progress − 1`. Le pas terminal
      (progress=1) s'aligne sur le but (cos→+1) ; le pas initial (progress=0) s'y oppose
      (cos→−1). ⇒ signal MONOTONE et d'ÉTENDUE FRANCHE le long d'une trajectoire (corrige le
      0.0086 dégénéré : `proj` n'était entraîné que sur la prédiction).
    - **discrimination (InfoNCE)** état→but : chaque état doit reconnaître SON but parmi les
      autres buts du batch (négatifs), pondéré par `progress` (un état terminal identifie son
      but, un état initial non). Les pas d'une même trajectoire sont masqués comme négatifs ;
      les buts vides (données sans goal) sont ignorés → dégrade gracieusement vers du JEPA nu.

    `z_state`/`z_goal` = proj(E_state(état)) / proj(E_state(but)), AVANT normalisation."""
    n = z_state.shape[0]
    if valid is None:                                            # défaut : but non vide = norme > 0
        valid = (z_goal.norm(dim=-1) > 1e-6).float()
    valid = valid.float()                                        # (n,) 1 si le texte du but existe
    zs = F.normalize(z_state, dim=-1)
    zg = F.normalize(z_goal, dim=-1)

    # --- alignement goal-relative : cos own-goal vers 2·progress−1 ---
    sim_own = (zs * zg).sum(dim=-1)                               # (n,) cos(état, son but)
    target = 2.0 * progress - 1.0
    denom = valid.sum().clamp_min(1.0)
    align = (((sim_own - target) ** 2) * valid).sum() / denom

    # --- discrimination état→but (InfoNCE), pondérée par progress ---
    logits = (zs @ zg.T) / temp                                  # (n,n)
    eye = torch.eye(n, dtype=torch.bool, device=z_state.device)
    same_traj = (traj_id.unsqueeze(1) == traj_id.unsqueeze(0)) & ~eye
    empty_goal = (valid.unsqueeze(0) < 0.5) & ~eye               # colonnes de buts vides
    logits = logits.masked_fill(same_traj | empty_goal, float("-inf"))
    labels = torch.arange(n, device=z_state.device)
    ce = F.cross_entropy(logits, labels, reduction="none")       # (n,)
    w = progress * valid                                         # terminal pèse le plus
    nce = (ce * w).sum() / w.sum().clamp_min(1.0)

    total = w_align * align + w_nce * nce
    return total, {
        "g_align": float(align.detach()),
        "g_nce": float(nce.detach()),
        "g_sim_own": float((sim_own * valid).sum().detach() / denom),
        "g_total": float(total.detach()),
    }
