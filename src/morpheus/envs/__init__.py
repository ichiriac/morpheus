"""Environnements. `mock` tourne sans dépendance ; `tau2` branche le vrai benchmark."""

from __future__ import annotations

from ..config import EvalConfig
from .base import Env


def build_env_factory(cfg: EvalConfig):
    """Renvoie `(make, n_tasks)` : une fabrique `make(task_index) -> Env` et le nombre
    EFFECTIF de tâches à jouer (τ² peut en offrir moins que `cfg.tasks`)."""
    if cfg.env == "mock":
        from .mock_env import make_mock_env

        make = lambda i: make_mock_env(task_index=i, seed=cfg.seed, buckets=cfg.turn_buckets,
                                       reveal_next=not cfg.mock_hard)
        return make, cfg.tasks
    if cfg.env == "tau2":
        from .tau2_adapter import build_tau2_factory

        return build_tau2_factory(cfg)
    raise ValueError(f"env inconnu : {cfg.env!r}")


__all__ = ["Env", "build_env_factory"]
