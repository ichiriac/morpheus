"""Environnements. `mock` tourne sans dépendance ; `tau2` branche le vrai benchmark."""

from __future__ import annotations

from ..config import EvalConfig
from .base import Env


def build_env_factory(cfg: EvalConfig):
    """Renvoie une fonction `make(task_index) -> Env` pour itérer sur les tâches."""
    if cfg.env == "mock":
        from .mock_env import make_mock_env

        return lambda i: make_mock_env(task_index=i, seed=cfg.seed, buckets=cfg.turn_buckets)
    if cfg.env == "tau2":
        from .tau2_adapter import make_tau2_env

        return lambda i: make_tau2_env(task_index=i, domain=cfg.domain)
    raise ValueError(f"env inconnu : {cfg.env!r}")


__all__ = ["Env", "build_env_factory"]
