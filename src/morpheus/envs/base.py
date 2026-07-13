"""Interface d'environnement (style Gym / τ²-bench)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..orchestrator.types import Action, Observation, StepResult


@runtime_checkable
class Env(Protocol):
    def reset(self) -> Observation:
        """Réinitialise l'épisode et renvoie l'observation initiale."""
        ...

    def step(self, action: Action) -> StepResult:
        """Applique l'action, renvoie observation/reward/done/info."""
        ...

    def goal(self) -> str:
        """Description de l'objectif de la tâche."""
        ...

    def tool_names(self) -> list[str]:
        """Noms des outils disponibles (pour la politique)."""
        ...

    def required_turns(self) -> int:
        """Nb de tours de la solution de référence (sert au bucketing des métriques)."""
        ...

    def system_context(self) -> str | None:
        """Manuel LÉGITIME de l'agent (ex. policy du domaine τ²), injecté au VRAI pas PROPOSER
        seulement (pas dans les rollouts imaginés). None si l'env n'en a pas (mock). Optionnel :
        l'orchestrateur lit `getattr(env, "system_context", lambda: None)()`."""
        ...
