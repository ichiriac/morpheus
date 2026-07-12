"""Types du domaine partagés par l'env, la politique, le world-model et la boucle.

Le champ `latent` de `State`/`Observation` est un placeholder : en Phase 1 il reste None
(le world-model raisonne en texte). En Phase 2+, `E_state(obs)` le remplira avec le vecteur
JEPA, sans changer le reste de l'interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """Un appel d'outil envisagé ou exécuté."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __str__(self) -> str:
        return f"{self.tool}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


@dataclass
class Observation:
    """Ce que l'environnement renvoie après une action (ou au reset)."""

    text: str
    tool_error: bool = False          # l'outil a-t-il renvoyé une erreur explicite ?
    latent: Any = None                # rempli par E_state en Phase 2+ (JEPA)


@dataclass
class State:
    """État courant de l'épisode, tel que vu par l'orchestrateur."""

    goal: str
    observation: Observation
    turn: int = 0
    history: list[str] = field(default_factory=list)
    latent: Any = None                # E_state(observation) en Phase 2+

    @property
    def text(self) -> str:
        return self.observation.text


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """Une entrée de trace par tour — matière première des métriques et de l'analyse."""

    turn: int
    candidates: list[str]
    chosen: str
    predicted_state: str | None
    real_state: str
    divergence: float
    surprise_route: str | None        # None | "ERROR" | "NOVELTY"
    reward: float
    done: bool
