"""Adaptateur τ²-bench → interface `Env` de morpheus.

τ²-bench (Sierra) n'est pas une dépendance obligatoire du scaffold : ce module l'importe
PARESSEUSEMENT et lève un message clair si le paquet n'est pas installé. On garde ainsi la
Phase 1 exécutable en mock, tout en ayant le point de branchement prêt.

Installation prévue (à confirmer selon la distribution retenue) :
    pip install morpheus[tau2]        # ou depuis le repo Sierra tau2-bench

⚠️ Cet adaptateur est un SQUELETTE : les noms d'API exacts de τ²-bench sont à câbler quand
on fige la version (cf. décision runtime dans specs/01). Les `TODO` marquent les points de
raccord. Tant qu'ils ne sont pas faits, `make_tau2_env` lève NotImplementedError explicite.
"""

from __future__ import annotations

from ..orchestrator.types import Action, Observation, StepResult


class Tau2Env:
    """Enveloppe une tâche τ²-bench derrière l'interface `Env`."""

    def __init__(self, task, domain: str) -> None:
        self._task = task
        self._domain = domain
        # TODO(tau2): instancier l'environnement/simulateur d'utilisateur de la tâche.

    def reset(self) -> Observation:
        # TODO(tau2): task.reset() → premier message/état ; mapper vers Observation.
        raise NotImplementedError("câbler Tau2Env.reset sur l'API τ²-bench")

    def step(self, action: Action) -> StepResult:
        # TODO(tau2): traduire Action (tool+args) vers l'appel d'outil τ²-bench,
        #             récupérer l'observation, la reward de tâche et le flag done.
        raise NotImplementedError("câbler Tau2Env.step sur l'API τ²-bench")

    def goal(self) -> str:
        # TODO(tau2): consigne/policy de la tâche.
        raise NotImplementedError

    def tool_names(self) -> list[str]:
        # TODO(tau2): noms des outils exposés par le domaine (retail/airline/telecom).
        raise NotImplementedError

    def required_turns(self) -> int:
        # τ²-bench ne fournit pas toujours une longueur de référence ; à défaut, estimer
        # via la trajectoire annotée (solution) si disponible, sinon retourner -1.
        return -1


def make_tau2_env(task_index: int, domain: str) -> Tau2Env:
    try:
        import tau2  # noqa: F401  (nom du paquet à confirmer)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "τ²-bench non installé. `pip install morpheus[tau2]` (ou depuis le repo Sierra). "
            "En attendant, utilise `--env mock` pour la Phase 1."
        ) from e

    # TODO(tau2): charger le jeu de tâches du domaine et sélectionner l'index.
    raise NotImplementedError(
        "Adaptateur τ²-bench à finaliser : câbler chargement des tâches + reset/step. "
        "Points de raccord marqués TODO(tau2) dans ce fichier."
    )
