"""Environnement mock « retail-lite » — multi-tours, paramétrable en longueur.

But : faire tourner la boucle de bout en bout SANS τ²-bench, et surtout produire une
courbe « réussite vs nombre de tours » exploitable — donc des tâches dont l'horizon de
solution varie (4 / 8 / 12 pas). Il injecte aussi des états surprenants pour exercer le
signal de divergence et le routeur (ERREUR vs NOUVEAUTÉ).

Tâche : traiter une demande client en franchissant une CHAÎNE d'étapes ordonnées. Chaque
étape a un outil attendu ; appeler le bon outil au bon moment avance d'un cran. Un pas sur
`novelty_at` renvoie un état légitime mais inattendu (nouveauté), un mauvais outil renvoie
une erreur d'outil (erreur).
"""

from __future__ import annotations

from ..orchestrator.types import Action, Observation, StepResult

# Chaîne d'outils de référence (ordre = plan attendu A→B→C→…).
_CHAIN = [
    "authenticate_user",
    "lookup_order",
    "check_refund_policy",
    "verify_payment_method",
    "check_inventory",
    "compute_refund_amount",
    "confirm_with_user",
    "issue_refund",
    "send_confirmation_email",
    "close_ticket",
    "log_resolution",
    "archive_case",
]
_DISTRACTORS = ["escalate_to_human", "cancel_order", "apply_discount", "noop"]


class MockRetailEnv:
    def __init__(self, length: int, novelty_at: int | None = None, seed: int = 0) -> None:
        self.length = max(1, min(length, len(_CHAIN)))
        self.chain = _CHAIN[: self.length]
        self.novelty_at = novelty_at
        self.seed = seed
        self.pos = 0
        self._done = False

    # --- API Env ---
    def reset(self) -> Observation:
        self.pos = 0
        self._done = False
        return Observation(text=f"Nouveau ticket. Prochaine étape attendue : {self.chain[0]}.")

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(Observation("épisode terminé"), 0.0, True, {"success": False})

        expected = self.chain[self.pos]
        if action.tool != expected:
            # mauvais outil → erreur d'outil (surprise de type ERREUR), on n'avance pas
            return StepResult(
                Observation(
                    text=f"ERREUR outil : '{action.tool}' inattendu, attendu '{expected}'.",
                    tool_error=True,
                ),
                reward=-0.05,
                done=False,
                info={"success": False},
            )

        self.pos += 1
        done = self.pos >= self.length
        self._done = done
        if done:
            return StepResult(
                Observation(text=f"Étape {expected} OK. Demande RÉSOLUE."),
                reward=1.0,
                done=True,
                info={"success": True},
            )

        nxt = self.chain[self.pos]
        # nouveauté légitime : état inattendu mais aligné vers le but
        if self.novelty_at is not None and self.pos == self.novelty_at:
            text = (
                f"Étape {expected} OK — note inattendue : le compte est de type PREMIUM, "
                f"traitement prioritaire. Prochaine étape attendue : {nxt}."
            )
        else:
            text = f"Étape {expected} OK. Prochaine étape attendue : {nxt}."
        return StepResult(Observation(text=text), reward=0.1, done=False, info={"success": False})

    def goal(self) -> str:
        return (
            "Résoudre la demande de remboursement du client en suivant, dans l'ordre, "
            f"les étapes : {' -> '.join(self.chain)}."
        )

    def tool_names(self) -> list[str]:
        return self.chain + _DISTRACTORS

    def required_turns(self) -> int:
        return self.length

    def system_context(self) -> str | None:
        # Le mock encode déjà l'objectif dans goal() ; pas de manuel de domaine séparé.
        return None


def make_mock_env(task_index: int, seed: int, buckets: list[int]) -> MockRetailEnv:
    """Génère une tâche dont la longueur cycle sur `buckets` (4/8/12), avec une nouveauté
    injectée à mi-parcours une tâche sur deux — pour peupler la courbe vs-tours."""
    length = buckets[task_index % len(buckets)]
    novelty = (length // 2) if (task_index % 2 == 0) else None
    return MockRetailEnv(length=length, novelty_at=novelty, seed=seed + task_index)
