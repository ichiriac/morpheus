"""Signal de divergence + routeur de surprise.

C'est le cœur scientifique de morpheus (cf. specs/01, section « routeur de surprise »).
En Phase 1, tout est volontairement MINIMAL et heuristique — l'objectif est d'avoir les
points d'ancrage (interfaces, features) instrumentés, pas encore un routeur appris.

- `divergence(pred, real)` : δ = 1 - recouvrement de tokens (proxy de dist latente).
  En Phase 2, remplacer par `dist(ŝ', E_state(obs_réelle))` dans l'espace JEPA.
- `route(...)` : ERREUR vs NOUVEAUTÉ, à partir des signaux du tableau specs/01.
  Phase 1 : règle simple. Phase 4 : petit classifieur appris.
"""

from __future__ import annotations

import re

ERROR = "ERROR"
NOVELTY = "NOVELTY"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]+", s.lower()))


def divergence(predicted: str, real: str) -> float:
    """δ ∈ [0, 1]. Proxy Phase 1 (Jaccard inversé) de la distance latente."""
    a, b = _tokens(predicted), _tokens(real)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return 1.0 - inter / union


class SurpriseRouter:
    """Désambiguïse une surprise : ERREUR (j'ai fauté) vs NOUVEAUTÉ (monde plus riche)."""

    def route(
        self,
        *,
        delta: float,
        tool_error: bool,
        score_before: float,
        score_after: float,
    ) -> str:
        """Règle Phase 1 combinant deux signaux du tableau specs/01 :
        - « signature de l'outil » : erreur explicite ⇒ ERREUR ;
        - « direction dans le latent » : si l'état s'éloigne du but ⇒ ERREUR, sinon NOUVEAUTÉ.
        """
        if tool_error:
            return ERROR
        if score_after < score_before:  # on s'est éloigné du but
            return ERROR
        return NOVELTY  # surprise mais on reste aligné vers le but ⇒ nouveauté légitime
