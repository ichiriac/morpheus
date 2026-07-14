"""Signal de divergence + routeur de surprise.

C'est le cœur scientifique de morpheus (cf. specs/01, section « routeur de surprise »).
En Phase 1, tout est volontairement MINIMAL et heuristique — l'objectif est d'avoir les
points d'ancrage (interfaces, features) instrumentés, pas encore un routeur appris.

- `divergence(pred, real)` : δ = 1 - recouvrement de tokens (proxy de dist latente).
  En Phase 2, remplacer par `dist(ŝ', E_state(obs_réelle))` dans l'espace JEPA.
- `SurpriseSignals` : le vecteur de signaux collecté à CHAQUE surprise (tableau specs/01
  + rubrique d'annotation `data/annotations`), journalisé dans `TraceStep.signals` —
  matière première du routeur APPRIS de Phase 4.
- `SurpriseRouter.route(signals)` : ERREUR vs NOUVEAUTÉ. Phase 1 : règle simple sur deux
  signaux (inchangée) ; les autres sont collectés mais pas encore consommés.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Iterable

ERROR = "ERROR"
NOVELTY = "NOVELTY"

# Outil synthétique « parler à l'utilisateur » (cf. envs/tau2_adapter). Répéter CET outil
# est du dialogue normal, pas une boucle — exclu du signal `repeated_tool` (revue des
# annotations 2026-07-14 : 18/28 répétitions étaient des respond_to_user légitimes).
DIALOGUE_TOOL = "respond_to_user"


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


def familiarity(real: str, past: Iterable[str]) -> float:
    """Proxy Phase 1 du signal « localité vs globalité » (specs/01) : recouvrement lexical
    MAX entre l'observation réelle et les observations passées de l'épisode. Haut = l'état
    ressemble à du déjà-vu (extension de la structure) ; bas = rupture avec tout l'observé."""
    best = 0.0
    for p in past:
        best = max(best, 1.0 - divergence(real, p))
        if best >= 1.0:
            break
    return best


@dataclass
class SurpriseSignals:
    """Signaux d'une surprise (δ > seuil), collectés par la boucle au moment du routage.

    Mapping specs/01 §routeur : `delta` (amplitude), `tool_error` (signature de l'outil),
    `score_before/after` (direction dans le latent), `kb_*` (cohérence RAG), `familiarity`
    (localité vs globalité, proxy lexical), `reducibility` (réductibilité, sonde LLM).
    `repeated_tool` / `is_user_turn` viennent de la rubrique d'annotation réelle
    (`data/annotations` : rationales `loop_no_progress`, `user_new_info`).

    None = signal NON SONDÉ (gating : `kb_*` si use_rag, `memory_hits` si use_memory,
    `reducibility` si use_reducibility) — l'absence est une information, pas un zéro.
    """

    delta: float                          # amplitude de la surprise (le déclencheur)
    tool_error: bool                      # signature de l'outil : erreur explicite ?
    score_before: float | None = None     # proximité au but AVANT le pas…
    score_after: float | None = None      # …et APRÈS : la « direction dans le latent »
    kb_top_score: float | None = None     # cohérence RAG : meilleur score BM25 (0 = KB muette)
    kb_hits: int | None = None            # nb de règles pertinentes (score > 0, top-k)
    memory_hits: int | None = None        # idem, mémoire épisodique
    familiarity: float | None = None      # localité : recouvrement max avec les obs passées
    repeated_tool: bool = False           # même outil HORS DIALOGUE qu'au pas précédent (sans erreur)
    is_user_turn: bool = False            # l'observation vient de l'utilisateur (respond_to_user)
    reducibility: float | None = None     # l'écart s'explique-t-il ? 1 = oui (sonde LLM opt-in)

    # Ordre FIGÉ des features denses (reproductibilité du classifieur Phase 4).
    VECTOR_FIELDS: ClassVar[tuple[str, ...]] = (
        "delta", "tool_error",
        "direction", "direction_probed",
        "kb_top_score", "kb_hits", "kb_probed",
        "memory_hits", "memory_probed",
        "familiarity", "familiarity_probed",
        "repeated_tool", "is_user_turn",
        "reducibility", "reducibility_probed",
    )

    @property
    def direction(self) -> float | None:
        """score_after − score_before (>0 = on se rapproche du but) ; None si non sondé."""
        if self.score_before is None or self.score_after is None:
            return None
        return self.score_after - self.score_before

    def to_dict(self) -> dict[str, Any]:
        """Rendu JSON-sérialisable pour `TraceStep.signals` (les None journalisés tels quels)."""
        return asdict(self)

    def as_vector(self) -> list[float]:
        """Features denses de shape FIXE (cf. VECTOR_FIELDS) pour le classifieur Phase 4 :
        chaque signal optionnel devient (valeur ou 0, indicateur sondé/non-sondé) — le
        classifieur peut ainsi apprendre de l'absence au lieu de la confondre avec zéro."""
        d = self.direction
        return [
            self.delta,
            1.0 if self.tool_error else 0.0,
            d if d is not None else 0.0, 0.0 if d is None else 1.0,
            self.kb_top_score if self.kb_top_score is not None else 0.0,
            float(self.kb_hits or 0), 0.0 if self.kb_hits is None else 1.0,
            float(self.memory_hits or 0), 0.0 if self.memory_hits is None else 1.0,
            self.familiarity if self.familiarity is not None else 0.0,
            0.0 if self.familiarity is None else 1.0,
            1.0 if self.repeated_tool else 0.0,
            1.0 if self.is_user_turn else 0.0,
            self.reducibility if self.reducibility is not None else 0.0,
            0.0 if self.reducibility is None else 1.0,
        ]


class SurpriseRouter:
    """Désambiguïse une surprise : ERREUR (j'ai fauté) vs NOUVEAUTÉ (monde plus riche)."""

    def route(self, signals: SurpriseSignals) -> str:
        """Règle Phase 1 INCHANGÉE (deux signaux du tableau specs/01) — le reste du vecteur
        est journalisé dans la trace, pas encore consommé : on instrumente d'abord, le
        routeur appris (Phase 4) tranchera sur trajectoires annotées.
        - « signature de l'outil » : erreur explicite ⇒ ERREUR ;
        - « direction dans le latent » : si l'état s'éloigne du but ⇒ ERREUR, sinon NOUVEAUTÉ.
        """
        if signals.tool_error:
            return ERROR
        d = signals.direction
        if d is not None and d < 0.0:   # on s'est éloigné du but
            return ERROR
        return NOVELTY  # surprise mais on reste aligné vers le but ⇒ nouveauté légitime
