"""Modèle du routeur appris (Phase 4 v0) : régression logistique ERREUR-vs-NOUVEAUTÉ.

Le plus petit classifieur possible — UNE couche W·x + sigmoïde — choisi à dessein :
~30 exemples ERREUR annotés seulement (data/annotations), chaque paramètre doit compter ;
et les poids se LISENT signal par signal (l'interprétabilité EST la thèse : quels signaux
désambiguïsent ERREUR vs NOUVEAUTÉ ?). numpy pur, sérialisé JSON → chargeable dans
loop.py partout, sans torch à l'inférence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.surprise import ERROR, NOVELTY, SurpriseSignals


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoïde numériquement stable (pas d'overflow d'exp sur les grands |x|)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class RouterModel:
    """Poids + standardisation appris. `w` se lit par feature : poids > 0 ⇒ pousse vers ERREUR."""

    feature_names: tuple[str, ...]
    mu: np.ndarray                     # standardisation (apprise sur le train uniquement)
    sigma: np.ndarray
    w: np.ndarray                      # LA couche W·x (un logit : ERREUR)
    b: float
    threshold: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """p(ERREUR) par ligne de X — features BRUTES, ordre `feature_names` (VECTOR_FIELDS)."""
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sigma
        return sigmoid(Z @ self.w + self.b)

    def route(self, signals: SurpriseSignals) -> str:
        """Même contrat que SurpriseRouter.route — interchangeable dans la boucle (Phase 4)."""
        p = float(self.predict_proba(np.asarray([signals.as_vector()]))[0])
        return ERROR if p >= self.threshold else NOVELTY

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_names": list(self.feature_names),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "w": self.w.tolist(),
            "b": self.b,
            "threshold": self.threshold,
            "meta": self.meta,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RouterModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(d["feature_names"]),
            mu=np.asarray(d["mu"], dtype=float),
            sigma=np.asarray(d["sigma"], dtype=float),
            w=np.asarray(d["w"], dtype=float),
            b=float(d["b"]),
            threshold=float(d.get("threshold", 0.5)),
            meta=d.get("meta", {}),
        )
