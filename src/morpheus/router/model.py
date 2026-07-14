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

# Signaux GATÉS : indicateur `*_probed` → (drapeau de config qui les sonde, colonnes de valeur).
# `as_vector()` épingle un signal non sondé à 0.0. Si l'entraînement l'a TOUJOURS sondé, cette
# valeur 0 est hors-distribution : standardisée elle vaut (0 − mu)/sigma, soit une dérive
# CONSTANTE du logit — un biais dû au régime, pas à la situation. D'où `regime_drift`.
GATED_SIGNALS: dict[str, tuple[str, tuple[str, ...]]] = {
    "kb_probed": ("use_rag", ("kb_top_score", "kb_hits")),
    "memory_probed": ("use_memory", ("memory_hits",)),
    "reducibility_probed": ("use_reducibility", ("reducibility",)),
    "direction_probed": ("use_world_model", ("direction",)),
}

# Au-delà de ce décalage de logit, le régime fausse les décisions : sigmoïde(1.0) = 0.73 vs
# 0.5 au repos. En-deçà, la dérive est réelle mais n'atteint pas le seuil de décision.
MAX_REGIME_DRIFT = 1.0


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

    def _at(self, name: str) -> int:
        return self.feature_names.index(name)

    def probe_rate(self, probed_col: str) -> float:
        """Fraction des exemples d'entraînement où ce signal était sondé.

        `mu` d'une colonne indicatrice EST son taux de sondage — le régime d'entraînement se
        lit donc dans le checkpoint, sans métadonnée à maintenir (et vaut pour les checkpoints
        déjà versionnés)."""
        return float(self.mu[self._at(probed_col)])

    def regime_drift(self, **live_flags: bool) -> tuple[float, dict[str, float]]:
        """Décalage SYSTÉMATIQUE du logit induit par un régime de sondage ≠ celui du train.

        Pour un signal sondé à l'entraînement mais PAS en live, la feature est épinglée à 0
        alors que le modèle a appris autour de `mu` ⇒ contribution figée `w·(0 − mu)/sigma`,
        appliquée à CHAQUE décision. On l'estime par rapport à la moyenne d'entraînement (la
        seule référence disponible hors-ligne). Positif ⇒ pousse vers ERREUR ; négatif ⇒ vers
        NOUVEAUTÉ. Retourne (total, détail par feature ; les contributions nulles sont omises).

        Le cas inverse (sondé en live, jamais au train) est inerte par construction : une
        colonne constante a un gradient nul ⇒ `w = 0` ⇒ contribution 0. Le signal est
        simplement IGNORÉ par le routeur — cf. `unused_live_signals`.
        """
        total, detail = 0.0, {}
        for probed_col, (flag, value_cols) in GATED_SIGNALS.items():
            if probed_col not in self.feature_names or live_flags.get(flag, False):
                continue                                  # sondé en live (ou feature absente)
            if self.probe_rate(probed_col) <= 0.0:
                continue                                  # jamais sondé au train non plus : régimes d'accord
            for col in (probed_col, *value_cols):
                if col not in self.feature_names:
                    continue
                i = self._at(col)
                shift = float(self.w[i] * (0.0 - self.mu[i]) / self.sigma[i])
                if shift:
                    detail[col] = shift
                    total += shift
        return total, detail

    def unused_live_signals(self, **live_flags: bool) -> list[str]:
        """Signaux sondés en live mais JAMAIS à l'entraînement ⇒ poids 0, routeur aveugle.

        Pas une erreur (aucune dérive) mais une limite à énoncer : `direction` en est le cas
        emblématique — le signal qui pourrait attraper `coherent_but_wrong` est disponible
        dans la boucle et ignoré par un routeur entraîné sans lui.
        """
        out = []
        for probed_col, (flag, _) in GATED_SIGNALS.items():
            if (probed_col in self.feature_names and live_flags.get(flag, False)
                    and self.probe_rate(probed_col) <= 0.0):
                out.append(probed_col.removesuffix("_probed"))
        return out

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
