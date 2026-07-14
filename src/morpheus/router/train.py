"""Entraînement du routeur de surprise (Phase 4 v0) : descente de gradient À LA MAIN.

`w ← w − η·∇` écrit explicitement — pas d'autograd : le gradient de la log-vraisemblance
logistique est analytique (`X.T @ (p − y)`). Protocole de mesure :

- **validation croisée stratifiée** (les deux classes dans chaque pli ; 30 ERREUR
  seulement → un simple split gâcherait des exemples) ;
- comparaison contre (a) la **classe majoritaire** et (b) la **règle heuristique Phase 1**
  (`SurpriseRouter`, 2 signaux) sur les MÊMES exemples : c'est LA mesure — les signaux
  instrumentés apportent-ils quelque chose au-delà de la règle ? ;
- **significativité par permutation** (numpy pur, même esprit que validate_goal_signal) ;
- la métrique pivot est la **balanced accuracy** (30/79 : l'accuracy nue récompenserait
  « toujours NOUVEAUTÉ » à 72 %).

Usage : `morpheus train-router --dataset data/router/dataset.jsonl` (CPU, instantané).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..agents.surprise import ERROR, SurpriseRouter, SurpriseSignals
from .model import RouterModel, sigmoid

N_PERM = 5000


# --------------------------- données --------------------------- #

def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """dataset.jsonl (scripts/build_router_dataset.py) → (X, y, rows). y : 1 = ERREUR."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"dataset vide : {path}")
    X = np.asarray([r["vector"] for r in rows], dtype=float)
    y = np.asarray([1.0 if str(r["label"]).upper() == ERROR else 0.0 for r in rows])
    return X, y, rows


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) sur X ; sigma des colonnes constantes clampé à 1 (pas de division par 0)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """k plis contenant chacun ~1/k de CHAQUE classe (indispensable avec 30 ERREUR)."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    return [np.asarray(sorted(f)) for f in folds]


# --------------------------- apprentissage --------------------------- #

def fit_logreg(X: np.ndarray, y: np.ndarray, *, epochs: int = 3000, lr: float = 0.5,
               l2: float = 1e-2) -> tuple[np.ndarray, float]:
    """Régression logistique, gradient plein-batch écrit à la main, classes équilibrées
    (chaque classe pèse 1/2 quel que soit son effectif). X doit être standardisé."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    n_pos = max(1.0, float(y.sum()))
    n_neg = max(1.0, float(n - y.sum()))
    sw = np.where(y > 0.5, 1.0 / (2.0 * n_pos), 1.0 / (2.0 * n_neg))  # somme = 1
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        r = sw * (p - y)               # résidu pondéré par classe
        grad_w = X.T @ r + l2 * w      # ∇(log-vraisemblance) + régularisation L2
        grad_b = float(r.sum())
        w -= lr * grad_w               # ← la marche 2, littéralement
        b -= lr * grad_b
    return w, b


# --------------------------- évaluation --------------------------- #

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true > 0.5
    yp = y_pred > 0.5
    tp = int((yt & yp).sum()); fn = int((yt & ~yp).sum())
    fp = int((~yt & yp).sum()); tn = int((~yt & ~yp).sum())
    rec_err = tp / max(1, tp + fn)      # rappel ERREUR
    rec_nov = tn / max(1, tn + fp)      # rappel NOUVEAUTÉ
    return {
        "TP_err": tp, "FN": fn, "FP": fp, "TN_nov": tn,
        "accuracy": round((tp + tn) / max(1, len(yt)), 3),
        "recall_error": round(rec_err, 3),
        "recall_novelty": round(rec_nov, 3),
        "balanced_accuracy": round((rec_err + rec_nov) / 2.0, 3),
    }


def heuristic_predictions(rows: list[dict]) -> np.ndarray:
    """La règle Phase 1 (SurpriseRouter) rejouée sur les MÊMES exemples — la baseline à battre."""
    router = SurpriseRouter()
    return np.asarray([
        1.0 if router.route(SurpriseSignals(**r["signals"])) == ERROR else 0.0
        for r in rows
    ])


def perm_p_balanced(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0,
                    n: int = N_PERM) -> float:
    """p unilatéral H0 : la balanced accuracy observée s'obtient avec des labels permutés."""
    rng = np.random.default_rng(seed)
    obs = confusion(y_true, y_pred)["balanced_accuracy"]
    hits = 0
    for _ in range(n):
        if confusion(rng.permutation(y_true), y_pred)["balanced_accuracy"] >= obs:
            hits += 1
    return hits / n


def cross_validate(X: np.ndarray, y: np.ndarray, *, folds: int, epochs: int, lr: float,
                   l2: float, seed: int, threshold: float = 0.5) -> tuple[np.ndarray, int]:
    """Prédictions out-of-fold (chaque exemple prédit par un modèle qui ne l'a PAS vu).
    Standardisation apprise par pli (sur le train seul) — pas de fuite train→val."""
    k = int(min(folds, y.sum(), (1 - y).sum()))   # jamais plus de plis que la classe rare
    preds = np.zeros_like(y)
    for val_idx in stratified_folds(y, k, seed):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        mu, sigma = standardize(X[train_mask])
        w, b = fit_logreg((X[train_mask] - mu) / sigma, y[train_mask],
                          epochs=epochs, lr=lr, l2=l2)
        p = sigmoid(((X[val_idx] - mu) / sigma) @ w + b)
        preds[val_idx] = (p >= threshold).astype(float)
    return preds, k


# --------------------------- entraînement complet --------------------------- #

def train_router(dataset: str, out_dir: str = "checkpoints/router", *, folds: int = 5,
                 epochs: int = 3000, lr: float = 0.5, l2: float = 1e-2,
                 seed: int = 0) -> int:
    X, y, rows = load_dataset(dataset)
    names = tuple(rows[0].get("vector_fields") or SurpriseSignals.VECTOR_FIELDS)
    if X.shape[1] != len(names):
        raise RuntimeError(f"vecteur de taille {X.shape[1]} ≠ {len(names)} features attendues")
    n_err = int(y.sum())
    print(f"=== Corpus === {len(y)} pas annotés | ERREUR={n_err} | NOUVEAUTÉ={len(y) - n_err}")

    # 1. mesure honnête : prédictions out-of-fold vs les deux baselines
    preds, k = cross_validate(X, y, folds=folds, epochs=epochs, lr=lr, l2=l2, seed=seed)
    learned = confusion(y, preds)
    learned["p_perm"] = round(perm_p_balanced(y, preds, seed=seed), 4)
    heur = confusion(y, heuristic_predictions(rows))
    majority = confusion(y, np.zeros_like(y))     # « toujours NOUVEAUTÉ » (classe majoritaire)

    print(f"\n[APPRIS]     CV stratifiée {k} plis (out-of-fold)")
    for kk, v in learned.items():
        print(f"      {kk}: {v}")
    print("[HEURISTIQUE Phase 1]  (tool_error + direction, mêmes exemples)")
    for kk, v in heur.items():
        print(f"      {kk}: {v}")
    print("[MAJORITAIRE]  (toujours NOUVEAUTÉ)")
    print(f"      accuracy: {majority['accuracy']}  balanced_accuracy: {majority['balanced_accuracy']}")

    gain = learned["balanced_accuracy"] - heur["balanced_accuracy"]
    verdict = ("✅ le routeur appris bat la règle Phase 1"
               if gain > 0 and learned["p_perm"] < 0.05
               else "❌ pas (encore) de gain démontré sur la règle Phase 1")
    print(f"\n=== VERDICT === {verdict} "
          f"(Δ balanced_accuracy = {gain:+.3f}, p_perm = {learned['p_perm']})")

    # 2. modèle final : fit sur TOUT le corpus (le checkpoint servi en Phase 4)
    mu, sigma = standardize(X)
    w, b = fit_logreg((X - mu) / sigma, y, epochs=epochs, lr=lr, l2=l2)
    model = RouterModel(
        feature_names=names, mu=mu, sigma=sigma, w=w, b=b,
        meta={"dataset": str(dataset), "n": int(len(y)), "n_error": n_err,
              "cv": learned, "heuristic": heur, "folds": k,
              "epochs": epochs, "lr": lr, "l2": l2, "seed": seed},
    )
    out = Path(out_dir)
    model.save(out / "router.json")

    print(f"\n=== Poids appris === (poids > 0 ⇒ pousse vers ERREUR ; features standardisées)")
    order = np.argsort(-np.abs(w))
    for i in order:
        print(f"      {names[i]:<22} {w[i]:+.3f}")
    print(f"\n✓ checkpoint : {out / 'router.json'}")
    return 0


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="morpheus train-router")
    p.add_argument("--dataset", required=True, help="dataset.jsonl (build_router_dataset.py)")
    p.add_argument("--out", default="checkpoints/router")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    return train_router(a.dataset, a.out, folds=a.folds, epochs=a.epochs,
                        lr=a.lr, l2=a.l2, seed=a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
