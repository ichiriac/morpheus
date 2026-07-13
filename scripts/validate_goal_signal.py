#!/usr/bin/env python
"""Validation du signal goal-relative de `JepaWorldModel.score_to_goal` (gate de l'étape 4).

Teste trois hypothèses sur de VRAIES trajectoires (PAS synthetic : token-leak du but dans
l'état terminal → corrélation trivialement gonflée) :

  H1  monotonie sur succès   : sur les rollouts RÉSOLUS, score_to_goal(goal, s_t) croît avec t
                               (corrélation de rang de Spearman t↑ vs score↑, en moyenne > 0).
  H2  séparation succès/échec: la PENTE moyenne de score_to_goal est > sur trajectoires
                               résolues que sur trajectoires échouées.
  H3  utilité routeur        : `score_after < score_before` sépare mieux que le hasard les pas
                               étiquetés ERREUR vs NOUVEAUTÉ (matrice de confusion vs annotation
                               MANUELLE).

Significativité par permutation (numpy pur, pas de scipy). Si H1/H2 échouent → l'espace n'est
pas goal-relative : passer à `P(z,a,g)` conditionné OU ajouter un terme d'alignement
but↔état-terminal dans la perte ; en attendant, score_to_goal reste un PROXY documenté.

Entrées (au choix) :
  --trajectories traj.jsonl   {goal, success, states:[str,...], scores?:[float,...],
                               annotations?:[{turn:int,label:"ERROR"|"NOVELTY"}]}
  --episodes runs/X/episodes.jsonl   format runner (states = trace.real_state ; goal requis
                               dans l'épisode — voir --require-goal). Annotations via --labels.
  --labels labels.jsonl       annotations MANUELLES H3 : {episode|id, turn, label}

Scores : fournis dans l'entrée, sinon calculés via --checkpoint <jepa.pt> (JepaWorldModel).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ALPHA = 0.05
MIN_LEN = 3          # longueur mini d'une trajectoire pour une pente/corrélation exploitable
N_PERM = 20000
_RNG = np.random.default_rng(0)


# --------------------------- stats (numpy pur) --------------------------- #

def _ranks(x: np.ndarray) -> np.ndarray:
    """Rangs moyens (gère les ex æquo)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    # moyenne des rangs pour les ex æquo
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum - 1) / 2.0
    return avg[inv]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(y) == 0:
        return 0.0
    return _pearson(_ranks(x), _ranks(y))


def slope(y: np.ndarray) -> float:
    """Pente OLS de y contre t = 0..n-1."""
    t = np.arange(len(y), dtype=float)
    if np.ptp(t) == 0:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def perm_p_mean_gt0(vals: np.ndarray, n: int = N_PERM) -> float:
    """p unilatéral H0: médiane/moyenne des `vals` = 0, via inversion de signe aléatoire."""
    vals = np.asarray(vals, float)
    obs = vals.mean()
    if obs <= 0:
        return 1.0
    signs = _RNG.choice([-1.0, 1.0], size=(n, len(vals)))
    perm = (signs * np.abs(vals)).mean(axis=1)
    return float((perm >= obs).mean())


def perm_p_group_gt(a: np.ndarray, b: np.ndarray, n: int = N_PERM) -> float:
    """p unilatéral H0: moyenne(a) <= moyenne(b), via permutation des étiquettes de groupe."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    if obs <= 0:
        return 1.0
    pool = np.concatenate([a, b]); na = len(a)
    diffs = np.empty(n)
    for i in range(n):
        p = _RNG.permutation(pool)
        diffs[i] = p[:na].mean() - p[na:].mean()
    return float((diffs >= obs).mean())


# --------------------------- chargement --------------------------- #

def load_trajectories(args) -> list[dict]:
    trajs: list[dict] = []
    if args.trajectories:
        for line in Path(args.trajectories).read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajs.append(json.loads(line))
    if args.episodes:
        labels = _load_labels(args.labels) if args.labels else {}
        for src in args.episodes:
            for i, line in enumerate(Path(src).read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                ep = json.loads(line)
                states = [s.get("real_state", "") for s in ep.get("trace", [])]
                # clé stable = nom du run (dossier) + index de tâche, indépendante du chemin passé
                key = f"{Path(src).parent.name}#{ep.get('task', i)}"
                anns = labels.get(key, [])
                trajs.append({
                    "id": key,
                    "goal": ep.get("goal"),   # None si non persisté (voir --require-goal)
                    "success": bool(ep.get("success")),
                    "states": states,
                    "annotations": anns,
                })
    return trajs


def _load_labels(path: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = r.get("episode") or r.get("id")
        out.setdefault(key, []).append({"turn": int(r["turn"]), "label": r["label"]})
    return out


def compute_scores(trajs: list[dict], checkpoint: str) -> None:
    """Remplit `scores` pour chaque trajectoire via JepaWorldModel.score_to_goal."""
    from morpheus.agents.jepa_world_model import JepaWorldModel

    wm = JepaWorldModel(checkpoint, device="auto")
    for tr in trajs:
        if tr.get("scores") is not None:
            continue
        goal = tr.get("goal")
        if not goal:
            continue
        tr["scores"] = [wm.score_to_goal(goal, s) for s in tr["states"]]


# --------------------------- hypothèses --------------------------- #

def _usable(tr: dict) -> bool:
    return tr.get("scores") is not None and len(tr["scores"]) >= MIN_LEN


def h1_monotonicity(trajs: list[dict]) -> dict:
    succ = [tr for tr in trajs if tr.get("success") and _usable(tr)]
    rhos = np.array([spearman(np.arange(len(tr["scores"])), np.asarray(tr["scores"]))
                     for tr in succ])
    if len(rhos) == 0:
        return {"status": "N/A", "reason": "0 trajectoire résolue exploitable", "n": 0}
    p = perm_p_mean_gt0(rhos)
    passed = rhos.mean() > 0 and p < ALPHA and len(rhos) >= 5
    return {"status": "PASS" if passed else ("FAIL" if len(rhos) >= 5 else "N/A"),
            "n": int(len(rhos)), "mean_rho": round(float(rhos.mean()), 3),
            "median_rho": round(float(np.median(rhos)), 3),
            "frac_positive": round(float((rhos > 0).mean()), 3), "p": round(p, 4)}


def h2_separation(trajs: list[dict]) -> dict:
    succ = [slope(np.asarray(tr["scores"])) for tr in trajs if tr.get("success") and _usable(tr)]
    fail = [slope(np.asarray(tr["scores"])) for tr in trajs
            if not tr.get("success") and _usable(tr)]
    if len(succ) < 3 or len(fail) < 3:
        return {"status": "N/A", "reason": f"classes insuffisantes (succès={len(succ)}, échecs={len(fail)}, min 3 chacune)",
                "n_success": len(succ), "n_fail": len(fail)}
    p = perm_p_group_gt(np.array(succ), np.array(fail))
    passed = np.mean(succ) > np.mean(fail) and p < ALPHA
    return {"status": "PASS" if passed else "FAIL", "n_success": len(succ), "n_fail": len(fail),
            "slope_success": round(float(np.mean(succ)), 4), "slope_fail": round(float(np.mean(fail)), 4),
            "p": round(p, 4)}


def h3_router(trajs: list[dict]) -> dict:
    # (pred ERREUR = score_after < score_before) vs label MANUEL, sur les pas annotés.
    y_true, y_pred = [], []
    for tr in trajs:
        scores = tr.get("scores")
        if not scores:
            continue
        for a in tr.get("annotations", []):
            k = a["turn"]
            if k < 1 or k >= len(scores):     # besoin de score_before (k-1) et score_after (k)
                continue
            before, after = scores[k - 1], scores[k]
            y_pred.append(after < before)      # True ⇒ prédit ERREUR
            y_true.append(a["label"].upper() == "ERROR")
    n = len(y_true)
    if n < 30:
        return {"status": "N/A", "reason": f"annotations manuelles insuffisantes ({n} pas, ~50 visés)", "n": n}
    yt, yp = np.array(y_true), np.array(y_pred)
    tp = int((yt & yp).sum()); tn = int((~yt & ~yp).sum())
    fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
    acc = (tp + tn) / n
    base = max(yt.mean(), 1 - yt.mean())       # taux du classifieur majoritaire
    # p : la prédiction fait-elle mieux que des labels permutés ?
    obs = acc
    perm = np.array([( (yp == _RNG.permutation(yt)).mean() ) for _ in range(N_PERM)])
    p = float((perm >= obs).mean())
    passed = acc > base and p < ALPHA
    return {"status": "PASS" if passed else "FAIL", "n": n,
            "confusion": {"TP_err": tp, "FP": fp, "FN": fn, "TN_nov": tn},
            "accuracy": round(acc, 3), "chance_baseline": round(float(base), 3), "p": round(p, 4)}


# --------------------------- main --------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Valide le signal goal-relative (H1/H2/H3).")
    ap.add_argument("--trajectories", help="JSONL {goal,success,states,scores?,annotations?}")
    ap.add_argument("--episodes", nargs="*", help="un ou plusieurs runs/X/episodes.jsonl")
    ap.add_argument("--labels", help="annotations manuelles H3 (JSONL {episode,turn,label})")
    ap.add_argument("--checkpoint", help="jepa.pt pour calculer les scores si absents")
    ap.add_argument("--require-goal", action="store_true",
                    help="échoue si des trajectoires n'ont pas de goal (score non calculable)")
    args = ap.parse_args(argv)

    if not args.trajectories and not args.episodes:
        ap.error("fournir --trajectories et/ou --episodes")

    trajs = load_trajectories(args)
    n_total = len(trajs)
    n_goal = sum(1 for t in trajs if t.get("goal"))
    n_succ = sum(1 for t in trajs if t.get("success"))

    if args.require_goal and n_goal < n_total:
        print(f"❌ {n_total - n_goal}/{n_total} trajectoires SANS goal persisté — "
              "score_to_goal incalculable. Ré-exporter avec le goal (voir runner).")
        return 2

    need_scores = any(t.get("scores") is None for t in trajs)
    if need_scores:
        if not args.checkpoint:
            print("⚠️  scores absents et --checkpoint non fourni : H1/H2/H3 seront N/A.")
        elif n_goal == 0:
            print("⚠️  aucun goal disponible : impossible de calculer les scores.")
        else:
            compute_scores(trajs, args.checkpoint)

    n_ann = sum(len(t.get("annotations", [])) for t in trajs)
    print(f"\n=== Corpus === trajectoires={n_total} | avec goal={n_goal} | résolues={n_succ}"
          f" | annotations H3 jointes={n_ann}")
    if args.checkpoint:
        print(f"    JEPA : {args.checkpoint}")

    results = {"H1_monotonie_succes": h1_monotonicity(trajs),
               "H2_separation_succes_echec": h2_separation(trajs),
               "H3_utilite_routeur": h3_router(trajs)}
    for name, res in results.items():
        print(f"\n[{res['status']}] {name}")
        for k, v in res.items():
            if k != "status":
                print(f"      {k}: {v}")

    gate = [results["H1_monotonie_succes"]["status"], results["H2_separation_succes_echec"]["status"]]
    print("\n=== VERDICT étape 4 ===")
    if "FAIL" in gate:
        print("  ❌ H1/H2 en échec → espace NON goal-relative. Action : entraîner P(z,a,g) "
              "conditionné sur le but, OU terme d'alignement but↔état-terminal dans la perte. "
              "score_to_goal reste un PROXY documenté.")
    elif "N/A" in gate:
        print("  ⏸️  H1/H2 non concluantes (données insuffisantes — cf. raisons). "
              "score_to_goal reste un PROXY tant que non validé.")
    else:
        print("  ✅ H1/H2 validées → signal goal-relative confirmé. score_to_goal n'est plus un proxy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
