#!/usr/bin/env python
"""Sanity RAPIDE du signal goal-relative d'un checkpoint JEPA (itération GPU).

Ne remplace PAS `validate_goal_signal.py` (H1/H2 sur de VRAIES trajectoires τ²) : c'est le
harnais d'itération pendant le fix. Mesure, sur des trajectoires SYNTHÉTIQUES contrôlées, si
`score_to_goal(goal, state)` :

  1. MONOTONIE   : croît le long d'une trajectoire (Spearman t↑ vs score↑), et d'ÉTENDUE
                   FRANCHE (max−min par trajectoire) — c.-à-d. plus le 0.0086 dégénéré.
  2. DISCRIMINATION : l'état terminal note plus haut pour SON but que pour les autres buts
                   (marge terminal_own − mean(terminal_autres_buts) > 0).

Usage :
    python scripts/check_goal_discrimination.py --checkpoint checkpoints/jepa_apigen_goal/jepa.pt
    [--n-episodes 30] [--device auto]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(y) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sanity discrimination goal-relative (synthétique).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from morpheus.agents.jepa_world_model import JepaWorldModel
    from morpheus.jepa.data import synthetic_transitions

    wm = JepaWorldModel(args.checkpoint, device=args.device)
    trans = synthetic_transitions(n_episodes=args.n_episodes, seed=1)

    # regrouper par trajectoire : états RÉSULTANTS (next_obs) dans l'ordre + but
    trajs: dict[int, dict] = {}
    for t in trans:
        tr = trajs.setdefault(t.traj_id, {"goal": t.goal, "states": [], "progress": []})
        tr["states"].append(t.next_obs)
        tr["progress"].append(t.progress)

    rhos, ranges, terminals = [], [], []
    for tid, tr in trajs.items():
        scores = np.array([wm.score_to_goal(tr["goal"], s) for s in tr["states"]])
        rhos.append(_spearman(np.arange(len(scores)), scores))
        ranges.append(float(scores.max() - scores.min()))
        terminals.append((tid, tr["goal"], tr["states"][-1], float(scores[-1])))

    # discrimination inter-buts : état terminal de A noté contre le but de A vs les autres buts
    margins = []
    goals = {tid: tr["goal"] for tid, tr in trajs.items()}
    for tid, goal, term_state, own in terminals:
        others = [wm.score_to_goal(g, term_state) for t2, g in goals.items() if t2 != tid]
        if others:
            margins.append(own - float(np.mean(others)))

    rhos, ranges, margins = np.array(rhos), np.array(ranges), np.array(margins)
    print(f"=== Sanity goal-relative | {args.checkpoint} ===")
    print(f"trajectoires synthétiques : {len(trajs)}")
    print(f"[MONOTONIE]      Spearman t↑vs score : moyenne={rhos.mean():+.3f} "
          f"médiane={np.median(rhos):+.3f} frac>0={np.mean(rhos > 0):.2f}")
    print(f"[ÉTENDUE]        max−min par traj    : moyenne={ranges.mean():.4f} "
          f"médiane={np.median(ranges):.4f}  (cible : franche, ≫ 0.0086)")
    print(f"[DISCRIMINATION] terminal own−autres  : moyenne={margins.mean():+.4f} "
          f"frac>0={np.mean(margins > 0):.2f}")

    ok = rhos.mean() > 0.3 and ranges.mean() > 0.05 and margins.mean() > 0.02
    print("\n" + ("✅ signal goal-relative FRANC (sanity synthétique)."
                  if ok else "⚠️ signal encore faible — ajuster w_goal / epochs / temp."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
