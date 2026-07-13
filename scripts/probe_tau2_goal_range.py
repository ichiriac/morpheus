#!/usr/bin/env python
"""Probe DIRECT du signal goal sur de vraies trajectoires τ² : étendue de `score_to_goal`
le long de chaque épisode, à comparer au 0.0086 dégénéré du constat.

    python scripts/probe_tau2_goal_range.py --checkpoint checkpoints/jepa_apigen_goal/jepa.pt \
        --episodes data/annotations/trajectories/*/episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from morpheus.agents.jepa_world_model import JepaWorldModel

    wm = JepaWorldModel(args.checkpoint, device=args.device)
    all_scores = []
    print(f"=== Probe τ² | {args.checkpoint} ===")
    for src in args.episodes:
        for i, line in enumerate(Path(src).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            ep = json.loads(line)
            goal = ep.get("goal")
            if not goal:
                continue
            states = [s.get("real_state", "") for s in ep.get("trace", [])]
            scores = np.array([wm.score_to_goal(goal, s) for s in states])
            if len(scores) < 2:
                continue
            all_scores.append(scores)
            rng = float(scores.max() - scores.min())
            tag = "✓" if ep.get("success") else "✗"
            print(f"  {Path(src).parent.name}#{ep.get('task', i)} {tag} "
                  f"n={len(scores):2d} min={scores.min():.3f} max={scores.max():.3f} "
                  f"étendue={rng:.4f} first={scores[0]:.3f} last={scores[-1]:.3f}")
    if all_scores:
        flat = np.concatenate(all_scores)
        per_ep_range = np.array([s.max() - s.min() for s in all_scores])
        print(f"\nglobal : min={flat.min():.3f} max={flat.max():.3f} "
              f"étendue_globale={flat.max() - flat.min():.4f}")
        print(f"étendue intra-épisode : moyenne={per_ep_range.mean():.4f} "
              f"médiane={np.median(per_ep_range):.4f}  (constat dégénéré : 0.0086)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
