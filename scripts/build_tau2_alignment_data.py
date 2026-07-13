#!/usr/bin/env python
"""SUITE DU FIX (transfert τ²) : construit un dataset d'ALIGNEMENT en-domaine à partir des
trajectoires τ² rejouées, avec un split TRAIN/VAL **par trajectoire** (anti-leak : la validation
`validate_goal_signal.py` ne doit jamais voir un état vu à l'entraînement).

Entrée : data/tau2_replay/retail.jsonl  {goal, success, states:[str], ...}
Sorties :
  data/tau2_replay/retail_align_train.jsonl  transitions {obs,action,next_obs,goal,progress,traj_id}
                                             (source `jsonl:` pour morpheus train-jepa)
  data/tau2_replay/retail_align_val.jsonl    trajectoires held-out {goal,success,states}
                                             (format `--trajectories` de validate_goal_signal)

Progress = position normalisée de l'état RÉSULTANT (next_obs) dans la trajectoire ∈ ]0,1], 1.0 au
dernier état. Le but retail est GÉNÉRIQUE (identique partout) → l'alignement apprend un axe de
PROGRESSION conditionné sur ce but (⇒ entraîner avec w_goal_nce=0 : l'InfoNCE est dégénéré quand
tous les buts sont identiques).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/tau2_replay/retail.jsonl")
    ap.add_argument("--out-dir", default="data/tau2_replay")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if len(r.get("states", [])) >= args.min_len]

    # split déterministe PAR TRAJECTOIRE (hash de l'index mélangé par graine, sans random global)
    order = sorted(range(len(rows)), key=lambda i: (hash((args.seed, i)) & 0xFFFFFFFF))
    n_val = max(3, int(len(order) * args.val_frac))
    val_idx = set(order[:n_val])

    out = Path(args.out_dir)
    train_path = out / "retail_align_train.jsonl"
    val_path = out / "retail_align_val.jsonl"

    n_train_tr = n_train_ep = n_val_ep = 0
    with train_path.open("w", encoding="utf-8") as ftr, val_path.open("w", encoding="utf-8") as fva:
        for i, r in enumerate(rows):
            goal = r.get("goal", "")
            states = r["states"]
            if i in val_idx:
                fva.write(json.dumps({"goal": goal, "success": bool(r.get("success")),
                                      "states": states}, ensure_ascii=False) + "\n")
                n_val_ep += 1
                continue
            T = len(states)
            for k in range(T - 1):
                tr = {"obs": states[k], "action": f"step_{k}", "next_obs": states[k + 1],
                      "goal": goal, "progress": (k + 1) / (T - 1), "traj_id": 100_000 + i,
                      "done": k + 1 == T - 1}
                ftr.write(json.dumps(tr, ensure_ascii=False) + "\n")
                n_train_tr += 1
            n_train_ep += 1

    n_succ_val = sum(1 for r in (json.loads(l) for l in
                     val_path.read_text(encoding="utf-8").splitlines()) if r["success"])
    print(f"train : {n_train_ep} trajectoires → {n_train_tr} transitions → {train_path}")
    print(f"val   : {n_val_ep} trajectoires held-out ({n_succ_val} succès / "
          f"{n_val_ep - n_succ_val} échecs) → {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
