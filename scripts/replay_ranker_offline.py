#!/usr/bin/env python
"""Rejoue HORS-LIGNE le classement du JEPA-WM sur les candidate sets DÉJÀ enregistrés.

Pourquoi ce script existe (2026-07-16) : le constat « l'écriture experte est dans le candidate
set, en [1..3], jamais en [0] — et la baseline prend [0] » redéfinit le module manquant comme un
**ranker**, c'est-à-dire exactement le world-model. Avant de payer des heures de GPU pour l'A/B,
on peut répondre à la question préalable — *le ranker préfère-t-il l'écriture à la relecture ?* —
sans AUCUN appel LLM :

    JepaWorldModel.rollout(policy, state, first, tools, horizon)   # horizon ignoré au-delà de 1
      z_goal = proj(E(state.goal))
      z_pred = predict_next(E(state.text), E(str(first)))
      score  = (cos(z_pred, z_goal) + 1) / 2

`policy` n'est jamais appelée à H=1 et `state.text` == `observation.text` (types.py) — or la trace
persiste `real_state` à chaque tour. Donc l'entrée du ranker est intégralement reconstructible :

    state.text(tour t) == trace[t-2].real_state          (t ≥ 2)
    state.text(tour 1) == observation du reset           → NON loguée (angle mort connu)

⇒ on rejoue le `max()` de loop.py:117 à l'identique (ties → plus petit index, comme `max`).

CE QUE ÇA MESURE (et pas plus) :
  · UPSIDE   — aux pas où l'écriture experte est dans le candidate set, le ranker l'élit-il ?
  · DOWNSIDE — sur les épisodes qui RÉUSSISSENT déjà, le ranker dévie-t-il de candidates[0] ?
               (le +26 points est une borne haute qui ignore ce risque de régression)
  · BRUIT    — l'étendue des scores DANS un candidate set. Si elle est dans le bruit, l'argmax
               n'est pas un choix, c'est un tirage. Référence : étendue intra-épisode 0.176
               (probe_tau2_goal_range.py) ; le dégénéré historique valait 0.0086.

CE QUE ÇA NE MESURE PAS : le contrefactuel. Dès que le ranker dévie au pas t, la suite de
l'épisode n'est plus celle enregistrée. Seul le PREMIER pas décisif de chaque épisode est un
rejeu honnête ; au-delà c'est de l'hypothétique. D'où `--first-decisive-only` (défaut) et le
report explicite des deux chiffres.

Usage :
  python scripts/replay_ranker_offline.py \
      --episodes runs/retail74_baseline/episodes.jsonl \
      --checkpoint checkpoints/jepa_tau2_align/jepa.pt \
      --manifest data/benchmarks/retail_nojudge.json --domain retail
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# backbone sentence-transformer en cache local (TODO Journal §4) : rester hors-ligne.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Même liste que scripts/build_write_negatives.py:38 — une écriture = une mutation de la DB.
WRITE = ("modify", "cancel", "return", "exchange", "update", "create", "delete")


def parse_action(s: str) -> tuple[str, dict] | None:
    """`tool(k='v', k2=['a'])` → ('tool', {'k': 'v', 'k2': ['a']}).

    `Action.__str__` utilise `repr()` sur les valeurs ⇒ la chaîne est littéral-évaluable, donc
    `ast` la reparse sans risque d'exécution (literal_eval sur les valeurs, pas eval).
    Renvoie None si la chaîne n'est pas parsable (candidat tronqué/malformé) — le compteur
    `unparsed` du rapport rend ces cas visibles au lieu de les avaler.
    """
    try:
        node = ast.parse(s.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        return node.func.id, args
    except Exception:
        return None


def norm(v):
    """Normalise pour comparer des arguments : listes → tuples triés (l'ordre des item_ids
    n'est pas sémantique côté τ²), scalaires inchangés."""
    if isinstance(v, (list, tuple)):
        return tuple(sorted(norm(x) for x in v))
    if isinstance(v, dict):
        return tuple(sorted((k, norm(x)) for k, x in v.items()))
    return v


def same_action(cand: tuple[str, dict], exp_name: str, exp_args: dict) -> bool:
    """Égalité SÉMANTIQUE (pas textuelle) : même outil + mêmes arguments à l'ordre près."""
    name, args = cand
    if name != exp_name:
        return False
    return {k: norm(v) for k, v in args.items()} == {k: norm(v) for k, v in exp_args.items()}


def load_expert(tasks_json: str, manifest_path: str) -> dict[int, list[tuple[str, dict]]]:
    """index du banc (= `task` dans episodes.jsonl) → actions expertes assistant.

    Lit `tasks.json` du domaine EN DIRECT plutôt que d'importer `tau2` : le paquet tire
    gymnasium & co dans le venv qui sert le vLLM en ce moment (TODO Journal §1/§2 — toute
    install qui touche la pile du serveur est un risque). Le JSON est la même source de
    vérité que `registry.get_tasks_loader`, et l'ordre du banc vient du MANIFESTE
    (tau2_adapter.py:368 : `tasks = [by_id[i] for i in want]`), pas de celui du domaine.
    """
    tasks = {str(t["id"]): t for t in json.loads(Path(tasks_json).read_text(encoding="utf-8"))}
    want = [str(i) for i in json.loads(Path(manifest_path).read_text())["task_ids"]]
    missing = [i for i in want if i not in tasks]
    if missing:
        raise SystemExit(f"ids du manifeste absents de {tasks_json} : {missing[:5]}")
    out: dict[int, list[tuple[str, dict]]] = {}
    for i, tid in enumerate(want):
        ec = tasks[tid].get("evaluation_criteria") or {}
        acts = ec.get("actions") or []
        out[i] = [(a["name"], dict(a.get("arguments") or {})) for a in acts
                  if a.get("requestor", "assistant") == "assistant"]
    return out


class _ActionText:
    """Action dont `str()` est imposé — sert à scorer le format d'ENTRAÎNEMENT sans modifier
    `types.Action` (le code de prod reste la référence de ce que loop.py fait vraiment)."""

    def __init__(self, tool: str, args: dict, text: str) -> None:
        self.tool, self.args, self._text = tool, args, text

    def __str__(self) -> str:
        return self._text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--checkpoint", default="checkpoints/jepa_tau2_align/jepa.pt")
    p.add_argument("--manifest", default="data/benchmarks/retail_nojudge.json")
    p.add_argument("--tasks-json",
                   default="/workspace/tau2-bench/data/tau2/domains/retail/tasks.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="JSONL détaillé (un objet par pas scoré)")
    p.add_argument("--action-format", choices=["live", "train"], default="live",
                   help="'live' = `Action.__str__` (types.py:23, ce que loop.py passe VRAIMENT au "
                        "world-model) ; 'train' = `name({json})` (replay_reference_trajectories.py:95, "
                        "le format sur lequel le prédicteur a été ENTRAÎNÉ). Les deux diffèrent — "
                        "cette option chiffre ce que coûte le mismatch.")
    a = p.parse_args()

    from morpheus.agents.jepa_world_model import JepaWorldModel
    from morpheus.orchestrator.types import Action, Observation, State

    wm = JepaWorldModel(a.checkpoint, device=a.device)
    expert = load_expert(a.tasks_json, a.manifest)
    eps = [json.loads(l) for l in open(a.episodes, encoding="utf-8")]

    rows: list[dict] = []
    unparsed = 0

    for e in eps:
        exp_all = expert.get(e["task"], [])
        exp_writes = [(n, ar) for n, ar in exp_all if any(k in n for k in WRITE)]
        trace = e["trace"]
        for idx, st in enumerate(trace):
            turn = st["turn"]
            if idx == 0:
                continue                       # state.text du tour 1 = reset obs, NON loguée
            state_text = trace[idx - 1]["real_state"]
            cands = st["candidates"]
            if len(cands) <= 1:
                continue                       # loop.py:114 court-circuite — rien à classer
            parsed = [parse_action(c) for c in cands]
            unparsed += sum(1 for x in parsed if x is None)

            state = State(goal=e["goal"], observation=Observation(text=state_text), turn=turn)
            scores = []
            for c, pc in zip(cands, parsed):
                tool, args = pc or (c.split("(")[0], {})
                if a.action_format == "train":
                    # Format d'ENTRAÎNEMENT du prédicteur (replay_reference_trajectories.py:95) :
                    # `name({json})`. `_ActionText` court-circuite `Action.__str__` pour que
                    # `rollout` reçoive exactement la chaîne voulue, sans toucher au code de prod.
                    txt = f"{tool}({json.dumps(args, ensure_ascii=False, default=str)})"
                    act = _ActionText(tool=tool, args=args, text=txt)
                else:
                    act = Action(tool=tool, args=args)
                sc, _ = wm.rollout(None, state, act, [], 1)
                scores.append(sc)
            # loop.py:117 à l'identique : `max` renvoie le PREMIER maximum (ties → index bas).
            best_i = max(range(len(cands)), key=lambda i: scores[i])

            # position(s) de l'écriture experte dans le candidate set
            exp_idx = [i for i, pc in enumerate(parsed)
                       if pc and any(same_action(pc, n, ar) for n, ar in exp_writes)]
            chosen_i = next((i for i, c in enumerate(cands) if c == st["chosen"]), None)

            rows.append({
                "task": e["task"], "success": e["success"], "turn": turn,
                "scores": [round(s, 6) for s in scores],
                "spread": round(max(scores) - min(scores), 6),
                "best_i": best_i, "chosen_i": chosen_i,
                "expert_write_idx": exp_idx,
                "ranker_picks_expert": bool(exp_idx) and best_i in exp_idx,
                "baseline_picks_expert": bool(exp_idx) and chosen_i in exp_idx,
                "cands": cands,
            })

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- rapport ----------------
    import statistics as stx

    spreads = [r["spread"] for r in rows]
    print(f"\n{'='*78}\nREJEU HORS-LIGNE DU RANKER — {a.episodes}\n{'='*78}")
    print(f"pas scorés (K>1, tour≥2) : {len(rows)}   · candidats non parsés : {unparsed}")
    print(f"épisodes : {len(eps)}  (succès {sum(1 for e in eps if e['success'])})")

    print(f"\n--- BRUIT : étendue des scores DANS un candidate set ---")
    if spreads:
        print(f"  médiane {stx.median(spreads):.4f} · moyenne {stx.mean(spreads):.4f} · "
              f"max {max(spreads):.4f} · min {min(spreads):.4f}")
        print(f"  référence : étendue intra-épisode 0.176 · dégénéré historique 0.0086")

    dec = [r for r in rows if r["expert_write_idx"]]
    print(f"\n--- UPSIDE : pas où l'écriture experte est dans le candidate set ---")
    print(f"  pas décisifs : {len(dec)}  sur {len(set(r['task'] for r in dec))} épisodes")
    if dec:
        hit = sum(1 for r in dec if r["ranker_picks_expert"])
        base = sum(1 for r in dec if r["baseline_picks_expert"])
        print(f"  ranker élit l'écriture experte   : {hit}/{len(dec)}  ({100*hit/len(dec):.0f}%)")
        print(f"  baseline élit l'écriture experte : {base}/{len(dec)}  ({100*base/len(dec):.0f}%)")
        # par épisode : le ranker sauve-t-il l'épisode à AU MOINS un pas ?
        by_ep: dict[int, bool] = {}
        for r in dec:
            by_ep[r["task"]] = by_ep.get(r["task"], False) or r["ranker_picks_expert"]
        print(f"  épisodes où le ranker élit l'écriture ≥1 fois : "
              f"{sum(by_ep.values())}/{len(by_ep)}   → {sorted(k for k,v in by_ep.items() if v)}")
        print(f"\n  détail (position experte → position élue par le ranker) :")
        for r in dec:
            flag = "✅" if r["ranker_picks_expert"] else "❌"
            print(f"    {flag} task {r['task']:>2} t{r['turn']:>2} | experte en {r['expert_write_idx']}"
                  f" | ranker → [{r['best_i']}] | baseline → [{r['chosen_i']}]"
                  f" | étendue {r['spread']:.4f}")

    succ = [r for r in rows if r["success"]]
    print(f"\n--- DOWNSIDE : déviation sur les épisodes qui RÉUSSISSENT déjà ---")
    if succ:
        dev = sum(1 for r in succ if r["best_i"] != r["chosen_i"])
        print(f"  pas scorés dans les succès : {len(succ)}")
        print(f"  le ranker DÉVIE de la baseline : {dev}/{len(succ)} ({100*dev/len(succ):.0f}%)")
        eps_dev = sorted(set(r["task"] for r in succ if r["best_i"] != r["chosen_i"]))
        print(f"  épisodes réussis touchés par ≥1 déviation : {len(eps_dev)}/"
              f"{len(set(r['task'] for r in succ))}  → {eps_dev}")
        print(f"  ⇒ autant de trajectoires dont l'issue n'est plus celle mesurée (risque de perte)")

    print(f"\n--- CONTRÔLE : le ranker a-t-il un biais de POSITION ? ---")
    from collections import Counter
    c = Counter(r["best_i"] for r in rows)
    print("  élus par le ranker :", dict(sorted(c.items())))
    cb = Counter(r["chosen_i"] for r in rows if r["chosen_i"] is not None)
    print("  élus par la baseline :", dict(sorted(cb.items())), " (attendu : tout en [0])")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
