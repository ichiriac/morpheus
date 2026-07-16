#!/usr/bin/env python
"""Décompose les ÉCHECS d'un run τ²-retail par mode, en confrontant les écritures de l'agent
aux `evaluation_criteria.actions` de référence.

Pourquoi ce script existe (2026-07-16) : la décomposition « à la main » d'un run est exactement
l'endroit où l'on fabrique des artefacts d'instrument. Deux se sont produits le même soir :

  1. Un filtre `key(cand) != key(chosen)` pour « lister les candidats non choisis » exclut par
     CONSTRUCTION tous les candidates[0], puisque la baseline choisit toujours [0]. Il produit
     mécaniquement « l'écriture experte n'est JAMAIS en [0] » — qui était la définition du
     filtre, pas une mesure. (Réel : elle y est dans 54 % des pas décisifs.)
  2. Compter une écriture experte comme « exécutée » sur la seule foi de `chosen`, SANS regarder
     `tool_error`, produit « 3 épisodes ont tout écrit et échoué quand même ». Réel : 0. Les
     trois avaient TENTÉ l'écriture experte et l'env l'avait REJETÉE — parce qu'une écriture
     fausse antérieure avait déjà fait sortir la commande de l'état requis.

⇒ La décomposition est un INSTRUMENT. Elle doit être versionnée et re-exécutable, pas retapée
   dans un heredoc. C'est aussi la condition pour chiffrer sa VARIANCE entre deux graines — or
   on sait que le mode d'échec lui-même change d'un run à l'autre à temperature 0.7 (les tâches
   0/1/5 ont donné C1/A/C2 sur un run et D/D/D sur le suivant).

MODES (une écriture = un outil dont le nom contient un verbe de mutation, cf. WRITE) :
  · aucune écriture tentée        — l'agent n'a jamais muté la DB
  · ÉCRITURE PRÉMATURÉE FAUSSE    — une mutation NON experte a RÉUSSI (tool_error=False).
                                    τ²-retail est IRRÉVERSIBLE (policy.md:84/110/118/130 : une
                                    commande ne se return/exchange/modify qu'une fois, et
                                    seulement depuis 'delivered'/'pending') ⇒ cette écriture
                                    peut FORCLORE définitivement l'écriture experte.
  · expert partiel / complet      — sous-ensemble / totalité des écritures expertes RÉUSSIES
  · pas d'écriture requise        — la tâche de référence n'en contient aucune

`--compare` prend un second run et chiffre la stabilité de la décomposition entre deux graines.

Usage :
  python scripts/decompose_failures.py --episodes runs/retail74_baseline/episodes.jsonl
  python scripts/decompose_failures.py --episodes runs/a/episodes.jsonl --compare runs/b/episodes.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

# Même liste que scripts/build_write_negatives.py:38 et scripts/replay_ranker_offline.py.
WRITE = ("modify", "cancel", "return", "exchange", "update", "create", "delete")

MODES = ["aucune écriture tentée", "ÉCRITURE PRÉMATURÉE FAUSSE (réussie)",
         "expert partiel", "expert complet", "pas d'écriture requise"]


def parse_action(s: str) -> tuple[str, dict] | None:
    """`tool(k='v')` → ('tool', {'k': 'v'}). `Action.__str__` (types.py:23) utilise `repr()` sur
    les valeurs ⇒ chaîne littéral-évaluable ⇒ `ast` la reparse sans exécuter quoi que ce soit."""
    try:
        node = ast.parse(s.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        return node.func.id, {kw.arg: ast.literal_eval(kw.value)
                              for kw in node.keywords if kw.arg}
    except Exception:
        return None


def norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(sorted(norm(x) for x in v))
    if isinstance(v, dict):
        return tuple(sorted((k, norm(x)) for k, x in v.items()))
    return v


def same_action(cand: tuple[str, dict], name: str, args: dict) -> bool:
    """Égalité SÉMANTIQUE : même outil + mêmes arguments à l'ordre près (l'ordre des item_ids
    n'est pas sémantique côté τ²)."""
    return cand[0] == name and ({k: norm(v) for k, v in cand[1].items()}
                                == {k: norm(v) for k, v in args.items()})


def is_write(tool: str) -> bool:
    return any(k in tool for k in WRITE)


def decompose(episodes_path: str, tasks_json: str, manifest: str) -> dict:
    tasks = {str(t["id"]): t for t in json.loads(Path(tasks_json).read_text(encoding="utf-8"))}
    want = [str(i) for i in json.loads(Path(manifest).read_text())["task_ids"]]
    eps = [json.loads(l) for l in open(episodes_path, encoding="utf-8")]

    by_mode: dict[str, list[int]] = {m: [] for m in MODES}
    foreclosed: list[int] = []
    per_task: dict[int, str] = {}

    for e in eps:
        if e["success"]:
            continue
        ec = tasks[want[e["task"]]].get("evaluation_criteria") or {}
        exp = [(a["name"], dict(a.get("arguments") or {}))
               for a in (ec.get("actions") or [])
               if a.get("requestor", "assistant") == "assistant"]
        exp_writes = [(n, ar) for n, ar in exp if is_write(n)]
        if not exp_writes:
            by_mode["pas d'écriture requise"].append(e["task"])
            per_task[e["task"]] = "pas d'écriture requise"
            continue

        ok_exp = bad_ok = attempted = exp_rejected = 0
        for st in e["trace"]:
            pc = parse_action(st["chosen"])
            if not pc or not is_write(pc[0]):
                continue
            attempted += 1
            hit = any(same_action(pc, n, ar) for n, ar in exp_writes)
            if hit and not st["tool_error"]:
                ok_exp += 1
            elif hit and st["tool_error"]:
                exp_rejected += 1          # tentée MAIS rejetée par l'env
            elif not hit and not st["tool_error"]:
                bad_ok += 1                # mutation NON experte qui a RÉUSSI

        # L'écriture experte a été tentée et rejetée APRÈS qu'une écriture fausse ait réussi :
        # la mutation fausse a consommé la transition irréversible.
        if exp_rejected and bad_ok:
            foreclosed.append(e["task"])

        if attempted == 0:
            m = "aucune écriture tentée"
        elif bad_ok:
            m = "ÉCRITURE PRÉMATURÉE FAUSSE (réussie)"
        elif ok_exp >= len(exp_writes):
            m = "expert complet"
        else:
            m = "expert partiel"
        by_mode[m].append(e["task"])
        per_task[e["task"]] = m

    return {"n_episodes": len(eps), "n_success": sum(1 for e in eps if e["success"]),
            "n_fail": sum(1 for e in eps if not e["success"]),
            "by_mode": by_mode, "foreclosed": foreclosed, "per_task": per_task}


def report(d: dict, label: str) -> None:
    print(f"\n{'='*92}\nDÉCOMPOSITION DES ÉCHECS — {label}\n{'='*92}")
    print(f"  épisodes {d['n_episodes']} · succès {d['n_success']} "
          f"({100*d['n_success']/d['n_episodes']:.1f}%) · échecs {d['n_fail']}")
    print()
    for m in MODES:
        v = d["by_mode"][m]
        print(f"  {m:>38} : {len(v):>2}  → {sorted(v)}")
    print()
    print(f"  ⚠️  écriture fausse ayant FORECLOS l'écriture experte : {len(d['foreclosed'])}"
          f" → {sorted(d['foreclosed'])}")
    print(f"      (experte TENTÉE puis rejetée par l'env, APRÈS une mutation fausse réussie)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--compare", default=None, help="second run (autre graine) → stabilité")
    p.add_argument("--manifest", default="data/benchmarks/retail_nojudge.json")
    p.add_argument("--tasks-json",
                   default="/workspace/tau2-bench/data/tau2/domains/retail/tasks.json")
    p.add_argument("--out", default=None, help="JSON du résultat")
    a = p.parse_args()

    d = decompose(a.episodes, a.tasks_json, a.manifest)
    report(d, a.episodes)
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")

    if a.compare:
        d2 = decompose(a.compare, a.tasks_json, a.manifest)
        report(d2, a.compare)
        common = set(d["per_task"]) & set(d2["per_task"])
        same = [t for t in common if d["per_task"][t] == d2["per_task"][t]]
        print(f"\n{'='*92}\nSTABILITÉ DE LA DÉCOMPOSITION ENTRE LES DEUX RUNS\n{'='*92}")
        print(f"  tâches en échec dans les DEUX runs : {len(common)}")
        if common:
            print(f"  même mode d'échec : {len(same)}/{len(common)} "
                  f"({100*len(same)/len(common):.0f}%)")
            for t in sorted(common - set(same)):
                print(f"    task {t:>2} : {d['per_task'][t]:>38}  →  {d2['per_task'][t]}")
        print("\n  ⇒ concevoir une architecture contre une décomposition dont la stabilité")
        print("    n'est pas chiffrée, c'est concevoir contre un seul tirage.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
