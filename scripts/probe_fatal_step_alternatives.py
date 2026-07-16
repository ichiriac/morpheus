#!/usr/bin/env python
"""Au PAS FATAL, le candidate set offrait-il une porte de sortie — et le ranker l'aurait-il prise ?

Pourquoi ce script existe (2026-07-16) : le mode d'échec dominant de τ²-retail est l'**écriture
prématurée fausse** (12/29 échecs, cf. `decompose_failures.py`). Le domaine est IRRÉVERSIBLE
(`policy.md:84/110/118/130`) : une mutation fausse RÉUSSIE fait sortir la commande de l'état
requis et FORCLÔT définitivement l'écriture experte. Le « marché adressable » de 3/39 mesuré par
`replay_ranker_offline.py` ne compte QUE les épisodes où l'experte a été proposée sans être
exécutée — il ne dit rien des 12.

La vraie question sur les 12 est autre : **au tour où l'agent a tiré l'écriture fatale, le
candidate set contenait-il une action NON fatale ?**

  · S'il contenait un `respond_to_user` (confirmation), une LECTURE (statut, détails), ou
    l'écriture EXPERTE elle-même ⇒ un ranker avait de quoi éviter la catastrophe, et le marché
    adressable est PLUS GRAND que 3.
  · S'il ne contenait que des variantes de l'écriture fatale ⇒ le ranker est définitivement hors
    jeu : aucun classement ne sauve un set où tout est fatal. C'est le PROPOSEUR.

Deux questions distinctes, mesurées séparément (ne pas les confondre) :
  Q1 — PROPOSEUR : une sortie existait-elle dans le set ?      → borne le marché
  Q2 — RANKER    : le JEPA l'aurait-il élue ?                  → dit s'il sait la saisir

PAS FATAL = le PREMIER tour où une mutation NON experte RÉUSSIT (`tool_error=False`). Avant, rien
n'est perdu ; après, l'épisode est déjà condamné — seul le premier compte.

NON FATAL = tout candidat qui n'est pas une mutation (lecture / `respond_to_user`), OU qui est
l'écriture experte exacte. Tout le reste est une mutation fausse de plus.

Usage :
  python scripts/probe_fatal_step_alternatives.py --episodes runs/retail74_baseline/episodes.jsonl
  # --checkpoint <jepa.pt> pour AUSSI répondre à Q2 (charge torch + MiniLM ; sans lui, Q1 seule)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from decompose_failures import WRITE, is_write, norm, parse_action, same_action  # noqa: E402

DIALOGUE = "respond_to_user"


def classify(cand: str, exp_writes: list[tuple[str, dict]]) -> str:
    """'experte' | 'dialogue' | 'lecture' | 'MUTATION FAUSSE' — le vocabulaire du pas fatal."""
    pc = parse_action(cand)
    if pc is None:
        return "illisible"
    tool = pc[0]
    if any(same_action(pc, n, a) for n, a in exp_writes):
        return "experte"
    if tool == DIALOGUE:
        return "dialogue"
    if is_write(tool):
        return "MUTATION FAUSSE"
    return "lecture"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--manifest", default="data/benchmarks/retail_nojudge.json")
    p.add_argument("--tasks-json",
                   default="/workspace/tau2-bench/data/tau2/domains/retail/tasks.json")
    p.add_argument("--checkpoint", default=None,
                   help="jepa.pt → répond AUSSI à Q2 (le ranker aurait-il élu la sortie ?)")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    tasks = {str(t["id"]): t for t in json.loads(Path(a.tasks_json).read_text(encoding="utf-8"))}
    want = [str(i) for i in json.loads(Path(a.manifest).read_text())["task_ids"]]
    eps = [json.loads(l) for l in open(a.episodes, encoding="utf-8")]

    wm = None
    if a.checkpoint:
        from morpheus.agents.jepa_world_model import JepaWorldModel
        wm = JepaWorldModel(a.checkpoint, device=a.device)
    from morpheus.orchestrator.types import Action, Observation, State

    rows = []
    for e in eps:
        if e["success"]:
            continue
        ec = tasks[want[e["task"]]].get("evaluation_criteria") or {}
        exp = [(x["name"], dict(x.get("arguments") or {})) for x in (ec.get("actions") or [])
               if x.get("requestor", "assistant") == "assistant"]
        exp_writes = [(n, ar) for n, ar in exp if is_write(n)]
        if not exp_writes:
            continue

        # PREMIER tour où une mutation NON experte réussit → le pas fatal.
        for idx, st in enumerate(e["trace"]):
            pc = parse_action(st["chosen"])
            if not pc or not is_write(pc[0]) or st["tool_error"]:
                continue
            if any(same_action(pc, n, ar) for n, ar in exp_writes):
                continue                                     # c'est l'experte : pas fatal
            kinds = [classify(c, exp_writes) for c in st["candidates"]]
            escapes = [i for i, k in enumerate(kinds) if k != "MUTATION FAUSSE"]
            r = {"task": e["task"], "turn": st["turn"], "fatal_tool": pc[0],
                 "kinds": kinds, "escapes": escapes, "n_cands": len(st["candidates"]),
                 "ranker_i": None, "ranker_escapes": None}
            # Q2 : le ranker aurait-il élu une sortie ? (state.text du tour 1 non logué)
            if wm is not None and idx > 0 and len(st["candidates"]) > 1:
                state = State(goal=e["goal"], turn=st["turn"],
                              observation=Observation(text=e["trace"][idx - 1]["real_state"]))
                sc = []
                for c in st["candidates"]:
                    t, ar = parse_action(c) or (c.split("(")[0], {})
                    sc.append(wm.rollout(None, state, Action(tool=t, args=ar), [], 1)[0])
                r["ranker_i"] = max(range(len(sc)), key=lambda i: sc[i])   # = loop.py:117
                r["ranker_escapes"] = r["ranker_i"] in escapes
            rows.append(r)
            break

    print(f"\n{'='*96}\nLE PAS FATAL OFFRAIT-IL UNE PORTE DE SORTIE ?  — {a.episodes}\n{'='*96}")
    print(f"  épisodes avec une écriture prématurée fausse RÉUSSIE : {len(rows)}\n")
    print(f"  {'task':>4} {'tour':>4} {'écriture fatale':>34} | candidate set | sorties")
    print("  " + "-" * 92)
    for r in rows:
        pretty = " ".join(
            ("F" if k == "MUTATION FAUSSE" else "E" if k == "experte"
             else "D" if k == "dialogue" else "L" if k == "lecture" else "?")
            for k in r["kinds"])
        mark = ""
        if r["ranker_i"] is not None:
            mark = ("  ranker→[%d] %s" % (r["ranker_i"], "SORT ✅" if r["ranker_escapes"] else "FATAL ❌"))
        print(f"  {r['task']:>4} {r['turn']:>4} {r['fatal_tool'][:34]:>34} |    {pretty:<11}"
              f"| {len(r['escapes'])}/{r['n_cands']}{mark}")
    print("\n  légende : E=écriture experte · D=dialogue · L=lecture · F=mutation fausse")

    with_escape = [r for r in rows if r["escapes"]]
    print(f"\n{'='*96}\nQ1 — PROPOSEUR : une sortie existait-elle ?\n{'='*96}")
    print(f"  pas fatals avec ≥1 candidat NON fatal : {len(with_escape)}/{len(rows)}"
          f"  ({100*len(with_escape)/max(1,len(rows)):.0f}%)")
    n_exp = sum(1 for r in rows if "experte" in r["kinds"])
    print(f"  dont l'écriture EXPERTE était dans le set : {n_exp}/{len(rows)}")
    tot = sum(len(r["kinds"]) for r in rows)
    from collections import Counter
    c = Counter(k for r in rows for k in r["kinds"])
    print(f"  composition des sets fatals ({tot} candidats) : "
          + " · ".join(f"{k} {v} ({100*v/tot:.0f}%)" for k, v in c.most_common()))

    scored = [r for r in rows if r["ranker_i"] is not None]
    if scored:
        good = [r for r in scored if r["ranker_escapes"]]
        print(f"\n{'='*96}\nQ2 — RANKER : l'aurait-il élue ?\n{'='*96}")
        print(f"  pas fatals scorés : {len(scored)}")
        print(f"  le ranker élit une action NON fatale : {len(good)}/{len(scored)}"
              f"  ({100*len(good)/len(scored):.0f}%)")

        # ⚠️ LE BON NULL N'EST PAS LE HASARD UNIFORME. Cet échantillon est DÉFINI par « la
        # baseline a tiré le fatal » et la baseline prend toujours [0] ⇒ [0] est fatal aux N
        # pas, PAR CONSTRUCTION. Le « 0/N » de la baseline est donc la définition de
        # l'échantillon, pas un résultat — et TOUTE politique qui s'écarte de [0] gagne ici.
        # Le null honnête est donc « ne jamais prendre [0], tirer au hasard dans le reste ».
        uni = sum(len(r["escapes"]) / r["n_cands"] for r in scored)
        never0 = sum(len([i for i in r["escapes"] if i != 0]) / max(1, r["n_cands"] - 1)
                     for r in scored)
        print(f"\n  NULLS :")
        print(f"    hasard uniforme sur tout le set  : ~{uni:.1f}/{len(scored)}"
              f"  ({100*uni/len(scored):.0f}%)   ← MAUVAIS null (ignore la sélection)")
        print(f"    « jamais [0], hasard sur le reste » : ~{never0:.1f}/{len(scored)}"
              f"  ({100*never0/len(scored):.0f}%)   ← LE BON null")
        print(f"    ranker − null honnête = {len(good)-never0:+.1f} pas sur {len(scored)}")

        # Le test qui tranche : le ranker fuit-il [0] PARCE QU'il est fatal, ou toujours ?
        print(f"\n  DISCRIMINATION — P(le ranker s'écarte de [0]) :")
        dev = sum(1 for r in scored if r["ranker_i"] != 0)
        print(f"    aux pas FATALS ([0] mauvais ⇒ s'écarter est BON) : {dev}/{len(scored)}"
              f" = {100*dev/len(scored):.0f} %")
        print(f"    ⇒ à comparer au taux SUR TOUS LES PAS (replay_ranker_offline.py : 83,5 %)")
        print(f"      et aux pas où l'écriture experte EST en [0] (s'écarter est MAUVAIS : 92,9 %).")
        print(f"      Mesuré le 2026-07-16 : 83,5 / 87,5 / 92,9 — PLAT. Le ranker ne discrimine")
        print(f"      pas : il fuit [0] à taux constant, quoi que [0] soit. Son « 7/8 » ici est")
        print(f"      un artefact de sélection, même famille que le filtre `cand != chosen`.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
