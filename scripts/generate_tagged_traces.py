#!/usr/bin/env python
"""Génère des trajectoires τ² TAGUÉES PAR CONSTRUCTION pour le routeur de surprise (Phase 4).

Le label ERREUR/NOUVEAUTÉ de chaque pas tagué vient de l'INTERVENTION contrôlée injectée
dans le rejeu des actions de référence — PAS d'une rubrique sur les features observables.
C'est ce qui casse la circularité du lot v1 (data/annotations/README.md : 108/109 labels
déductibles de tool_error / is_user_turn / repeated_tool) : le générateur construit DEUX
PAIRES MINIMALES observationnellement indiscernables, que seule l'intervention sépare —

  PAIRE A (outil OK, pas de répétition, pas de dialogue) :
    - synth_wrong_write  → ERROR   : action d'ÉCRITURE de la référence rejouée sur une
      entité ÉTRANGÈRE à la tâche (id valide d'une AUTRE tâche) qui RÉUSSIT. Le
      cohérent-mais-faux de la thèse (cf. retail_postfix#1 t15 : modify sur la mauvaise
      commande, aucun signal d'outil).
    - synth_wrong_read   → ERROR   : lecture d'une entité étrangère à la tâche (réussit).
    - synth_detour       → NOVELTY : lecture LÉGITIME hors-référence — pré-fetch d'une
      entité que la référence utilise PLUS TARD (donc appartenant à la tâche).
    - synth_ref_success  → NOVELTY : pas nominal de la référence experte (ancre).
    La rubrique prédit NOVELTY pour les quatre ⇒ elle RATE les deux ERROR.

  PAIRE B (même outil qu'au pas précédent, sans erreur préalable) :
    - synth_loop         → ERROR   : ré-exécution VERBATIM de la lecture précédente
      (aucune info nouvelle — boucle hors transfer, casse la monoculture du lot v1).
    - synth_legit_repeat → NOVELTY : 2e appel consécutif du même outil avec d'AUTRES args
      DANS la référence experte (progrès légitime, ex. get_product_details ×2).
    La rubrique prédit ERROR pour les deux ⇒ elle RATE le NOVELTY.

  Ancre restante : synth_ref_error → ERROR (erreur d'outil naturelle du rejeu).

GARDES DE PURETÉ (vérifiées à la génération, sinon l'épisode-variante est écarté) :
  wrong_* / detour : pas de tool_error au pas tagué ET outil ≠ outil du pas précédent
  (repeated_tool=False) ; loop : le pas précédent est une lecture SANS erreur (repeated_tool
  =True) ; legit_repeat : les deux pas de la paire réussissent, args différents.

La divergence δ de chaque pas est calculée avec le JepaWorldModel (--checkpoint) :
δ_t = divergence(P(proj(E(s_{t-1})), enc(a_t)), s_t) — exactement le régime JEPA-WM live
(loop.py + jepa_world_model.py). δ_1 = 0.0 (« non mesuré », convention features.py : pas
d'observation d'ouverture dans un rejeu outil-seul).

Splits SANS FUITE : découpe PAR TÂCHE (seed fixe) ; les tâches des trajectoires déjà
annotées (lot v1) sont FORCÉES côté train (--force-train-from). airline = jeu d'essai
hors-domaine complet (--split all).

Sortie (versionnable) :
  <out-root>/synth_<domaine>_<split>/episodes.jsonl   — traces au format runner
  <out-root>/synth_<domaine>_<split>/labels.jsonl     — {episode, turn, label, rationale,
                                                          chosen, evidence, annotator}
Usage :
  python scripts/generate_tagged_traces.py --domain retail --split train \
      --checkpoint checkpoints/jepa_tau2_align/jepa.pt \
      --force-train-from data/annotations/trajectories/retail_postfix/episodes.jsonl
  python scripts/generate_tagged_traces.py --domain retail --split test  [idem]
  python scripts/generate_tagged_traces.py --domain airline --split all  [idem]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from morpheus.envs.tau2_adapter import _build_goal, _looks_like_error  # noqa: E402
from morpheus.orchestrator.types import Action                         # noqa: E402

ANNOTATOR = "construction"          # le label vient de l'intervention, pas d'un juge
READ_PREFIXES = ("get_", "find_", "list_", "search_", "calculate", "check_", "lookup_")
MAX_SWAP_ATTEMPTS = 8               # essais d'entité étrangère par variante
MIN_ACTIONS = 3                     # tâches trop courtes = pas de contexte pour injecter


def _quiet_tau2() -> None:
    try:
        from loguru import logger

        logger.disable("tau2")
    except Exception:
        pass


def _is_read(name: str) -> bool:
    return name.startswith(READ_PREFIXES)


def _action_str(name: str, args: dict) -> str:
    """Format IDENTIQUE au live : str(Action) de orchestrator/types.py (parité KB/mémoire)."""
    return str(Action(tool=name, args=dict(args or {})))


def _id_args(args: dict) -> list[str]:
    """Arguments porteurs d'identité d'entité, par ordre de préférence de swap :
    entité principale (order/reservation) → autres *_id → listes *_ids."""
    keys = list((args or {}).keys())
    primary = [k for k in keys if k in ("order_id", "reservation_id")]
    scalar = [k for k in keys if k.endswith("_id") and k not in primary]
    lists = [k for k in keys if k.endswith("_ids") and isinstance(args[k], list) and args[k]]
    return primary + scalar + lists


def _collect_pools(tasks) -> tuple[dict[str, list], dict[str, set]]:
    """(pool global {nom_arg → valeurs}, univers PAR TÂCHE {task_id → toutes valeurs d'id}).
    Une entité est « étrangère » à une tâche si sa valeur n'apparaît dans AUCUNE action de
    référence de cette tâche — c'est LA définition du label wrong_* (par construction)."""
    pool: dict[str, list] = {}
    own: dict[str, set] = {}
    for t in tasks:
        tid = str(t.id)
        own.setdefault(tid, set())
        for a in t.evaluation_criteria.actions or []:
            for k, v in (a.arguments or {}).items():
                if not (k.endswith("_id") or k.endswith("_ids")):
                    continue
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    own[tid].add(x)
                    bucket = pool.setdefault(k, [])
                    if x not in bucket:
                        bucket.append(x)
    return pool, own


def _foreign_values(pool: dict, own: set, arg: str, rng: random.Random) -> list:
    """Valeurs valides ailleurs dans le domaine mais ÉTRANGÈRES à la tâche courante."""
    base = arg[:-1] if arg.endswith("_ids") else arg     # item_ids → item_id (pool scalaire)
    cands = [v for v in pool.get(arg, []) + pool.get(base, []) if v not in own]
    rng.shuffle(cands)
    return cands[:MAX_SWAP_ATTEMPTS]


def _replay_actions(env_constructor, specs: list[tuple[str, dict]]) -> tuple[list[str], list[bool]]:
    """Exécute une séquence (nom, args) contre un env FRAIS ; renvoie (états, erreurs).
    Même mécanique que scripts/replay_reference_trajectories.py (env.get_response direct)."""
    from tau2.data_model.message import ToolCall

    env = env_constructor(solo_mode=False)
    states: list[str] = []
    errors: list[bool] = []
    for i, (name, args) in enumerate(specs):
        tc = ToolCall(id=f"c{i}", name=name, arguments=dict(args or {}), requestor="assistant")
        tm = env.get_response(tc)
        content = tm.content if isinstance(tm.content, str) else json.dumps(tm.content, default=str)
        states.append(content)
        errors.append(bool(getattr(tm, "error", False)) or _looks_like_error(content))
    return states, errors


def _episode(run: str, task_id: str, goal: str, specs, states, errors, wm) -> dict:
    """Trace au format runner (episodes.jsonl) avec δ JEPA canonique par pas."""
    trace = []
    chosen_strs = [_action_str(n, a) for n, a in specs]
    for i, (chosen, real, err) in enumerate(zip(chosen_strs, states, errors)):
        if i == 0 or wm is None:
            delta = 0.0                                   # pas d'état précédent → « non mesuré »
        else:
            zhat = wm._predict_latent(states[i - 1], chosen)
            delta = wm.divergence(zhat, real)
        trace.append({
            "turn": i + 1, "candidates": [chosen], "chosen": chosen,
            "predicted_state": None, "real_state": real, "divergence": delta,
            "surprise_route": None, "reward": 0.0, "done": i == len(specs) - 1,
            "retrieved_facts": [], "tool_error": err,
        })
    return {"goal": goal, "success": False, "turns": len(trace), "total_reward": 0.0,
            "required_turns": len(specs), "task": task_id, "trace": trace,
            "synthetic": True, "run": run}


def _label(run: str, task_id: str, turn: int, label: str, rationale: str,
           chosen: str, evidence: str) -> dict:
    return {"episode": f"{run}#{task_id}", "turn": turn, "label": label,
            "rationale": rationale, "chosen": chosen, "evidence": evidence[:200],
            "annotator": ANNOTATOR, "annotated_on": "2026-07-14"}


def generate_for_task(task, pool, own_ids, env_constructor, goal, run, wm,
                      rng: random.Random) -> tuple[list[dict], list[dict], dict]:
    """Tous les épisodes-variantes d'une tâche. Renvoie (épisodes, labels, stats)."""
    ref = [(a.name, dict(a.arguments or {})) for a in task.evaluation_criteria.actions or []]
    tid = str(task.id)
    episodes: list[dict] = []
    labels: list[dict] = []
    stats: dict[str, int] = {}

    def emit(kind: str) -> None:
        stats[kind] = stats.get(kind, 0) + 1

    # ---------- épisode 0 : rejeu NOMINAL (ancres + legit_repeat naturels) ----------
    states, errors = _replay_actions(env_constructor, ref)
    ep_run = run
    ep = _episode(ep_run, tid, goal, ref, states, errors, wm)
    episodes.append(ep)
    # legit_repeat : 2e appel consécutif même outil, args ≠, les deux SANS erreur.
    # C'est la référence EXPERTE : ce pas est du progrès légitime PAR CONSTRUCTION —
    # la rubrique (repeated_tool → ERREUR) le classerait faux.
    tagged_legit = False
    for i in range(1, len(ref)):
        if (not tagged_legit and ref[i][0] == ref[i - 1][0] and ref[i][1] != ref[i - 1][1]
                and not errors[i] and not errors[i - 1]):
            labels.append(_label(ep_run, tid, i + 1, "NOVELTY", "synth_legit_repeat",
                                 _action_str(*ref[i]), states[i]))
            emit("legit_repeat")
            tagged_legit = True
    # ancre tool_success : premier pas sans erreur NI répétition (signature paire A)
    for i in range(len(ref)):
        if not errors[i] and (i == 0 or ref[i][0] != ref[i - 1][0]):
            labels.append(_label(ep_run, tid, i + 1, "NOVELTY", "synth_ref_success",
                                 _action_str(*ref[i]), states[i]))
            emit("ref_success")
            break
    # ancre tool_error naturelle (une par tâche max)
    for i in range(len(ref)):
        if errors[i]:
            labels.append(_label(ep_run, tid, i + 1, "ERROR", "synth_ref_error",
                                 _action_str(*ref[i]), states[i]))
            emit("ref_error")
            break

    if len(ref) < MIN_ACTIONS:
        return episodes, labels, stats

    # ---------- variante wrong_write / wrong_read : entité ÉTRANGÈRE qui réussit ----------
    # Priorité aux ÉCRITURES (le cohérent-mais-faux de la thèse), en partant de la fin
    # (l'action la plus conséquente) ; repli sur une lecture si aucune écriture ne passe.
    def try_wrong(indices: list[int], rationale: str) -> bool:
        for j in indices:
            name, args = ref[j]
            if j > 0 and ref[j - 1][0] == name:
                continue                                   # garde : repeated_tool doit rester False
            for arg in _id_args(args):
                for foreign in _foreign_values(pool, own_ids, arg, rng):
                    new_args = dict(args)
                    if arg.endswith("_ids"):
                        lst = list(new_args[arg])
                        lst[0] = foreign
                        new_args[arg] = lst
                    else:
                        new_args[arg] = foreign
                    # test ÉCONOME : préfixe + swap seulement ; rejeu complet si le swap passe
                    probe_specs = ref[:j] + [(name, new_args)]
                    _, e_probe = _replay_actions(env_constructor, probe_specs)
                    if e_probe[j]:
                        continue                           # l'entité étrangère a erré → pas « cohérent »
                    specs = ref[:j] + [(name, new_args)] + ref[j + 1:]
                    s2, e2 = _replay_actions(env_constructor, specs)
                    if e2[j]:
                        continue                           # (ne devrait pas arriver : même préfixe)
                    ep2 = _episode(f"{run}", f"{tid}.{rationale.removeprefix('synth_')}",
                                   goal, specs, s2, e2, wm)
                    episodes.append(ep2)
                    labels.append(_label(run, ep2["task"], j + 1, "ERROR", rationale,
                                         _action_str(name, new_args), s2[j]))
                    emit(rationale.removeprefix("synth_"))
                    return True
        return False

    writes = [j for j in range(1, len(ref)) if not _is_read(ref[j][0]) and _id_args(ref[j][1])]
    reads = [j for j in range(1, len(ref)) if _is_read(ref[j][0]) and _id_args(ref[j][1])]
    if not try_wrong(list(reversed(writes)), "synth_wrong_write"):
        try_wrong(reads, "synth_wrong_read")
    elif reads:
        try_wrong(reads, "synth_wrong_read")               # les deux si possible

    # ---------- variante detour : PRÉ-FETCH d'une entité utilisée PLUS TARD ----------
    # k = insertion après ref[k-1] ; le détour duplique une LECTURE future (entité de la
    # tâche ⇒ légitime par construction), outil ≠ outil du pas précédent (garde paire A).
    k = max(1, len(ref) // 2)
    detour = None
    for j in range(len(ref) - 1, k - 1, -1):               # la plus tardive d'abord
        nm, ag = ref[j]
        if _is_read(nm) and _id_args(ag) and nm != ref[k - 1][0]:
            detour = (nm, dict(ag))
            break
    if detour is not None:
        specs = ref[:k] + [detour] + ref[k:]
        s3, e3 = _replay_actions(env_constructor, specs)
        if not e3[k]:
            ep3 = _episode(run, f"{tid}.detour", goal, specs, s3, e3, wm)
            episodes.append(ep3)
            labels.append(_label(run, ep3["task"], k + 1, "NOVELTY", "synth_detour",
                                 _action_str(*detour), s3[k]))
            emit("detour")

    # ---------- variante loop : ré-exécution VERBATIM de la lecture précédente ----------
    loop_j = None
    for j in range(len(ref)):
        if _is_read(ref[j][0]) and not errors[j] and (j + 1 >= len(ref) or ref[j + 1][0] != ref[j][0]):
            loop_j = j
            break
    if loop_j is not None:
        specs = ref[:loop_j + 1] + [ref[loop_j]] + ref[loop_j + 1:]
        s4, e4 = _replay_actions(env_constructor, specs)
        if not e4[loop_j + 1]:
            ep4 = _episode(run, f"{tid}.loop", goal, specs, s4, e4, wm)
            episodes.append(ep4)
            labels.append(_label(run, ep4["task"], loop_j + 2, "ERROR", "synth_loop",
                                 _action_str(*ref[loop_j]), s4[loop_j + 1]))
            emit("loop")

    return episodes, labels, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Traces τ² taguées par construction (Phase 4).")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--split", choices=("train", "test", "all"), default="all")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="limiter le nb de tâches (0 = toutes)")
    ap.add_argument("--checkpoint", default="checkpoints/jepa_tau2_align/jepa.pt",
                    help="JEPA pour δ canonique (régime JEPA-WM live) ; '' = δ non mesuré")
    ap.add_argument("--out-root", default="data/annotations/trajectories")
    ap.add_argument("--force-train-from", nargs="*", default=[],
                    help="episodes.jsonl déjà annotés : leurs tâches sont FORCÉES côté train "
                         "(aucune fuite vers le jeu d'essai)")
    args = ap.parse_args(argv)

    _quiet_tau2()
    from tau2.registry import registry

    env_constructor = registry.get_env_constructor(args.domain)
    tasks = [t for t in registry.get_tasks_loader(args.domain)()
             if t.evaluation_criteria and t.evaluation_criteria.actions
             and all(a.requestor == "assistant" for a in t.evaluation_criteria.actions)]
    if args.limit:
        tasks = tasks[: args.limit]

    # split PAR TÂCHE, déterministe ; tâches déjà annotées forcées côté train
    forced_train: set[str] = set()
    for src in args.force_train_from:
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            if line.strip():
                forced_train.add(str(json.loads(line).get("task")))
    ids = sorted(str(t.id) for t in tasks)
    rng_split = random.Random(args.seed)
    shuffled = [i for i in ids if i not in forced_train]
    rng_split.shuffle(shuffled)
    n_test = round(len(ids) * args.test_frac)
    test_ids = set(shuffled[:n_test])
    if args.split == "train":
        tasks = [t for t in tasks if str(t.id) not in test_ids]
    elif args.split == "test":
        tasks = [t for t in tasks if str(t.id) in test_ids]

    wm = None
    if args.checkpoint:
        from morpheus.agents.jepa_world_model import JepaWorldModel

        wm = JepaWorldModel(args.checkpoint, device="auto")
        print(f"δ canonique via {args.checkpoint}")

    pool, own = _collect_pools([t for t in registry.get_tasks_loader(args.domain)()
                                if t.evaluation_criteria and t.evaluation_criteria.actions])
    goal = _build_goal(None, solo=False, domain=args.domain)
    run = f"synth_{args.domain}_{args.split}"
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    all_eps: list[dict] = []
    all_labels: list[dict] = []
    totals: dict[str, int] = {}
    for t in tasks:
        rng = random.Random(f"{args.domain}:{t.id}:{args.seed}")
        eps, labs, st = generate_for_task(
            t, pool, own.get(str(t.id), set()), env_constructor, goal, run, wm, rng)
        all_eps.extend(eps)
        all_labels.extend(labs)
        for k, v in st.items():
            totals[k] = totals.get(k, 0) + v
        print(f"  [{t.id}] {len(eps)} épisodes, {len(labs)} labels : "
              + ", ".join(f"{k}×{v}" for k, v in sorted(st.items())))

    (out_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in all_eps) + "\n", encoding="utf-8")
    (out_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(l, ensure_ascii=False) for l in all_labels) + "\n", encoding="utf-8")

    n_err = sum(1 for l in all_labels if l["label"] == "ERROR")
    print(f"\n✅ {out_dir}/  — {len(all_eps)} épisodes, {len(all_labels)} labels "
          f"({n_err} ERREUR / {len(all_labels) - n_err} NOUVEAUTÉ)")
    print("   par classe : " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    print(f"   split {args.split} : {len(tasks)} tâches (test_ids={len(test_ids)}, "
          f"forcées train={len(forced_train)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
