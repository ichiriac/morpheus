#!/usr/bin/env python
"""FABRIQUE DE NÉGATIFS DURS : des paires (écriture EXPERTE, écriture-IMPOSTEUR) à granularité
d'ARGUMENT, étiquetées par l'oracle τ² lui-même.

Pourquoi (établi 2026-07-15, cf. BENCHMARKS.md + probe_action_granularity.py) : le mode d'échec
dominant de Qwen au harnais propre est la FAUSSETÉ CONFIANTE — une écriture bien formée, qui
n'erre pas, dont UN argument est faux. `tool_error` est aveugle, `divergence` est aveugle (le
world-model prédit CORRECTEMENT le payload d'une action fausse mais valide : l'environnement
exécute fidèlement les mauvais ordres), et le chemin latent aussi (cos = 0.9997 entre deux
écritures qui ne diffèrent que par un ID de variante).

⚠️ FRONTIÈRE : ces paires nourrissent le **coût** (le Q : « était-ce la bonne action ? »), PAS le
prédicteur. Réentraîner le prédicteur là-dessus serait une faute — prédire la confirmation d'une
écriture fausse-mais-valide est facile ET correct. Le prédicteur doit rester un météorologue
honnête ; c'est le juge qu'on n'a jamais entraîné.

Corruptions (un seul argument à la fois, valeur VALIDE dans le domaine mais FAUSSE pour la tâche —
c'est ce qui en fait des négatifs DURS, pas des erreurs de format) :
  · `new_item_ids`  → une AUTRE variante du MÊME produit (le cas C1, le plus dur : deux IDs à dix
                      chiffres, cos MiniLM 0.9991)
  · `order_id`      → une AUTRE commande du MÊME utilisateur (le cas C2)
  · `item_ids`      → un AUTRE article de la même commande

Chaque imposteur est REJOUÉ contre l'évaluateur τ² officiel : on GARDE la paire seulement si
l'expert donne db_reward=1.0 ET l'imposteur db_reward<1.0. L'étiquette n'est pas supposée, elle
est vérifiée par l'oracle.

  python scripts/build_write_negatives.py --domain retail --limit 40
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

WRITE = ("modify", "cancel", "return", "exchange", "update", "create", "delete")


def _mute_tau2() -> None:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass


def _db(domain: str) -> dict:
    import tau2
    root = Path(tau2.__file__).resolve().parents[1]
    for p in (root / f"data/tau2/domains/{domain}/db.json",
              Path(f"/root/tau2-bench/data/tau2/domains/{domain}/db.json")):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"db.json introuvable pour le domaine {domain!r}")


def _variant_siblings(db: dict, item_id: str) -> list[str]:
    """Les autres variantes du MÊME produit — des substituts que le catalogue rend plausibles."""
    for prod in db["products"].values():
        if item_id in (prod.get("variants") or {}):
            return [v for v in prod["variants"] if v != item_id]
    return []


def _user_other_orders(db: dict, order_id: str) -> list[str]:
    """Les autres commandes du MÊME utilisateur (le cas `coherent_but_wrong` de la tâche 2)."""
    o = db["orders"].get(order_id)
    if not o:
        return []
    uid = o.get("user_id")
    return [k for k, v in db["orders"].items() if v.get("user_id") == uid and k != order_id]


def _corruptions(db: dict, act) -> list[tuple[str, str, dict]]:
    """(nom_de_la_corruption, argument_touché, arguments_corrompus). Un seul argument change."""
    args = dict(act.arguments or {})
    out: list[tuple[str, str, dict]] = []

    if isinstance(args.get("new_item_ids"), list) and args["new_item_ids"]:
        orig = args["new_item_ids"][0]
        sib = _variant_siblings(db, orig)
        if sib:
            a = copy.deepcopy(args)
            a["new_item_ids"] = [sib[0]] + list(args["new_item_ids"][1:])
            out.append(("variante_swap", "new_item_ids", a))

    if isinstance(args.get("order_id"), str):
        others = _user_other_orders(db, args["order_id"])
        if others:
            a = copy.deepcopy(args)
            a["order_id"] = others[0]
            out.append(("commande_swap", "order_id", a))

    if isinstance(args.get("item_ids"), list) and len(args["item_ids"]) >= 1:
        o = db["orders"].get(args.get("order_id") or "")
        if o:
            pool = [i["item_id"] for i in o.get("items", [])
                    if i["item_id"] not in args["item_ids"]]
            if pool:
                a = copy.deepcopy(args)
                a["item_ids"] = [pool[0]] + list(args["item_ids"][1:])
                out.append(("article_swap", "item_ids", a))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--limit", type=int, default=None, help="nb de tâches à traiter")
    ap.add_argument("--out", default="data/tau2_replay/retail_write_negatives.jsonl")
    args = ap.parse_args(argv)

    _mute_tau2()
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.registry import registry

    db = _db(args.domain)
    tasks = registry.get_tasks_loader(args.domain)()
    env_ctor = registry.get_env_constructor(args.domain)
    if args.limit:
        tasks = tasks[: args.limit]

    def replay(task, actions) -> float | None:
        """Rejoue `actions` contre un env FRAIS et renvoie le db_reward de l'évaluateur officiel."""
        env = env_ctor(solo_mode=False)
        traj = []
        for i, a in enumerate(actions):
            tc = ToolCall(id=f"c{i}", name=a.name, arguments=a.arguments or {},
                          requestor=a.requestor)
            tm = env.get_response(tc)
            traj.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
            traj.append(tm)
        try:
            ri = EnvironmentEvaluator.calculate_reward(env_ctor, task, list(traj), solo_mode=False)
            return float(ri.reward)
        except Exception:
            return None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = n_tasks = n_rejected = 0
    by_kind: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as f:
        for t in tasks:
            acts = list(t.evaluation_criteria.actions or [])
            writes = [a for a in acts if any(k in a.name for k in WRITE)]
            if len(writes) != 1:       # v0 : une seule écriture ⇒ attribution non ambiguë
                continue
            w = writes[0]
            base = replay(t, acts)
            if base is None or base < 1.0:
                continue               # la référence elle-même doit valoir 1.0, sinon on ne sait rien
            n_tasks += 1

            for kind, argname, bad_args in _corruptions(db, w):
                corrupted = [copy.deepcopy(a) for a in acts]
                for a in corrupted:
                    if a.name == w.name and (a.arguments or {}) == (w.arguments or {}):
                        a.arguments = bad_args
                        break
                r = replay(t, corrupted)
                if r is None or r >= 1.0:
                    # L'oracle dit que l'imposteur RÉUSSIT quand même ⇒ ce n'est pas un négatif.
                    n_rejected += 1
                    continue
                f.write(json.dumps({
                    "task_id": str(t.id), "tool": w.name, "kind": kind, "arg": argname,
                    "expert_args": w.arguments, "impostor_args": bad_args,
                    "db_expert": base, "db_impostor": r,
                    "reads": [{"name": a.name, "arguments": a.arguments}
                              for a in acts if a is not w],
                }, ensure_ascii=False) + "\n")
                n_pairs += 1
                by_kind[kind] = by_kind.get(kind, 0) + 1
            print(f"  tâche {t.id}: {n_pairs} paires cumulées", file=sys.stderr)

    print(f"\n{n_pairs} paires vérifiées par l'oracle (sur {n_tasks} tâches à écriture unique) → {out}")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:<16} {v:>4}")
    print(f"  rejetées (l'imposteur réussissait quand même, db≥1.0) : {n_rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
