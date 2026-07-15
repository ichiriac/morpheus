#!/usr/bin/env python
"""Construit le MANIFESTE du banc « sans juge » : les tâches τ² dont le reward est le `db` SEUL.

Pourquoi ce banc existe (établi le 2026-07-15, cf. BENCHMARKS.md) :
`tau2.evaluator.evaluator_nl_assertions.NLAssertionsEvaluator.calculate_reward` sort en
`return reward=1.0` AVANT tout appel LLM quand `nl_assertions` est vide. Pour ces tâches, la
composante NL vaut 1.0 par vérité vacue ⇒ `reward = db × 1 = db` : le protocole τ² OFFICIEL, sans
juge dans la boucle, sans distorsion. C'est un banc PROPRE.

C'est nécessaire parce que le juge NL, mesuré, ne discrimine pas : sur le seul appel réel du smoke
`retail_attrib` (assertion « there are 10 t-shirt options available »), verdict MET alors que
l'agent avait listé 9 options sans jamais dire « 10 ». Faux positif 1/1.

⚠️ BIAIS CONNU ET ASSUMÉ : les tâches sans assertions ont des consignes utilisateur plus courtes
(533 vs 643 caractères ; AUC de Mann-Whitney 0.339, z≈−2.8, p≈0.005) ⇒ le banc SOUS-ÉCHANTILLONNE
l'extraction dialogique et surestimera donc légèrement la compétence. Biais rapporté, dans ce
sens-là il est acceptable. Les tâches AVEC assertions restent une piste secondaire, flaggée
« juge non validé », jusqu'à ce qu'un juge recale un imposteur.

Le manifeste est la DÉFINITION du banc : versionné, tous les bras (baseline / world-model /
Sonnet 4.6 / Qwen-natif) tournent sur exactement cette liste, même seed.

  python scripts/build_nojudge_bench.py --domain retail
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--out", default=None, help="défaut : data/benchmarks/<domain>_nojudge.json")
    args = ap.parse_args(argv)

    from tau2.registry import registry

    tasks = registry.get_tasks_loader(args.domain)()
    vac, jug = [], []
    for t in tasks:
        ec = t.evaluation_criteria
        # La condition EXACTE de la sortie anticipée du juge (cf. calculate_reward) :
        # `evaluation_criteria is None` OU `not nl_assertions` ⇒ NL=1.0 sans appel LLM.
        (vac if (ec is None or not ec.nl_assertions) else jug).append(str(t.id))

    out = Path(args.out or f"data/benchmarks/{args.domain}_nojudge.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "domain": args.domain,
        "criterion": "evaluation_criteria is None or not nl_assertions "
                     "→ NLAssertionsEvaluator.calculate_reward sort en reward=1.0 sans appel LLM "
                     "⇒ reward = db × 1 = db (protocole τ² officiel, aucun juge dans la boucle)",
        "n_total": len(tasks),
        "n_selected": len(vac),
        "n_excluded_judged": len(jug),
        "known_bias": "les tâches sélectionnées ont des consignes utilisateur plus courtes "
                      "(533 vs 643 car., AUC 0.339, p≈0.005) ⇒ sous-échantillonne l'extraction "
                      "dialogique ⇒ surestime légèrement la compétence. Assumé et rapporté.",
        "task_ids": vac,
        "excluded_judged_task_ids": jug,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"banc « sans juge » {args.domain} : {len(vac)}/{len(tasks)} tâches → {out}")
    print(f"  exclues (nl_assertions non vide ⇒ juge appelé, non validé) : {len(jug)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
