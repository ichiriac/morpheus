#!/usr/bin/env python
"""CONTRÔLE A/B — l'ANCIEN _SYS vs le NOUVEAU, à instrument CONSTANT, avec les sorties BRUTES.

Pourquoi ce script existe. Le test d'entrée a rendu 1/4 avec le _SYS corrigé (60180bb). On
voudrait conclure « la correction a nui », mais on ne peut pas : le 3/4 de référence était une
PRÉDICTION, jamais une mesure, et le script qui devait la tester est mort non commité. « 1/4 »
n'est donc attribuable à rien — ni au prompt, ni à l'instrument.

Ce script isole LA variable : mêmes cas (`HELD_OUT` importé, pas recopié), même code, même
modèle, même température, même db ; seul `_SYS` change.

Deux garde-fous que le score seul ne donne pas :
  1. LE BRUT EST AFFICHÉ. Un parse raté (`VERDICT:` absent → None) et un vrai INVALIDE se
     ressemblent dans un score et pas du tout dans une conclusion. On regarde avant d'interpréter.
  2. LA CONSTANTE EST RAPPELÉE. 2 cas attendent VALIDE, 2 INVALIDE ⇒ toute réponse constante
     score 2/4. Un 1/4 est SOUS le tampon-encreur : ça demande une explication.

  python scripts/probe_verifier_prompt_ab.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_verifier_zeroshot as P  # noqa: E402  (l'instrument, importé tel quel)

# ─────────────────────────────────────────────────────────────────────────────────────────
# ANCIEN _SYS — copie VERBATIM de scripts/probe_verifier_zeroshot.py à 79eba98, l'état dans
# lequel la sonde a réellement tourné hier (verdict « CONTAMINÉ »). Vérifiable :
#   git show 79eba98:scripts/probe_verifier_zeroshot.py | sed -n '/^_SYS = ($/,/^)$/p'
# ─────────────────────────────────────────────────────────────────────────────────────────
SYS_OLD = (
    "Tu es un VÉRIFICATEUR d'actions d'agent de service client (domaine retail). On te donne : la "
    "demande de l'utilisateur, les données du catalogue et des commandes, et UNE action d'écriture "
    "que l'agent s'apprête à exécuter.\n"
    "Ta tâche : vérifier CHAQUE argument de l'action contre la demande et les données. L'action "
    "est-elle celle que l'utilisateur a demandée, sur les BONNES entités, avec les BONS "
    "identifiants ?\n"
    "Une action peut être parfaitement bien formée, s'exécuter sans erreur, ET être fausse (mauvaise "
    "commande, mauvaise variante d'un produit, mauvais article). C'est exactement ce que tu dois "
    "détecter.\n"
    "Réponds STRICTEMENT par une seule ligne : `VERDICT: VALIDE` ou `VERDICT: INVALIDE`, suivie "
    "d'une ligne `RAISON: <une phrase>`."
)

# NOUVEAU _SYS = celui de l'instrument courant. Importé, jamais recopié : si quelqu'un le modifie,
# ce contrôle suit automatiquement au lieu de comparer contre une copie périmée.
SYS_NEW = P._SYS

VARIANTS = [("ANCIEN (79eba98)", SYS_OLD), ("NOUVEAU (60180bb)", SYS_NEW)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen_tau2_retail74.yaml")
    ap.add_argument("--raw", action="store_true", help="afficher le BRUT intégral de chaque appel")
    args = ap.parse_args(argv)

    from tau2.registry import registry

    from morpheus.config import Config
    from morpheus.llm import build_llm

    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass

    cfg = Config.load(ROOT / args.config)
    cfg.policy.temperature = 0.0
    llm = build_llm(cfg.policy)
    db = P._db()
    tasks = {str(t.id): t for t in registry.get_tasks_loader("retail")()}

    def request_of(tid: str) -> str:
        us = tasks[tid].user_scenario
        return str(getattr(us, "instructions", None) or us)

    print("=" * 88)
    print("CONTRÔLE A/B — ancien _SYS vs nouveau, MÊME instrument, MÊMES cas (HELD_OUT importé)")
    print(f"modèle {cfg.policy.model} | température 0.0 | {len(P.HELD_OUT)} cas")
    print("=" * 88)

    scores: dict[str, int] = {}
    verdicts: dict[str, list] = {}

    for vname, vsys in VARIANTS:
        print("\n" + "█" * 88)
        print(f"█ VARIANTE : {vname}")
        print("█" * 88)
        n_ok = 0
        n_unparsed = 0
        seq = []
        for label, tid, tool, a, expect_valid in P.HELD_OUT:
            got, why, raw = P._ask(llm, request_of(tid), P._catalog_for(db, a), tool, a,
                                   sys_prompt=vsys)
            ok = (got == expect_valid)
            n_ok += ok
            n_unparsed += (got is None)
            seq.append(got)
            v = {True: "VALIDE", False: "INVALIDE", None: "IMPARSABLE ⚠️"}[got]
            print(f"\n  {'✅' if ok else '❌'} {label}")
            print(f"      obtenu {v}  | attendu {'VALIDE' if expect_valid else 'INVALIDE'}")
            print(f"      ─── SORTIE BRUTE ({len(raw)} car.) " + "─" * 34)
            for line in (raw if args.raw else raw[:600]).splitlines() or ["(VIDE)"]:
                print(f"      │ {line}")
            if not args.raw and len(raw) > 600:
                print(f"      │ … (+{len(raw)-600} car. — relancer avec --raw)")
            print("      " + "─" * 60)
        scores[vname] = n_ok
        verdicts[vname] = seq
        print(f"\n  ➜ {vname} : test d'entrée {n_ok}/4"
              f"{f'  ⚠️ {n_unparsed} IMPARSABLE(S)' if n_unparsed else '  (tout parse)'}")
        print(f"     séquence : {['VALIDE' if s else 'INVALIDE' if s is False else 'None' for s in seq]}")

    # ---------- lecture ----------
    old, new = scores["ANCIEN (79eba98)"], scores["NOUVEAU (60180bb)"]
    print("\n" + "=" * 88)
    print("LECTURE (les interprétations étaient enregistrées AVANT de voir ces chiffres)")
    print("=" * 88)
    print(f"  ANCIEN  : {old}/4")
    print(f"  NOUVEAU : {new}/4")
    print(f"  constante (tout-VALIDE ou tout-INVALIDE) : 2/4  ← la vraie référence, pas 0")
    print()
    if old == 3:
        print("  ⇒ ANCIEN=3/4 : la correction du prompt est NUISIBLE (3→{}).".format(new))
        print("    La prédiction « reste à 3/4 » est falsifiée dans la direction non envisagée :")
        print("    on prédisait l'immobilité, on mesure une DÉGRADATION. La spec « arguments seuls,")
        print("    complétude interdite » a cassé un vérificateur qui marchait mieux avant.")
    elif old == new:
        print(f"  ⇒ ANCIEN=NOUVEAU={old}/4 : le prompt n'explique RIEN. Si {old} != 3, alors le 3/4")
        print("    attendu était une propriété de l'instrument perdu — prédiction scellée contre un")
        print("    fantôme, donc INÉVALUABLE. Leçon : un instrument non commité rend ses résultats")
        print("    infalsifiables.")
    else:
        print(f"  ⇒ ANCIEN={old} vs NOUVEAU={new} : la direction attribue au PROMPT.")
    print()
    print("  ⚠️ CE QUE CETTE MESURE NE DÉCIDE PAS : elle ferme une prédiction, elle ne ressuscite")
    print("     pas le Q. Même à 4/4 le Q resterait mort — 0/21 rattrapable sur épisodes réels,")
    print("     contre-productif sur 4, marché adressable inexistant. 4 cas choisis à la main ne")
    print("     pèsent rien contre ça. Ce qu'on achète ici, c'est la CALIBRATION des prédictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
