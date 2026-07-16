#!/usr/bin/env python
"""SONDE : un Qwen-VÉRIFICATEUR zero-shot recale-t-il ce que le Qwen-GÉNÉRATEUR rate ?

L'expérience la moins chère du chantier « fonction de coût » : ZÉRO entraînement. Si un
vérificateur promptté discrimine les écritures fausses-mais-valides, le premier Q de morpheus est
un PROMPT. S'il tamponne, c'est la preuve qu'il faut un Q hybride (features exactes là où le monde
est exact : `==` sur les IDs, appartenance à la table d'options ; du flou seulement pour extraire
l'intention du dialogue).

Le pari : vérifier est plus facile que générer, et un vérificateur a un profil d'échec DIFFÉRENT
du générateur — c'est tout l'intérêt de l'asymétrie.

Deux jeux :
  1. FABRIQUE (scripts/build_write_negatives.py) : paires (écriture experte, imposteur à UN
     argument près), étiquetées par l'oracle τ² (db_expert=1.0, db_impostor<1.0).
  2. HELD-OUT — les 3 échecs RÉELS du smoke `retail_cap2500`, ceux de la taxonomie :
     C1 tâche 0 (1 variante sur 2), C2 tâche 2 (mauvaise commande), et l'écriture experte
     correspondante comme positif. C'est le test d'entrée : un vérificateur qui ne recale pas
     ces trois-là, avec le contexte complet sous les yeux, n'est pas viable.

  python scripts/probe_verifier_zeroshot.py --limit 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PAIRS = ROOT / "data/tau2_replay/retail_write_negatives.jsonl"

_SYS = (
    "Tu es un VÉRIFICATEUR d'actions d'agent de service client (domaine retail). On te donne : la "
    "demande de l'utilisateur, les données du catalogue et des commandes, et UNE action d'écriture "
    "que l'agent s'apprête à exécuter.\n"
    "Ta tâche, et ELLE SEULE : vérifier la JUSTESSE DES ARGUMENTS de cette action. Chaque argument "
    "(identifiant de commande, d'article, de variante, méthode de paiement) désigne-t-il bien "
    "l'entité que la demande de l'utilisateur et les données imposent ?\n"
    "Une action peut être parfaitement bien formée, s'exécuter sans erreur, ET être fausse (mauvaise "
    "commande, mauvaise variante d'un produit, mauvais article). C'est exactement ce que tu dois "
    "détecter.\n"
    "INTERDICTION — tu ne juges PAS la COMPLÉTUDE. Ne te demande jamais si l'action traite toute la "
    "demande, s'il manque une étape, si d'autres actions devraient suivre, ni si elle est prématurée. "
    "Une action dont TOUS les arguments sont justes est VALIDE, même si elle ne couvre qu'une partie "
    "de la demande. Un seul argument faux suffit à la rendre INVALIDE.\n"
    "Réponds STRICTEMENT par une seule ligne : `VERDICT: VALIDE` ou `VERDICT: INVALIDE`, suivie "
    "d'une ligne `RAISON: <une phrase>`."
)


def _db(domain="retail") -> dict:
    # Le clone tau2-bench a vécu sur /root (éphémère, perdu au restart) puis sur /workspace
    # (persistant). On demande son chemin à tau2 lui-même ; les deux emplacements connus
    # restent en secours.
    cands = []
    try:
        from tau2.utils.utils import DATA_DIR

        cands.append(Path(DATA_DIR) / f"tau2/domains/{domain}/db.json")
    except Exception:
        pass
    cands += [Path(f"/workspace/tau2-bench/data/tau2/domains/{domain}/db.json"),
              Path(f"/root/tau2-bench/data/tau2/domains/{domain}/db.json")]
    for p in cands:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"db.json {domain} introuvable (essayé : {[str(c) for c in cands]})")


def _catalog_for(db: dict, args: dict) -> str:
    """Le catalogue PERTINENT : la commande visée + les variantes des produits concernés.
    C'est ce qu'un agent a réellement vu via ses lectures — pas plus."""
    out = []
    oid = args.get("order_id")
    if oid and oid in db["orders"]:
        o = db["orders"][oid]
        out.append(f"[COMMANDE {oid}] statut={o.get('status')} user={o.get('user_id')}")
        for it in o.get("items", []):
            out.append(f"  article item_id={it['item_id']} produit={it.get('name')} "
                       f"product_id={it.get('product_id')} options={it.get('options')}")
    ids = list(args.get("new_item_ids") or []) + list(args.get("item_ids") or [])
    seen = set()
    for iid in ids:
        for prod in db["products"].values():
            if iid in (prod.get("variants") or {}) and prod["product_id"] not in seen:
                seen.add(prod["product_id"])
                out.append(f"[PRODUIT {prod['name']} product_id={prod['product_id']}] variantes :")
                for vid, v in prod["variants"].items():
                    out.append(f"  item_id={vid} options={v['options']} "
                               f"dispo={v.get('available')} prix={v.get('price')}")
    return "\n".join(out) or "(aucune donnée)"


def _fmt(tool: str, args: dict) -> str:
    return f"{tool}({json.dumps(args, ensure_ascii=False)})"


def _ask(llm, request: str, catalog: str, tool: str, args: dict,
         sys_prompt: str | None = None) -> tuple[bool | None, str, str]:
    """Rend (verdict, raison, BRUT). Le brut est rendu pour que l'appelant puisse le montrer :
    un `None` (parse raté) et un vrai INVALIDE se ressemblent dans un score mais pas du tout
    dans une conclusion. `sys_prompt` permet de rejouer un ancien _SYS à instrument constant
    (cf. scripts/probe_verifier_prompt_ab.py)."""
    from morpheus.llm.base import system, user
    from morpheus.text import strip_reasoning

    prompt = (f"[DEMANDE DE L'UTILISATEUR]\n{request}\n\n"
              f"[DONNÉES CATALOGUE / COMMANDES]\n{catalog}\n\n"
              f"[ACTION D'ÉCRITURE PROPOSÉE]\n{_fmt(tool, args)}\n\n"
              "Cette action est-elle correcte ?")
    raw = strip_reasoning(llm.complete([system(sys_prompt or _SYS), user(prompt)]))
    m = re.search(r"VERDICT:\s*(VALIDE|INVALIDE)", raw, re.I)
    if not m:
        return None, raw[:80], raw
    r = re.search(r"RAISON:\s*(.+)", raw)
    return m.group(1).upper() == "VALIDE", (r.group(1)[:90] if r else ""), raw


# Les 3 échecs RÉELS du smoke retail_cap2500 + les écritures expertes correspondantes.
# ⚠️ 2 cas attendent VALIDE et 2 attendent INVALIDE : une réponse CONSTANTE score 2/4.
# Tout score < 2 est donc SOUS le tampon-encreur et demande une explication (parse ? biais ?),
# pas une interprétation directe. Constante de module pour que toute sonde qui compare des
# variantes tourne PROVABLEMENT sur les mêmes cas.
HELD_OUT = [
    ("tâche 0 · C1 · Qwen (1 variante fausse)", "0", "exchange_delivered_order_items",
     {"order_id": "#W2378156", "item_ids": ["1151293680", "4983901480"],
      "new_item_ids": ["2299424241", "7747408585"],
      "payment_method_id": "credit_card_9513926"}, False),
    ("tâche 0 · C1 · EXPERT (attendu VALIDE)", "0", "exchange_delivered_order_items",
     {"order_id": "#W2378156", "item_ids": ["1151293680", "4983901480"],
      "new_item_ids": ["7706410293", "7747408585"],
      "payment_method_id": "credit_card_9513926"}, True),
    ("tâche 2 · C2 · Qwen (mauvaise commande)", "2", "return_delivered_order_items",
     {"order_id": "#W6679257", "item_ids": ["5996159312"],
      "payment_method_id": "credit_card_9513926"}, False),
    ("tâche 2 · C2 · EXPERT (attendu VALIDE)", "2", "return_delivered_order_items",
     {"order_id": "#W2378156",
      "item_ids": ["4602305039", "4202497723", "9408160950"],
      "payment_method_id": "credit_card_9513926"}, True),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="nb de paires de la fabrique")
    ap.add_argument("--config", default="configs/qwen_tau2_retail74.yaml")
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
    cfg.policy.temperature = 0.0          # vérificateur : déterministe
    llm = build_llm(cfg.policy)
    db = _db()
    tasks = {str(t.id): t for t in registry.get_tasks_loader("retail")()}

    def request_of(tid: str) -> str:
        us = tasks[tid].user_scenario
        return str(getattr(us, "instructions", None) or us)

    print("=" * 82)
    print("SONDE — Qwen-VÉRIFICATEUR zero-shot (température 0)")
    print("=" * 82)

    # ---------- 1. HELD-OUT : les 3 échecs RÉELS du smoke ----------
    print("\n" + "-" * 82)
    print("HELD-OUT — les 3 écritures RÉELLES du smoke retail_cap2500 (test d'entrée)")
    print("-" * 82)
    n_ok = 0
    n_unparsed = 0
    for label, tid, tool, a, expect_valid in HELD_OUT:
        got, why, _raw = _ask(llm, request_of(tid), _catalog_for(db, a), tool, a)
        n_unparsed += (got is None)
        ok = (got == expect_valid)
        n_ok += ok
        v = {True: "VALIDE", False: "INVALIDE", None: "IMPARSABLE"}[got]
        print(f"  {'✅' if ok else '❌'} {label:<42} → {v:<10} (attendu "
              f"{'VALIDE' if expect_valid else 'INVALIDE'})")
        if why:
            print(f"      raison : {why}")
    print(f"\n  test d'entrée : {n_ok}/4    (référence : une réponse CONSTANTE score 2/4 —")
    print(f"                            2 cas attendent VALIDE, 2 attendent INVALIDE)")
    if n_unparsed:
        print(f"  ⚠️ {n_unparsed}/4 IMPARSABLES : le score ne mesure PAS le jugement du modèle mais "
              f"le parse.\n     Regarder le brut (scripts/probe_verifier_prompt_ab.py) avant toute "
              f"conclusion.")
    elif n_ok < 2:
        print(f"  ⚠️ {n_ok}/4 est SOUS la constante (2/4) alors que tout parse. Ce n'est donc pas un "
              f"artefact\n     de parsing : le modèle dévie de la constante du mauvais côté. À "
              f"expliquer, pas à interpréter tel quel.")

    # ---------- 2. FABRIQUE : paires étiquetées par l'oracle ----------
    if not PAIRS.exists():
        print(f"\n(pas de paires : lancer scripts/build_write_negatives.py)")
        return 0
    rows = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: args.limit]
    print("\n" + "-" * 82)
    print(f"FABRIQUE — {len(rows)} paires étiquetées par l'ORACLE τ² (db_expert=1, db_impostor<1)")
    print("-" * 82)

    tp = fp = tn = fn = unp = 0
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        req = request_of(r["task_id"])
        for args_, is_expert in ((r["expert_args"], True), (r["impostor_args"], False)):
            got, _, _ = _ask(llm, req, _catalog_for(db, args_), r["tool"], args_)
            if got is None:
                unp += 1
                continue
            if is_expert:
                tp += got; fn += (not got)          # expert : VALIDE attendu
            else:
                tn += (not got); fp += got          # imposteur : INVALIDE attendu
                k = by_kind.setdefault(r["kind"], [0, 0])
                k[0] += (not got); k[1] += 1
    n_e, n_i = tp + fn, tn + fp
    print(f"  ÉCRITURES EXPERTES  (VALIDE attendu)   : {tp}/{n_e} correctes "
          f"({100*tp/max(n_e,1):.0f}%)  ← faux rejets : {fn}")
    print(f"  IMPOSTEURS          (INVALIDE attendu) : {tn}/{n_i} recalés   "
          f"({100*tn/max(n_i,1):.0f}%)  ← TAMPONNÉS À TORT : {fp}")
    if unp:
        print(f"  réponses imparsables : {unp}")
    print(f"\n  détection par type de corruption :")
    for k, (ok, n) in sorted(by_kind.items()):
        print(f"    {k:<16} {ok:>3}/{n:<3} recalés ({100*ok/max(n,1):>3.0f}%)")
    rec_e, rec_i = 100 * tp / max(n_e, 1), 100 * tn / max(n_i, 1)
    bal = (rec_e + rec_i) / 2
    print(f"\n  exactitude ÉQUILIBRÉE : {bal:.1f}%   (hasard = 50%)")
    # ⚠️ L'exactitude équilibrée SEULE ne décide de rien : elle moyenne deux erreurs aux
    # conséquences opposées. Dans une boucle MPC le vérificateur CLASSE des candidats — un faux
    # rejet veto une action CORRECTE, ce qui est aussi coûteux qu'un imposteur tamponné. Et le
    # test d'entrée (les 3 échecs RÉELS) prime sur toute moyenne : un vérificateur qui rate le cas
    # dur avec le contexte complet sous les yeux n'est pas viable, quelle que soit sa moyenne.
    verdict = []
    if rec_i < 70:
        verdict.append("TAMPON : il ne recale pas les imposteurs")
    if rec_e < 80:
        verdict.append(f"INUTILISABLE COMME GARDE : {100-rec_e:.0f}% des écritures EXPERTES sont "
                       f"rejetées à tort (il veto des actions correctes)")
    if n_ok < 4:
        verdict.append(f"TEST D'ENTRÉE ÉCHOUÉ ({n_ok}/4 sur les échecs RÉELS)")
    if verdict:
        print("  ⇒ NON VIABLE en l'état :")
        for v in verdict:
            print(f"     · {v}")
        print("     Il y a du SIGNAL (imposteurs > hasard), mais pas un Q utilisable tel quel.")
    else:
        print("  ⇒ le vérificateur DISCRIMINE et passe le test d'entrée : le 1er Q peut être un PROMPT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
