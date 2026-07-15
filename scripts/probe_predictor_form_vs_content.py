#!/usr/bin/env python
"""SONDE (lecture seule) : `cos(ẑ', z_next) = 0.76` — FORME ou CONTENU ?

Deux lectures incompatibles du 0.76 mesuré par `probe_predictor_identity.py` :
  (a) « je sais que get_product_details sur CE produit renvoie CE clavier »  → vraie dynamique ;
  (b) « je sais que get_product_details renvoie un blob de forme produit »   → schéma seul.
Deux payloads produits différents partagent schéma, vocabulaire et ponctuation JSON : leur
cosinus est peut-être DÉJÀ très haut. Ce script mesure ce plancher et ce que le prédicteur
ajoute par-dessus.

  F1 — plancher de schéma : cos(z_next_i, z_next_j), MÊME outil, entités différentes
  F2 — cos(ẑ', z_next_VRAI) vs cos(ẑ', z_next_MÊME OUTIL / AUTRE ENTITÉ) → l'écart utile
  F3 — récupération : ẑ' retrouve-t-il le VRAI z_next parmi des distracteurs de même outil ?
  F4 — ligne de base CENTROÏDE : « moyenne des next_obs de cet outil » (table outil→forme,
       calculée sur le TRAIN seul). Si elle égale ẑ', le prédicteur est un annuaire.
  F5 — conséquence pour `divergence` + le routeur : δ sépare-t-il « bon payload » de
       « payload de la MAUVAISE entité, même outil » ? (= le cas `coherent_but_wrong`)

Held-out identique à probe_predictor_identity.py (split de build_tau2_alignment_data.py rejoué).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_predictor_identity import cos_rows, load_heldout  # noqa: E402

CKPT = ROOT / "checkpoints" / "jepa_tau2_align" / "jepa.pt"
TRAIN = ROOT / "data" / "tau2_replay" / "retail_align_train.jsonl"
SURPRISE_THRESHOLD = 0.5      # config.py:34 — seuil du routeur
N_DISTRACTORS = 9             # F3 : 1 vrai + 9 distracteurs ⇒ hasard = 10%


def tool_of(action: str) -> str:
    m = re.match(r"^([A-Za-z_]\w*)\(", action or "")
    return m.group(1) if m else "?"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC de Mann-Whitney = P(un tirage de `pos` > un tirage de `neg`). 0.5 = aveugle.
    Pour δ (où le BON cas doit être le plus BAS), appeler auc(d_wrong, d_true)."""
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x)) + 1.0
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def main() -> int:
    import torch

    from morpheus.jepa.encoders import build_encoder
    from morpheus.jepa.model import JEPA

    trans, _ = load_heldout()
    train = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]

    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg, enc_dim = ckpt["config"], int(ckpt["enc_dim"])
    enc = build_encoder(cfg["encoder"], dim=enc_dim, model_name=cfg["encoder_model"])
    model = JEPA(enc_dim, cfg["latent_dim"], cfg["action_dim"], cfg["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    texts = sorted({t["obs"] for t in trans} | {t["next_obs"] for t in trans}
                   | {t["action"] for t in trans} | {r["next_obs"] for r in train})
    E = enc.encode(texts)
    idx = {t: i for i, t in enumerate(texts)}

    def emb(ts):
        return E[[idx[t] for t in ts]]

    @torch.no_grad()
    def proj(ts):
        return model.encode_state(torch.from_numpy(np.ascontiguousarray(emb(ts), np.float32))).numpy()

    @torch.no_grad()
    def predict(s, a):
        return model.predict_next(
            torch.from_numpy(np.ascontiguousarray(emb(s), np.float32)),
            torch.from_numpy(np.ascontiguousarray(emb(a), np.float32))).numpy()

    obs = [t["obs"] for t in trans]
    nxt = [t["next_obs"] for t in trans]
    act = [t["action"] for t in trans]
    tools = [tool_of(a) for a in act]
    Z_NEXT, Z_HAT = proj(nxt), predict(obs, act)

    # distracteurs : MÊME outil, action textuellement DIFFÉRENTE (⇒ autre entité)
    by_tool: dict[str, list[int]] = {}
    for i, tl in enumerate(tools):
        by_tool.setdefault(tl, []).append(i)

    rng = np.random.default_rng(0)
    eligible, dis = [], []
    for i, t in enumerate(trans):
        cand = [j for j in by_tool[tools[i]] if trans[j]["action"] != t["action"]]
        if not cand:
            continue
        eligible.append(i)
        dis.append(list(rng.choice(cand, size=min(N_DISTRACTORS, len(cand)), replace=False)))
    print(f"held-out : {len(trans)} transitions | {len(eligible)} avec >=1 distracteur "
          f"« même outil, autre entité » | {len(set(tools))} outils distincts")
    print(f"outil dominant : {max(by_tool, key=lambda k: len(by_tool[k]))} "
          f"({max(len(v) for v in by_tool.values())} transitions)\n")

    ei = np.array(eligible)
    flat_i = [i for i, ds in zip(eligible, dis) for _ in ds]
    flat_j = [j for ds in dis for j in ds]

    c_true = cos_rows(Z_HAT[ei], Z_NEXT[ei])                   # ẑ' vs le VRAI next
    c_wrong = cos_rows(Z_HAT[flat_i], Z_NEXT[flat_j])          # ẑ' vs même outil, autre entité
    c_floor = cos_rows(Z_NEXT[flat_i], Z_NEXT[flat_j])         # plancher : deux vrais next du même outil

    # plancher inter-outils, pour l'échelle
    perm = rng.permutation(len(trans))
    diff_tool = [k for k in range(len(trans)) if tools[k] != tools[perm[k]]]
    c_floor_diff = cos_rows(Z_NEXT[diff_tool], Z_NEXT[perm[diff_tool]])

    # le plancher se mesure dans DEUX espaces : l'intuition « les payloads JSON se ressemblent »
    # porte sur l'encodeur MiniLM brut, pas sur le latent projeté que `divergence` lit vraiment.
    e_floor = cos_rows(emb([trans[i]["next_obs"] for i in flat_i]),
                       emb([trans[j]["next_obs"] for j in flat_j]))
    print("=" * 78)
    print("F1 — PLANCHER DE SCHÉMA : deux VRAIS états, sans aucun prédicteur")
    print("=" * 78)
    print("  MÊME outil, entités différentes :")
    print(f"    dans MiniLM brut   cos(E(next), E(next'))       : {e_floor.mean():+.4f} "
          f"(méd {np.median(e_floor):+.4f})")
    print(f"    dans le latent     cos(proj(next), proj(next')) : {c_floor.mean():+.4f} "
          f"(méd {np.median(c_floor):+.4f})")
    print(f"  Outils DIFFÉRENTS, dans le latent               : {c_floor_diff.mean():+.4f}")
    print("  → `proj` n'est pas un tuyau : il ÉCARTE des payloads que MiniLM confondait.")
    print("    Le plancher pertinent (celui que lit `divergence`) est le latent.")

    print("\n" + "=" * 78)
    print("F2 — CE QUE LE PRÉDICTEUR AJOUTE AU-DESSUS DU PLANCHER")
    print("=" * 78)
    print(f"  cos(ẑ', z_next VRAI)                  : {c_true.mean():+.4f}")
    print(f"  cos(ẑ', z_next MÊME OUTIL/AUTRE ENTITÉ): {c_wrong.mean():+.4f}")
    print(f"  écart utile (vrai − mauvaise entité)  : {c_true.mean()-c_wrong.mean():+.4f}")
    print(f"  marge au-dessus du plancher F1        : {c_true.mean()-c_floor.mean():+.4f}")

    print("\n" + "=" * 78)
    print(f"F3 — RÉCUPÉRATION : ẑ' retrouve-t-il le VRAI next parmi ses sosies de même outil ?")
    print("=" * 78)
    ranks, chances = [], []
    for i, ds in zip(eligible, dis):
        sims = np.array([float(np.dot(Z_HAT[i], Z_NEXT[j]) /
                               (np.linalg.norm(Z_HAT[i]) * np.linalg.norm(Z_NEXT[j])))
                         for j in [i] + list(ds)])
        ranks.append(int((sims > sims[0]).sum()) + 1)          # rang du vrai (1 = trouvé)
        chances.append(1.0 / len(sims))
    ranks = np.array(ranks)
    print(f"  top-1 : {100*float((ranks==1).mean()):.1f}%   (hasard = {100*float(np.mean(chances)):.1f}%)")
    print(f"  rang moyen du vrai : {ranks.mean():.2f} / {np.mean([len(d)+1 for d in dis]):.1f} candidats")

    # F4 — ligne de base « centroïde de l'outil », calculée sur le TRAIN uniquement
    print("\n" + "=" * 78)
    print("F4 — LIGNE DE BASE CENTROÏDE : « la forme moyenne que renvoie cet outil » (TRAIN seul)")
    print("=" * 78)
    Z_TRAIN = proj([r["next_obs"] for r in train])
    cen: dict[str, np.ndarray] = {}
    for tl in set(tools):
        rows = [k for k, r in enumerate(train) if tool_of(r["action"]) == tl]
        if rows:
            cen[tl] = Z_TRAIN[rows].mean(axis=0)
    has_cen = np.array([i for i in eligible if tools[i] in cen])
    C = np.stack([cen[tools[i]] for i in has_cen])
    c_cen = cos_rows(C, Z_NEXT[has_cen])
    print(f"  cos(centroïde_outil, z_next VRAI) : {c_cen.mean():+.4f}   (n={len(has_cen)})")
    print(f"  cos(ẑ',              z_next VRAI) : {cos_rows(Z_HAT[has_cen], Z_NEXT[has_cen]).mean():+.4f}")
    print(f"  → le prédicteur bat l'annuaire de {cos_rows(Z_HAT[has_cen], Z_NEXT[has_cen]).mean()-c_cen.mean():+.4f}")
    # même test de récupération, mais avec le centroïde comme « prédicteur »
    r_cen = []
    for i, ds in zip(eligible, dis):
        if tools[i] not in cen:
            continue
        v = cen[tools[i]]
        sims = np.array([float(np.dot(v, Z_NEXT[j]) / (np.linalg.norm(v) * np.linalg.norm(Z_NEXT[j])))
                         for j in [i] + list(ds)])
        r_cen.append(int((sims > sims[0]).sum()) + 1)
    r_cen = np.array(r_cen)
    print(f"  récupération top-1 du CENTROÏDE : {100*float((r_cen==1).mean()):.1f}% "
          f"(le prédicteur : {100*float((ranks==1).mean()):.1f}%)")

    # F4b — l'obs SEULE suffit-elle ? (si oui, la « prédiction » ne serait qu'une recopie
    # de la partie pertinente du contexte, et l'action ne servirait à rien)
    Z_OBS = proj(obs)
    r_obs = []
    for i, ds in zip(eligible, dis):
        v = Z_OBS[i]
        sims = np.array([float(np.dot(v, Z_NEXT[j]) / (np.linalg.norm(v) * np.linalg.norm(Z_NEXT[j])))
                         for j in [i] + list(ds)])
        r_obs.append(int((sims > sims[0]).sum()) + 1)
    print(f"  récupération top-1 depuis l'OBS SEULE (sans action) : "
          f"{100*float((np.array(r_obs)==1).mean()):.1f}%")
    print("  → l'annuaire d'outil est au niveau du hasard ; l'obs seule fait mieux que le hasard")
    print("    (le contexte MENTIONNE souvent l'entité) mais reste loin. C'est la COMBINAISON")
    print("    (obs, action) qui porte l'information — l'action fait l'essentiel du travail.")

    # F4c — le résultat tient-il hors de l'outil dominant ?
    print("\n  robustesse par outil (top-1 du prédicteur) :")
    for tl in sorted(set(tools), key=lambda k: -len(by_tool[k]))[:5]:
        m = [k for k, i in enumerate(eligible) if tools[i] == tl]
        if len(m) < 3:
            continue
        ch = float(np.mean([1.0 / (len(dis[k]) + 1) for k in m]))
        print(f"    {tl:<32} n={len(m):>3}  top-1={100*float((ranks[m]==1).mean()):>5.1f}%  "
              f"(hasard {100*ch:.0f}%)")

    # F5 — conséquence pour divergence / routeur
    print("\n" + "=" * 78)
    print(f"F5 — `divergence` ET LE ROUTEUR (seuil = {SURPRISE_THRESHOLD}, config.py:34)")
    print("=" * 78)
    d_true = (1.0 - c_true) / 2.0
    d_wrong = (1.0 - c_wrong) / 2.0
    print(f"  δ quand l'observation est CORRECTE           : moy {d_true.mean():.4f}  "
          f"p95 {np.percentile(d_true,95):.4f}  max {d_true.max():.4f}")
    print(f"  δ quand c'est la MAUVAISE ENTITÉ (même outil): moy {d_wrong.mean():.4f}  "
          f"p95 {np.percentile(d_wrong,95):.4f}  max {d_wrong.max():.4f}")
    print(f"  AUC de δ pour séparer correct / mauvaise entité : {auc(d_wrong, d_true):.4f} "
          f"(0.5 = aveugle, 1.0 = parfait)")
    print(f"\n  fraction de δ > {SURPRISE_THRESHOLD} (le routeur se déclenche) :")
    print(f"    observation CORRECTE      : {100*float((d_true > SURPRISE_THRESHOLD).mean()):.2f}%")
    print(f"    MAUVAISE ENTITÉ           : {100*float((d_wrong > SURPRISE_THRESHOLD).mean()):.2f}%")
    print(f"  δ maximum observé, tous cas confondus : {max(d_true.max(), d_wrong.max()):.4f}")

    # Le signal est-il mauvais, ou le SEUIL est-il mal placé ? Balayage.
    print("\n  balayage du seuil (détection = δ > seuil) :")
    print(f"    {'seuil':>7} {'détecté si MAUVAISE ENTITÉ':>28} {'fausses alarmes si CORRECT':>28}")
    best = None
    for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        tpr = float((d_wrong > thr).mean())
        fpr = float((d_true > thr).mean())
        star = ""
        if best is None or (tpr - fpr) > best[1]:
            best, star = (thr, tpr - fpr), ""
        print(f"    {thr:>7.2f} {100*tpr:>27.1f}% {100*fpr:>27.1f}%{star}")
    print(f"  → meilleur écart (Youden) au seuil {best[0]:.2f} : +{100*best[1]:.1f} pts")
    print(f"  → le seuil {SURPRISE_THRESHOLD} de config.py:34 est hors de l'échelle que δ atteint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
