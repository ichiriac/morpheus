#!/usr/bin/env python
"""SONDE (lecture seule) : le prédicteur JEPA a-t-il appris l'identité (ẑ' ≈ z) ?

Question tranchée : le world-model `checkpoints/jepa_tau2_align/jepa.pt` influence-t-il
réellement le choix d'action de `orchestrator/loop.py:105-110` ? Ce script n'entraîne rien,
ne modifie rien : il mesure sur des transitions HELD-OUT (jamais vues à l'entraînement).

Held-out : reconstruit depuis `data/tau2_replay/retail.jsonl` en rejouant la logique de split
de `scripts/build_tau2_alignment_data.py:34-41` (min-len 3, hash((seed,i)), val_frac 0.25).
La reconstruction est VÉRIFIÉE identique au `retail_align_val.jsonl` livré (assert), qui lui
n'a pas les actions — d'où le rejeu.

Notation :  z = proj(E(obs))   z_next = proj(E(next_obs))   ẑ' = predict_next(E(obs), E(action))

  A — effondrement vers l'identité : cos(ẑ',z) vs cos(ẑ',z_next) vs baseline cos(z,z_next)
  B — sensibilité à l'action     : cos(ẑ'(s,a_vraie), ẑ'(s,a_autre)) à état fixé
  C — dispersion des scores entre K=3 candidats (la quantité qui pilote loop.py:108)
  D — `score_to_goal` lit-il son argument `goal` ? (but faux / but vide) + corrélation au pas
  E — contrôles : mécanisme du classement + décalage de format train/inférence

Sur C : les candidats sont tirés de deux distributions, car le résultat en dépend —
  · pool GLOBAL   : négatifs faciles (actions d'autres épisodes, souvent hors-sujet) ;
  · MÊME ÉPISODE  : négatifs durs, plus proches de ce que Qwen propose vraiment à un état donné.
Et l'accord « argmax == action loguée » est ventilé par succès/échec : sur un épisode ÉCHOUÉ
l'action loguée n'est PAS une référence de qualité, donc s'en écarter n'est pas une faute.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CKPT = ROOT / "checkpoints" / "jepa_tau2_align" / "jepa.pt"
RETAIL = ROOT / "data" / "tau2_replay" / "retail.jsonl"
VAL_SHIPPED = ROOT / "data" / "tau2_replay" / "retail_align_val.jsonl"

SEED, VAL_FRAC, MIN_LEN = 0, 0.25, 3
N_OTHER_ACTIONS = 10      # sonde B
K_CANDIDATES = 3          # sonde C — K de loop.py

# Sonde D : un but volontairement HORS-DOMAINE (agent télécom, pas retail).
FAKE_GOAL_TELECOM = (
    "Tu es un agent de support technique du domaine telecom. Diagnostique la panne de ligne "
    "mobile de l'abonné, vérifie l'état du réseau et de la carte SIM, redémarre le routeur si "
    "nécessaire, et ouvre un ticket d'incident auprès du service réseau si la panne persiste."
)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))


def cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return np.sum(An * Bn, axis=1)


def to_score(c: np.ndarray) -> np.ndarray:
    """Exactement la transformation de JepaWorldModel.rollout : (cos+1)/2 borné [0,1]."""
    return np.clip((c + 1.0) / 2.0, 0.0, 1.0)


def load_heldout() -> tuple[list[dict], str]:
    rows = [json.loads(l) for l in RETAIL.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if len(r.get("states", [])) >= MIN_LEN]
    order = sorted(range(len(rows)), key=lambda i: (hash((SEED, i)) & 0xFFFFFFFF))
    val_idx = set(order[:max(3, int(len(order) * VAL_FRAC))])

    # garde-fou : la reconstruction doit rendre le val livré à l'identique, sinon tout est faux.
    shipped = [json.loads(l) for l in VAL_SHIPPED.read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [{"goal": r.get("goal", ""), "success": bool(r.get("success")), "states": r["states"]}
            for i, r in enumerate(rows) if i in val_idx]
    assert mine == shipped, "split held-out != retail_align_val.jsonl livré — logique divergente"

    trans, goal = [], ""
    for i, r in enumerate(rows):
        if i not in val_idx:
            continue
        goal = r.get("goal", "")
        states, acts, T = r["states"], r.get("actions") or [], len(r["states"])
        if len(acts) < T:                        # transitions à VRAIES actions uniquement
            continue
        for k in range(1, T):
            trans.append({"obs": states[k - 1], "action": acts[k], "next_obs": states[k],
                          "goal": goal, "progress": k / (T - 1), "step": k, "T": T, "traj": i,
                          "success": bool(r.get("success")),
                          "siblings": [a for j, a in enumerate(acts[1:T], 1) if j != k]})
    return trans, goal


def action_pool(exclude: set[str]) -> list[str]:
    """Toutes les actions RÉELLES distinctes du corpus (train inclus) — négatifs plausibles."""
    rows = [json.loads(l) for l in RETAIL.read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = {a for r in rows for a in (r.get("actions") or []) if a}
    return sorted(pool - exclude)


def main() -> int:
    import torch

    from morpheus.agents.jepa_world_model import JepaWorldModel
    from morpheus.jepa.encoders import build_encoder
    from morpheus.jepa.model import JEPA

    trans, goal = load_heldout()
    print(f"held-out : {len({t['traj'] for t in trans})} trajectoires → {len(trans)} transitions "
          f"à vraies actions | buts distincts = {len({t['goal'] for t in trans})}")

    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg, enc_dim = ckpt["config"], int(ckpt["enc_dim"])
    enc = build_encoder(cfg.get("encoder", "hashing"), dim=enc_dim,
                        model_name=cfg.get("encoder_model", "sentence-transformers/all-MiniLM-L6-v2"))
    model = JEPA(enc_dim, cfg.get("latent_dim", 256), cfg.get("action_dim", 128), cfg.get("hidden", 512))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- cache d'embeddings : E_state est gelé, chaque texte n'est encodé qu'une fois ---
    pool = action_pool(exclude=set())
    texts = sorted({t["obs"] for t in trans} | {t["next_obs"] for t in trans}
                   | {t["action"] for t in trans} | set(pool)
                   | {goal, FAKE_GOAL_TELECOM, ""})
    E = enc.encode(texts)
    idx = {t: i for i, t in enumerate(texts)}

    def emb(ts: list[str]) -> np.ndarray:
        return E[[idx[t] for t in ts]]

    @torch.no_grad()
    def proj(ts: list[str]) -> np.ndarray:
        return model.encode_state(torch.from_numpy(np.ascontiguousarray(emb(ts), np.float32))).numpy()

    @torch.no_grad()
    def predict(states: list[str], actions: list[str]) -> np.ndarray:
        s = torch.from_numpy(np.ascontiguousarray(emb(states), np.float32))
        a = torch.from_numpy(np.ascontiguousarray(emb(actions), np.float32))
        return model.predict_next(s, a).numpy()

    # garde-fou : le chemin batché doit rendre EXACTEMENT ce que la classe de prod calcule.
    wm = JepaWorldModel(CKPT, device="cpu")
    t0 = trans[0]
    assert np.allclose(predict([t0["obs"]], [t0["action"]])[0],
                       np.asarray(wm.predict(type("S", (), {"text": t0["obs"]})(), t0["action"]),
                                  dtype=np.float32), atol=1e-5), "chemin batché != JepaWorldModel"
    print("garde-fou : chemin batché == JepaWorldModel.predict ✓\n")

    obs = [t["obs"] for t in trans]
    nxt = [t["next_obs"] for t in trans]
    act = [t["action"] for t in trans]
    Z, Z_NEXT, Z_HAT = proj(obs), proj(nxt), predict(obs, act)
    z_goal = proj([goal])[0]

    # ================= A — effondrement vers l'identité =================
    c_hat_z = cos_rows(Z_HAT, Z)
    c_hat_next = cos_rows(Z_HAT, Z_NEXT)
    c_z_next = cos_rows(Z, Z_NEXT)
    print("=" * 78)
    print("A — EFFONDREMENT VERS L'IDENTITÉ  (held-out, n=%d)" % len(trans))
    print("=" * 78)
    for name, v in [("cos(ẑ', z)      [recopie l'état ?]", c_hat_z),
                    ("cos(ẑ', z_next) [prédiction]", c_hat_next),
                    ("cos(z, z_next)  [BASELINE : ne rien faire]", c_z_next)]:
        print(f"  {name:<44} moy={v.mean():+.4f}  méd={np.median(v):+.4f}  σ={v.std():.4f}")
    gain = c_hat_next.mean() - c_z_next.mean()
    print(f"\n  → gain du prédicteur sur la baseline « recopier » : {gain:+.4f}")
    print(f"  → le prédicteur bat-il « ne rien faire » ? "
          f"{'OUI' if gain > 0 else 'NON — il est PIRE que ne rien faire'} "
          f"({100*float((c_hat_next > c_z_next).mean()):.1f}% des transitions)")

    # ================= B — sensibilité à l'action =================
    rng = np.random.default_rng(0)
    others = [[p for p in rng.choice(pool, size=N_OTHER_ACTIONS + 3, replace=False)
               if p != t["action"]][:N_OTHER_ACTIONS] for t in trans]
    flat_s = [t["obs"] for t, o in zip(trans, others) for _ in o]
    flat_a = [a for o in others for a in o]
    Z_HAT_OTHER = predict(flat_s, flat_a)
    Z_HAT_REP = np.repeat(Z_HAT, [len(o) for o in others], axis=0)
    c_ab = cos_rows(Z_HAT_REP, Z_HAT_OTHER)
    print("\n" + "=" * 78)
    print(f"B — SENSIBILITÉ À L'ACTION  (état fixé, N={N_OTHER_ACTIONS} autres actions réelles)")
    print("=" * 78)
    print(f"  cos(ẑ'(s, a_vraie), ẑ'(s, a_autre))   moy={c_ab.mean():+.6f}  "
          f"méd={np.median(c_ab):+.6f}  min={c_ab.min():+.6f}  σ={c_ab.std():.6f}")
    print(f"  → 1.0 signifie « action ignorée ». Écart à 1.0 : {1.0 - c_ab.mean():.6f}")
    # référence d'échelle : à ACTION fixée, combien deux ÉTATS différents s'écartent-ils ?
    perm = rng.permutation(len(trans))
    c_states = cos_rows(Z_HAT, Z_HAT[perm])
    print(f"  référence : cos(ẑ'(s1,·), ẑ'(s2,·)) entre états différents = {c_states.mean():+.6f}")
    print(f"  → l'état déplace {(1-c_states.mean())/max(1e-12, 1-c_ab.mean()):.1f}× plus que l'action")

    # ================= C — dispersion des scores entre candidats =================
    print("\n" + "=" * 78)
    print(f"C — DISPERSION DES SCORES ENTRE CANDIDATS  (K={K_CANDIDATES}, pilote loop.py:108)")
    print("=" * 78)
    succ_mask = np.array([t["success"] for t in trans])
    print(f"  held-out : {int(succ_mask.sum())} transitions d'épisodes RÉUSSIS, "
          f"{int((~succ_mask).sum())} d'épisodes ÉCHOUÉS")

    for source, label in [("global", "pool GLOBAL (négatifs faciles)"),
                          ("same_traj", "MÊME ÉPISODE (négatifs durs, ≈ candidats Qwen)")]:
        # les transitions sans assez de négatifs internes sont ÉCARTÉES (pas tout le bloc).
        sub = [t for t in trans
               if source == "global" or len(t["siblings"]) >= K_CANDIDATES - 1]
        if len(sub) < 20:
            print(f"\n  [{label}] : écarté — seulement {len(sub)} transitions éligibles")
            continue
        m_succ = np.array([t["success"] for t in sub])
        stats = []
        for seed in range(10):                   # 10 graines : le tirage fait bouger le résultat
            r = np.random.default_rng(seed)
            cand = [[t["action"]] + list(r.choice(pool if source == "global" else t["siblings"],
                                                  size=K_CANDIDATES - 1, replace=False))
                    for t in sub]
            flat_s = [t["obs"] for t, cs in zip(sub, cand) for _ in cs]
            flat_a = [a for cs in cand for a in cs]
            S = to_score(cos_rows(predict(flat_s, flat_a), np.tile(z_goal, (len(flat_a), 1))))
            S = S.reshape(len(sub), K_CANDIDATES)        # colonne 0 = action VRAIE
            hit = S.argmax(axis=1) == 0
            stats.append((S.std(axis=1).mean(), S.mean(axis=1).std(),
                          (S.max(axis=1) - S.min(axis=1)).mean(), hit.mean(),
                          hit[m_succ].mean(), hit[~m_succ].mean()))
        A = np.array(stats)
        intra, inter, rng_, hit, hit_s, hit_f = A.mean(axis=0)
        n_s = int(m_succ.sum())
        se = np.sqrt((1 / K_CANDIDATES) * (1 - 1 / K_CANDIDATES) / len(sub))
        se_s = np.sqrt((1 / K_CANDIDATES) * (1 - 1 / K_CANDIDATES) / max(1, n_s))
        print(f"\n  [{label}]  (n={len(sub)}, moyenne sur {len(stats)} graines)")
        print(f"    écart-type INTRA-état (entre les {K_CANDIDATES} candidats) : {intra:.6f}")
        print(f"    écart-type INTER-états (scores moyens par état)   : {inter:.6f}")
        print(f"    ratio intra/inter                                 : {intra/max(1e-12,inter):.4f}")
        print(f"    étendue max−min intra-état                        : {rng_:.6f}")
        print(f"    argmax == action loguée : {hit*100:.1f}%  (hasard {100/K_CANDIDATES:.1f}%, "
              f"z={(hit-1/K_CANDIDATES)/se:+.2f}, σ_graines={A[:,3].std()*100:.1f} pts)")
        print(f"      · épisodes RÉUSSIS (n={n_s}, action loguée = référence crédible) : "
              f"{hit_s*100:.1f}%  z={(hit_s-1/K_CANDIDATES)/se_s:+.2f}")
        print(f"      · épisodes ÉCHOUÉS (n={int((~m_succ).sum())}, action loguée NON "
              f"référence) : {hit_f*100:.1f}%")

    # ================= D — score_to_goal lit-il son argument goal ? =================
    print("\n" + "=" * 78)
    print("D — `score_to_goal` LIT-IL SON ARGUMENT `goal` ?")
    print("=" * 78)
    variants = {"but RETAIL (vrai)": goal, "but TELECOM (faux)": FAKE_GOAL_TELECOM,
                "but VIDE ('')": ""}
    base = None
    for name, g in variants.items():
        zg = proj([g])[0]
        sc = to_score(cos_rows(Z, np.tile(zg, (len(Z), 1))))
        if base is None:
            base = sc
        d = float(np.abs(sc - base).mean())
        print(f"  {name:<20} score moy={sc.mean():.6f}  σ={sc.std():.6f}  "
              f"|Δ| vs vrai but={d:.6f}")
        # le RANG est ce qui compte pour un argmax : l'ordre des états change-t-il ?
        if name != "but RETAIL (vrai)":
            rho = float(np.corrcoef(np.argsort(np.argsort(sc)), np.argsort(np.argsort(base)))[0, 1])
            print(f"  {'':20} corrélation de RANG avec le vrai but : rho={rho:+.4f}")

    steps = np.array([t["step"] for t in trans], dtype=np.float64)
    prog = np.array([t["progress"] for t in trans], dtype=np.float64)
    s_true = to_score(cos_rows(Z, np.tile(z_goal, (len(Z), 1))))
    for label, x in [("indice du pas k", steps), ("progress = k/(T−1)", prog)]:
        r = float(np.corrcoef(s_true, x)[0, 1])
        print(f"  corr(score_to_goal, {label:<20}) : r={r:+.4f}  R²={r*r:.4f}")
    # R² de la cible d'alignement telle qu'entraînée : cos ≈ 2·progress − 1
    c_true = cos_rows(Z, np.tile(z_goal, (len(Z), 1)))
    tgt = 2.0 * prog - 1.0
    mse = float(((c_true - tgt) ** 2).mean())
    var = float(tgt.var())
    print(f"  cible d'alignement cos≈2·progress−1 : MSE={mse:.4f} vs var(cible)={var:.4f} "
          f"⇒ R²={1 - mse/var:+.4f} (held-out 25%)")

    # ================= E — contrôles =================
    print("\n" + "=" * 78)
    print("E — CONTRÔLES")
    print("=" * 78)

    # E1 — sur QUOI le classement des candidats se fait-il ? Hypothèse : « à quel point cette
    # action apparaît TARD dans un épisode », i.e. le score reste un compteur de pas (cf. D).
    rows = [json.loads(l) for l in RETAIL.read_text(encoding="utf-8").splitlines() if l.strip()]
    a_prog: dict[str, list[float]] = {}
    for r in rows:
        st, ac = r.get("states", []), r.get("actions") or []
        if len(ac) < len(st) or len(st) < 2:
            continue
        for k in range(1, len(st)):
            a_prog.setdefault(ac[k], []).append(k / (len(st) - 1))
    a_prog_mean = {a: float(np.mean(v)) for a, v in a_prog.items()}

    r2 = np.random.default_rng(0)
    cand = [[t["action"]] + list(r2.choice(pool, size=K_CANDIDATES - 1, replace=False))
            for t in trans]
    flat_s = [t["obs"] for t, cs in zip(trans, cand) for _ in cs]
    flat_a = [a for cs in cand for a in cs]
    S = to_score(cos_rows(predict(flat_s, flat_a), np.tile(z_goal, (len(flat_a), 1))))
    S = S.reshape(len(trans), K_CANDIDATES)
    P = np.array([[a_prog_mean.get(a, np.nan) for a in cs] for cs in cand])
    ok = ~np.isnan(P).any(axis=1)
    r_sp = float(np.corrcoef(S[ok].ravel(), P[ok].ravel())[0, 1])
    print(f"  corr(score d'un candidat, progress moyen de CETTE action dans le corpus) : "
          f"r={r_sp:+.4f}")
    print(f"  argmax == candidat le plus « tardif » : "
          f"{100*float((S[ok].argmax(1) == P[ok].argmax(1)).mean()):.1f}% (hasard {100/K_CANDIDATES:.1f}%)")
    print(f"  progress moyen : action loguée={P[ok][:,0].mean():.3f} | "
          f"action élue par l'argmax={P[ok][np.arange(int(ok.sum())), S[ok].argmax(1)].mean():.3f}")
    print("  → le classement suit « cette action arrive-t-elle tard dans un épisode ? »,")
    print("    pas « cette action est-elle la bonne ICI » (le score ne regarde pas l'état courant).")

    # E2 — décalage de format : l'entraînement a vu `tool({"k": "v"})` (JSON, cf.
    # build_tau2_alignment_data.py:64) ; loop.py encode `str(Action)` = `tool(k='v')`
    # (types.py:22-23). Ce n'est pas le même texte → pas la même embedding.
    def to_infer_fmt(a: str) -> str:
        m = re.match(r"^([A-Za-z_]\w*)\((.*)\)$", a, re.S)
        if not m:
            return a
        try:
            d = ast.literal_eval(m.group(2).strip()) if m.group(2).strip() else {}
        except Exception:
            return a
        if not isinstance(d, dict):
            return a
        from morpheus.orchestrator.types import Action
        return str(Action(tool=m.group(1), args=d))

    pairs = [(t["action"], to_infer_fmt(t["action"])) for t in trans]
    conv = [p for p in pairs if p[0] != p[1]]
    print(f"\n  format des actions : {len(conv)}/{len(pairs)} transitions changent de texte "
          f"entre entraînement et inférence")
    if conv:
        print(f"    TRAIN (build_tau2_alignment_data.py:64) : {conv[0][0][:64]}")
        print(f"    INFER (loop.py → types.py:22 str(Action)) : {conv[0][1][:64]}")
    # cache local : les textes format-inférence ne sont pas dans le cache principal.
    t_fmt = sorted({x for p in pairs for x in p} | {t["obs"] for t in trans})
    E_f = enc.encode(t_fmt)
    i_f = {t: i for i, t in enumerate(t_fmt)}

    @torch.no_grad()
    def predict_f(states: list[str], actions: list[str]) -> np.ndarray:
        s = torch.from_numpy(np.ascontiguousarray(E_f[[i_f[t] for t in states]], np.float32))
        a = torch.from_numpy(np.ascontiguousarray(E_f[[i_f[t] for t in actions]], np.float32))
        return model.predict_next(s, a).numpy()

    c_fmt = cos_rows(E_f[[i_f[p[0]] for p in pairs]], E_f[[i_f[p[1]] for p in pairs]])
    obs_l = [t["obs"] for t in trans]
    z_tr = predict_f(obs_l, [p[0] for p in pairs])
    z_in = predict_f(obs_l, [p[1] for p in pairs])
    print(f"    cos(E(a_train), E(a_infer)) = {c_fmt.mean():.4f}  →  "
          f"cos(ẑ'_train-fmt, ẑ'_infer-fmt) = {cos_rows(z_tr, z_in).mean():.4f}")
    # l'écart de format tient-il face à l'écart entre deux actions DIFFÉRENTES ? (sonde B = 0.69)
    print(f"    à comparer à la sonde B : deux actions différentes donnent {c_ab.mean():.4f}")
    print("  → le prédicteur est sondé ici dans le format d'ENTRAÎNEMENT (favorable) ; en prod")
    print("    il reçoit un format qu'il n'a jamais vu. Tout ce qui précède est un plafond.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
