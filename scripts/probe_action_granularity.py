#!/usr/bin/env python
"""SONDE (lecture seule) : le chemin latent voit-il la granularité d'ARGUMENT ?

La taxonomie des échecs DB (BENCHMARKS.md, run `retail_cap2500`) dit que le mode d'échec dominant
de Qwen au harnais propre est la FAUSSETÉ CONFIANTE — une écriture bien formée, qui n'erre pas,
mais dont UN argument est faux :

  C1 (tâche 0) : bonne commande, bons item_ids, bon paiement ; new_item_ids
                 ['2299424241', …] au lieu de ['7706410293', …] — 1 VARIANTE sur 2.
  C2 (tâche 2) : bon outil, MAUVAISE commande (#W6679257 au lieu de #W2378156).

Pour que le MPC puisse départager ces candidats, il faut que `ẑ' = P(proj(E(s)), enc_action(E(a)))`
BOUGE quand cet argument change. Or `a` est un texte passé dans MiniLM (sac de sous-mots) puis
compressé en 128d par `enc_action` : deux IDs à dix chiffres pourraient être indiscernables.

On mesure, à ÉTAT FIXÉ (l'état réel sur lequel la décision a été prise) :
  1. cos(E(a_faux), E(a_vrai))                        — l'encodeur MiniLM gelé
  2. cos(enc_action(E(a_faux)), enc_action(E(a_vrai))) — l'encodeur d'action appris
  3. cos(ẑ'(s, a_faux), ẑ'(s, a_vrai))                — le prédicteur (ce que le MPC score)
  4. |Δ score| entre les deux candidats               — CE QUI PILOTE loop.py:108

Repères d'échelle établis le 2026-07-15 (probe_predictor_identity.py) :
  · deux actions RÉELLEMENT différentes  → cos(ẑ', ẑ') = 0.69   (sonde B)
  · dispersion intra-état des scores      → σ = 0.055           (sonde C)
Si le cos ici ≈ 1.0 et |Δ score| ≪ 0.055, le chemin latent est AVEUGLE à la granularité qui décide
du DB, et le Q devra prendre une autre entrée que la géométrie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CKPT = ROOT / "checkpoints" / "jepa_tau2_align" / "jepa.pt"

# État réel sur lequel la tâche 0 a décidé son écriture (real_state du t11, run retail_cap2500).
STATE = ('tool: {"order_id": "#W2378156", "user_id": "yusuf_rossi_9620", "address": {"address1": '
         '"763 Broadway", "address2": "Suite 135", "city": "Philadelphia", "country": "USA", '
         '"state": "PA", "zip": "19122"}, "items": [{"name": "Mechanical Keyboard", "product_id": '
         '"1656367028", "item_id": "1151293680", "price": 272.33, "options": {"switch type": '
         '"linear", "backlight": "RGB", "size": "full size"}}], "fulfillments": [], "status": '
         '"delivered", "payment_history": []}')

# C1 — l'écriture RÉELLE de Qwen (t12, tâche 0) vs sa version corrigée. UN ID de variante change.
C1_WRONG = ("exchange_delivered_order_items(order_id='#W2378156', item_ids=['1151293680', "
            "'4983901480'], new_item_ids=['2299424241', '7747408585'], "
            "payment_method_id='credit_card_9513926')")
C1_RIGHT = ("exchange_delivered_order_items(order_id='#W2378156', item_ids=['1151293680', "
            "'4983901480'], new_item_ids=['7706410293', '7747408585'], "
            "payment_method_id='credit_card_9513926')")

# C2 — bon outil, MAUVAISE commande (le `coherent_but_wrong` de la tâche 2).
C2_WRONG = ("return_delivered_order_items(order_id='#W6679257', item_ids=['5996159312'], "
            "payment_method_id='credit_card_9513926')")
C2_RIGHT = ("return_delivered_order_items(order_id='#W2378156', item_ids=['4602305039', "
            "'4202497723', '9408160950'], payment_method_id='credit_card_9513926')")

# Repères d'échelle : un changement GROSSIER (outil entier) et une action d'un autre registre.
COARSE = "get_order_details(order_id='#W2378156')"
DIALOGUE = "respond_to_user(text='Quelle variante souhaitez-vous exactement ?')"

# Sonde B (probe_predictor_identity.py) : deux actions réelles différentes, à état fixé.
REF_B = 0.6932
REF_C_SIGMA = 0.055   # dispersion intra-état des scores entre candidats (sonde C)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def main() -> int:
    import torch

    from morpheus.jepa.encoders import build_encoder
    from morpheus.jepa.model import JEPA

    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg, enc_dim = ckpt["config"], int(ckpt["enc_dim"])
    enc = build_encoder(cfg["encoder"], dim=enc_dim, model_name=cfg["encoder_model"])
    model = JEPA(enc_dim, cfg["latent_dim"], cfg["action_dim"], cfg["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    texts = [STATE, C1_WRONG, C1_RIGHT, C2_WRONG, C2_RIGHT, COARSE, DIALOGUE]
    E = {t: e for t, e in zip(texts, enc.encode(texts))}

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)).unsqueeze(0)

    with torch.no_grad():
        z_state = model.encode_state(T(E[STATE])).squeeze(0).numpy()
        z_goal = model.encode_state(T(E[STATE])).squeeze(0).numpy()  # remplacé plus bas
        A = {t: model.enc_action(T(E[t])).squeeze(0).numpy() for t in texts}
        P = {t: model.predict_next(T(E[STATE]), T(E[t])).squeeze(0).numpy() for t in texts}

    # le but τ² est une CONSTANTE : on le relit depuis le corpus (c'est celui que score_to_goal lit)
    import json
    goal = json.loads(next(iter(open(ROOT / "data/tau2_replay/retail.jsonl"))))["goal"]
    with torch.no_grad():
        z_goal = model.encode_state(T(enc.encode([goal])[0])).squeeze(0).numpy()

    def score(t):   # exactement JepaWorldModel.rollout : (cos+1)/2 borné
        return max(0.0, min(1.0, (cos(P[t], z_goal) + 1.0) / 2.0))

    print("=" * 84)
    print("SONDE — le chemin latent voit-il la granularité d'ARGUMENT ?  (état FIXÉ, réel)")
    print("=" * 84)
    print(f"repères : deux actions réellement différentes → cos(ẑ',ẑ') = {REF_B:.4f} (sonde B)")
    print(f"          dispersion intra-état des scores    → σ = {REF_C_SIGMA:.4f}     (sonde C)\n")

    for label, w, r, desc in [
        ("C1", C1_WRONG, C1_RIGHT, "1 ID de VARIANTE change (2299424241 → 7706410293)"),
        ("C2", C2_WRONG, C2_RIGHT, "MAUVAISE commande (#W6679257 → #W2378156)"),
    ]:
        print("-" * 84)
        print(f"{label} — {desc}")
        print(f"  1. cos(E(faux), E(vrai))                  = {cos(E[w], E[r]):.6f}   [MiniLM gelé]")
        print(f"  2. cos(enc_action(faux), enc_action(vrai)) = {cos(A[w], A[r]):.6f}   [encodeur d'action]")
        print(f"  3. cos(ẑ'(s,faux), ẑ'(s,vrai))            = {cos(P[w], P[r]):.6f}   [prédicteur]")
        sw, sr = score(w), score(r)
        d = abs(sw - sr)
        print(f"  4. score MPC : faux={sw:.6f}  vrai={sr:.6f}  →  |Δ| = {d:.6f}")
        print(f"     ⇒ |Δ| vaut {d/REF_C_SIGMA:.1%} de la dispersion intra-état (σ={REF_C_SIGMA})")
        print(f"     ⇒ l'argmax préfère : {'la VRAIE ✅' if sr > sw else 'la FAUSSE ❌'}")

    print("-" * 84)
    print("REPÈRES D'ÉCHELLE sur le même état (pour situer les cos ci-dessus) :")
    print(f"  cos(ẑ'(écriture C1 fausse), ẑ'(get_order_details))   = {cos(P[C1_WRONG], P[COARSE]):.6f}  [autre outil]")
    print(f"  cos(ẑ'(écriture C1 fausse), ẑ'(respond_to_user))     = {cos(P[C1_WRONG], P[DIALOGUE]):.6f}  [autre registre]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
