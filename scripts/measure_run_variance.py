#!/usr/bin/env python
"""VARIANCE RUN-À-RUN + puissance du banc : l'A/B est-il seulement FAISABLE à cette taille ?

Le chiffre à produire AVANT toute décision d'architecture. Deux runs du MÊME bras (baseline,
même config, même pile, même seed, `temperature: 0.7`) sur les MÊMES tâches ne donnent pas le
même résultat. L'ampleur de cet écart borne ce que le banc peut détecter : tout gain du
world-model inférieur au bruit est indétectable, et un A/B lancé dans ces conditions serait
mort-né sans que rien ne le signale.

Trois sorties :
  1. BASELINE     — estimation ponctuelle + IC de Wilson sur le run complet.
  2. VARIANCE     — sur les tâches en RECOUVREMENT entre deux runs. L'écart net est trompeur
                    (les basculements se compensent) : c'est le TAUX DE BASCULEMENT qui compte.
  3. FAISABILITÉ  — McNemar (les deux bras tournent sur les mêmes tâches ⇒ test APPARIÉ) :
                    quel écart net minimal est significatif, et à quelle taille d'échantillon.

  python scripts/measure_run_variance.py runs/retail74_baseline runs/retail74_baseline_run2
"""
from __future__ import annotations

import argparse
import json
import math
from math import comb
from pathlib import Path


def load(d: Path) -> list[dict]:
    p = d / "episodes.jsonl" if d.is_dir() else d
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def task_id(ep: dict) -> str:
    t = ep.get("task")
    return str(t.get("id")) if isinstance(t, dict) else str(t)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson — correct aux petits n et près des bords, contrairement à Wald."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_p(b: int, c: int) -> float:
    """Binomial exact bilatéral sur les paires discordantes (pas le χ², faux aux petits n)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def min_detectable(n_tasks: int, p_disc: float) -> tuple[int, int, float] | None:
    """Le split de paires discordantes le MOINS extrême encore significatif à 5%."""
    nd = round(p_disc * n_tasks)
    best = None
    for k in range(0, nd // 2 + 1):
        if mcnemar_p(k, nd - k) < 0.05:
            best = k
    if best is None:
        return None
    return (nd - best, best, (nd - best - best) / n_tasks * 100)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="1er run (ex: runs/retail74_baseline)")
    ap.add_argument("run_b", type=Path, help="2e run, MÊME bras (ex: runs/retail74_baseline_run2)")
    ap.add_argument("--sizes", type=int, nargs="*", default=[74, 114],
                    help="tailles de banc à évaluer")
    a = ap.parse_args(argv)

    RA, RB = load(a.run_a), load(a.run_b)
    full = RB if len(RB) >= len(RA) else RA
    k = sum(1 for d in full if d.get("success"))
    lo, hi = wilson(k, len(full))

    print("=" * 78)
    print(f"1. BASELINE — {a.run_b if full is RB else a.run_a} (n={len(full)})")
    print("=" * 78)
    print(f"   succès       : {k}/{len(full)} = {100*k/len(full):.1f}%")
    print(f"   IC 95% Wilson: [{lo:.1f}% , {hi:.1f}%]   (±{(hi-lo)/2:.1f} pts)")
    print(f"   reward moyen : {sum(d.get('total_reward', 0) for d in full)/len(full):.3f}")

    ma = {task_id(d): bool(d.get("success")) for d in RA}
    mb = {task_id(d): bool(d.get("success")) for d in RB}
    com = sorted([x for x in ma if x in mb], key=lambda s: int(s) if s.isdigit() else 0)
    if not com:
        print("\n!! aucune tâche en recouvrement — variance non mesurable")
        return 1

    sa, sb = sum(ma[x] for x in com), sum(mb[x] for x in com)
    b = sum(1 for x in com if ma[x] and not mb[x])
    c = sum(1 for x in com if not ma[x] and mb[x])
    p_disc = (b + c) / len(com)

    print()
    print("=" * 78)
    print(f"2. VARIANCE RUN-À-RUN — {len(com)} tâches en recouvrement, MÊME bras, MÊME config")
    print("=" * 78)
    print(f"   run A : {sa}/{len(com)} = {100*sa/len(com):.1f}%")
    print(f"   run B : {sb}/{len(com)} = {100*sb/len(com):.1f}%")
    print(f"   écart NET : {abs(100*sa/len(com) - 100*sb/len(com)):.1f} pts")
    print(f"   BASCULEMENTS : {b+c}/{len(com)} = {100*p_disc:.0f}%  "
          f"({c} échec→succès, {b} succès→échec)")
    print(f"   stables      : {len(com)-b-c}/{len(com)} = {100*(1-p_disc):.0f}%")
    print(f"   ⚠️ l'écart net est le SOLDE de {b+c} basculements qui se compensent en partie.")
    print(f"      La grandeur qui borne le banc est le taux de basculement ({100*p_disc:.0f}%),")
    print(f"      pas l'écart net ({abs(100*sa/len(com) - 100*sb/len(com)):.1f} pts).")
    print(f"   contrôle : McNemar sur ces 2 runs du MÊME bras -> p={mcnemar_p(b, c):.3f}")
    print(f"      (doit être NON significatif : c'est le même bras. Sinon, un facteur a changé.)")

    print()
    print("=" * 78)
    print("3. FAISABILITÉ DE L'A/B (McNemar apparié — les 2 bras tournent sur les mêmes tâches)")
    print("=" * 78)
    for N in a.sizes:
        r = min_detectable(N, p_disc)
        if r is None:
            print(f"   n={N:4d} : aucun écart détectable")
        else:
            hi_, lo_, net = r
            print(f"   n={N:4d} : ~{round(p_disc*N)} paires discordantes -> split min {hi_}/{lo_}"
                  f" -> écart NET minimal détectable = {net:.1f} pts")
    print()
    print("   Taille requise (approx normale, se_net = sqrt(p_disc/n)) :")
    for g in (5, 10, 15, 20):
        need = math.ceil((1.96 / (g / 100)) ** 2 * p_disc)
        print(f"     gain de {g:2d} pts -> n ≈ {need:4d} tâches")
    print()
    print("   Trois leviers si le seuil est trop haut — la variance mesurée ici vient de")
    print("   `policy.temperature`, ce n'est PAS du bruit irréductible :")
    print("     · plus de tâches  : coûteux, borné par le domaine (retail = 114 max)")
    print("     · température ↓   : à 0.0 la politique est déterministe, la variance s'effondre")
    print("                         (contrepartie : moins d'exploration — à mesurer, 1 run)")
    print("     · K runs répétés  : divise l'erreur par √K, sans construire une tâche de plus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
