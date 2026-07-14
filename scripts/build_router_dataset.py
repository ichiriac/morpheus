#!/usr/bin/env python
"""Jointure traces ↔ annotations → dataset d'entraînement du routeur de surprise (Phase 4).

Joint les annotations manuelles ERREUR/NOUVEAUTÉ (`data/annotations/tau2_error_novelty.jsonl`,
clé `episode = "<run>#<task>"` + `turn` 1-indexé — mêmes conventions que
scripts/validate_goal_signal.py) aux trajectoires journalisées, recalcule les signaux du
routeur HORS-LIGNE (`morpheus.router.features.signals_for_episode` : fidèle à loop.py,
même détecteur d'erreur τ², même requête KB/mémoire), et écrit un JSONL prêt pour
`morpheus train-router`.

Signaux par pas annoté :
- déterministes, recalculés partout : tool_error, familiarity, repeated_tool, is_user_turn,
  kb_top_score/kb_hits (KB versionnée data/kb/<domaine>.md), memory_hits (replay FactMemory) ;
- score_before/after : SEULEMENT si --checkpoint (JepaWorldModel → torch requis) — sinon
  None + indicateur « non sondé ». ⚠️ validité PAR DISTRIBUTION (cf. data/annotations/README :
  goal générique en retail non-solo → scorer surtout significatif sur les runs solo) ;
- reducibility : non rejouable hors-ligne (sonde LLM) → repris des traces si journalisé.

Usage :
  python scripts/build_router_dataset.py \\
    --episodes data/annotations/trajectories/retail_postfix/episodes.jsonl \\
               data/annotations/trajectories/telecom_solo_postfix/episodes.jsonl \\
    --labels data/annotations/tau2_error_novelty.jsonl \\
    --out data/router/dataset.jsonl \\
    [--checkpoint checkpoints/jepa_tau2_align/jepa.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morpheus.agents.knowledge import KnowledgeBase, locate_policy      # noqa: E402
from morpheus.agents.surprise import SurpriseSignals                    # noqa: E402
from morpheus.router.features import signals_for_episode                # noqa: E402

_DOMAINS = ("retail", "telecom", "airline")


def _load_labels(path: str) -> dict[str, list[dict]]:
    """{episode: [{turn, label, rationale}, …]} — clé identique à validate_goal_signal."""
    out: dict[str, list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = r.get("episode") or r.get("id")
        out.setdefault(key, []).append({
            "turn": int(r["turn"]),
            "label": str(r["label"]).upper(),
            "rationale": r.get("rationale", ""),
        })
    return out


def _domain_of(run_name: str, fallback: str) -> str:
    for d in _DOMAINS:
        if d in run_name.lower():
            return d
    return fallback


def _kb_for(domain: str, cache: dict[str, KnowledgeBase | None]) -> KnowledgeBase | None:
    if domain not in cache:
        try:
            path = locate_policy(domain)
            cache[domain] = KnowledgeBase.from_policy_file(path, domain)
            print(f"  KB {domain!r} ← {path} ({len(cache[domain])} règles)")
        except FileNotFoundError as e:
            print(f"  ⚠️  KB {domain!r} introuvable → signaux kb_* non sondés ({e})")
            cache[domain] = None
    return cache[domain]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Construit le dataset du routeur (Phase 4).")
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="un ou plusieurs <run>/episodes.jsonl (format runner)")
    ap.add_argument("--labels", required=True,
                    help="annotations manuelles JSONL {episode, turn, label, rationale?}")
    ap.add_argument("--out", default="data/router/dataset.jsonl")
    ap.add_argument("--checkpoint", default=None,
                    help="jepa.pt pour sonder score_before/after (torch requis)")
    ap.add_argument("--domain", default="retail",
                    help="domaine KB de repli si non déductible du nom du run")
    ap.add_argument("--no-kb", action="store_true", help="ne pas sonder les signaux kb_*")
    ap.add_argument("--no-memory", action="store_true", help="ne pas rejouer la mémoire épisodique")
    ap.add_argument("--rag-top-k", type=int, default=3)
    ap.add_argument("--memory-top-k", type=int, default=3)
    args = ap.parse_args(argv)

    labels = _load_labels(args.labels)
    n_labels = sum(len(v) for v in labels.values())
    print(f"annotations : {n_labels} pas sur {len(labels)} épisodes ({args.labels})")

    scorer = None
    if args.checkpoint:
        from morpheus.agents.jepa_world_model import JepaWorldModel  # import torch paresseux

        scorer = JepaWorldModel(args.checkpoint, device="auto")
        print(f"scores : sondés via {args.checkpoint}")
        print("  ⚠️  validité PAR DISTRIBUTION (goal générique retail non-solo — cf. "
              "data/annotations/README.md)")
    else:
        print("scores : NON sondés (pas de --checkpoint) → score_before/after = None")

    kb_cache: dict[str, KnowledgeBase | None] = {}
    rows: list[dict] = []
    n_eps_joined = 0
    missed: list[str] = []

    for src in args.episodes:
        run = Path(src).parent.name
        kb = None if args.no_kb else _kb_for(_domain_of(run, args.domain), kb_cache)
        for i, line in enumerate(Path(src).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            ep = json.loads(line)
            key = f"{run}#{ep.get('task', i)}"
            anns = labels.get(key)
            if not anns:
                continue
            n_eps_joined += 1
            steps = ep.get("trace", [])
            scores = None
            if scorer is not None and ep.get("goal"):
                scores = [scorer.score_to_goal(ep["goal"], s.get("real_state", ""))
                          for s in steps]
            sigs = signals_for_episode(
                steps, kb=kb, rag_top_k=args.rag_top_k,
                use_memory=not args.no_memory, memory_top_k=args.memory_top_k,
                scores=scores,
            )
            for a in anns:
                idx = a["turn"] - 1                     # TraceStep.turn est 1-indexé
                if not 0 <= idx < len(sigs):
                    missed.append(f"{key} turn={a['turn']} (hors trace, {len(sigs)} pas)")
                    continue
                rows.append({
                    "episode": key,
                    "turn": a["turn"],
                    "label": a["label"],
                    "rationale": a["rationale"],
                    "chosen": steps[idx].get("chosen", ""),
                    "signals": sigs[idx].to_dict(),
                    "vector": sigs[idx].as_vector(),
                    "vector_fields": list(SurpriseSignals.VECTOR_FIELDS),
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")

    n_err = sum(1 for r in rows if r["label"] == "ERROR")
    print(f"\n✓ {out} : {len(rows)} pas ({n_err} ERREUR / {len(rows) - n_err} NOUVEAUTÉ) "
          f"depuis {n_eps_joined} épisodes joints")
    # couverture des signaux optionnels : fraction sondée (non-None)
    for f in ("score_before", "score_after", "kb_top_score", "memory_hits", "reducibility"):
        n_probed = sum(1 for r in rows if r["signals"].get(f) is not None)
        print(f"    {f:<14} sondé sur {n_probed}/{len(rows)}")
    if missed:
        print(f"  ⚠️  {len(missed)} annotation(s) non jointes : " + " ; ".join(missed))
    if not rows:
        print("❌ aucune ligne produite — vérifier la correspondance episode/turn.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
