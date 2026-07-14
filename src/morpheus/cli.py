"""Point d'entrée CLI.

Exemples :
    # smoke test complet, sans GPU ni clé API (stub + mock) :
    python -m morpheus.cli run --config configs/phase1.yaml

    # baseline nue vs world-model, en surchargeant depuis la ligne de commande :
    python -m morpheus.cli run --config configs/phase1.yaml --no-world-model --out runs/baseline
    python -m morpheus.cli run --config configs/phase1.yaml --out runs/phase1
"""

from __future__ import annotations

import argparse

from .config import Config
from .eval.metrics import summarize
from .eval.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="morpheus")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="jouer une expérience et tracer la courbe vs-tours")
    run.add_argument("--config", required=True, help="chemin du YAML de config")
    run.add_argument("--out", default=None, help="dossier de sortie (défaut : eval.out_dir)")
    run.add_argument("--tasks", type=int, default=None, help="surcharge le nb de tâches")
    run.add_argument("--env", default=None, choices=["mock", "tau2"], help="surcharge l'env")
    run.add_argument("--no-world-model", action="store_true",
                     help="baseline ReAct nue (Phase 0) : désactive le lookahead")
    run.add_argument("--k", type=int, default=None, help="surcharge k_candidates (K)")
    run.add_argument("--horizon", type=int, default=None, help="surcharge l'horizon (H)")
    run.add_argument("--concurrency", type=int, default=None,
                     help="rollouts LLM concurrents (>1 => vLLM batche)")

    chk = sub.add_parser("check-llm",
                         help="valider le branchement LLM + le format de sortie de la politique")
    chk.add_argument("--config", required=True, help="chemin du YAML de config")

    tj = sub.add_parser("train-jepa", help="entraîner le world-model latent JEPA (Phase 2)")
    tj.add_argument("--config", required=True, help="chemin du YAML JEPA (cf. configs/jepa.yaml)")

    tr = sub.add_parser("train-router",
                        help="entraîner le routeur de surprise appris (Phase 4) — CPU, numpy pur")
    tr.add_argument("--dataset", required=True,
                    help="dataset.jsonl produit par scripts/build_router_dataset.py")
    tr.add_argument("--out", default="checkpoints/router", help="dossier de sortie (router.json)")
    tr.add_argument("--folds", type=int, default=5, help="plis de validation croisée stratifiée")
    tr.add_argument("--epochs", type=int, default=3000)
    tr.add_argument("--lr", type=float, default=0.5)
    tr.add_argument("--l2", type=float, default=1e-2)
    tr.add_argument("--seed", type=int, default=0)

    di = sub.add_parser("inspect-data",
                        help="charger une source de trajectoires et afficher un aperçu normalisé")
    di.add_argument("--source", required=True, help="synthetic | jsonl:<path> | hf:<name>")
    di.add_argument("--limit", type=int, default=200)
    di.add_argument("--alfworld", action="store_true")
    di.add_argument("--steps-key", default=None)

    ik = sub.add_parser("inspect-kb",
                        help="charger une policy τ² en KB, lister les règles et tester une requête")
    ik.add_argument("--domain", default="retail", help="retail | telecom | airline | mock")
    ik.add_argument("--policy", default=None, help="chemin direct d'un policy.md (sinon dérivé du domaine)")
    ik.add_argument("--data-dir", default=None, help="racine des données τ² (sinon $TAU2_DATA_DIR / ./tau2-bench/data)")
    ik.add_argument("--query", default=None, help="requête de test (ex. un état surprenant)")
    ik.add_argument("--k", type=int, default=3, help="nb de règles à récupérer")

    args = parser.parse_args(argv)

    if args.cmd == "check-llm":
        from .diagnostics import check_llm

        return check_llm(Config.load(args.config))

    if args.cmd == "train-jepa":
        from .jepa.train import JepaConfig, train

        train(JepaConfig.load(args.config))
        return 0

    if args.cmd == "train-router":
        from .router.train import train_router

        return train_router(args.dataset, args.out, folds=args.folds, epochs=args.epochs,
                            lr=args.lr, l2=args.l2, seed=args.seed)

    if args.cmd == "inspect-data":
        from .jepa.data import describe_records, load_transitions

        trans = load_transitions(args.source, limit=args.limit,
                                 alfworld=args.alfworld, steps_key=args.steps_key)
        print(describe_records(trans, k=5))
        return 0

    if args.cmd == "inspect-kb":
        from .agents.knowledge import KnowledgeBase, locate_policy

        path = locate_policy(args.domain, args.policy, args.data_dir)
        kb = KnowledgeBase.from_policy_file(path, args.domain)
        print(f"KB {args.domain!r} ← {path}")
        print(f"{len(kb)} règles, {len(kb.sections)} sections\n")
        for r in kb.rules:
            print(f"  • [{r.section or '(intro)'}] {' '.join(r.text.split())[:110]}")
        if args.query:
            print(f"\nRequête : {args.query!r}")
            hits = kb.score(args.query)[: args.k]
            for s, r in hits:
                print(f"  [{s:5.2f}] [{r.section or '(intro)'}] {' '.join(r.text.split())[:100]}")
        return 0

    if args.cmd == "run":
        cfg = Config.load(args.config)
        if args.tasks is not None:
            cfg.eval.tasks = args.tasks
        if args.env is not None:
            cfg.eval.env = args.env
        if args.no_world_model:
            cfg.orchestrator.use_world_model = False
        if args.k is not None:
            cfg.orchestrator.k_candidates = args.k
        if args.horizon is not None:
            cfg.orchestrator.horizon = args.horizon
        if args.concurrency is not None:
            cfg.orchestrator.concurrency = args.concurrency
        metric = run_experiment(cfg, out_dir=args.out)
        print(summarize(metric))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
