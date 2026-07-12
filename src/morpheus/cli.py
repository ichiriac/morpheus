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

    chk = sub.add_parser("check-llm",
                         help="valider le branchement LLM + le format de sortie de la politique")
    chk.add_argument("--config", required=True, help="chemin du YAML de config")

    tj = sub.add_parser("train-jepa", help="entraîner le world-model latent JEPA (Phase 2)")
    tj.add_argument("--config", required=True, help="chemin du YAML JEPA (cf. configs/jepa.yaml)")

    di = sub.add_parser("inspect-data",
                        help="charger une source de trajectoires et afficher un aperçu normalisé")
    di.add_argument("--source", required=True, help="synthetic | jsonl:<path> | hf:<name>")
    di.add_argument("--limit", type=int, default=200)
    di.add_argument("--alfworld", action="store_true")
    di.add_argument("--steps-key", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "check-llm":
        from .diagnostics import check_llm

        return check_llm(Config.load(args.config))

    if args.cmd == "train-jepa":
        from .jepa.train import JepaConfig, train

        train(JepaConfig.load(args.config))
        return 0

    if args.cmd == "inspect-data":
        from .jepa.data import describe_records, load_transitions

        trans = load_transitions(args.source, limit=args.limit,
                                 alfworld=args.alfworld, steps_key=args.steps_key)
        print(describe_records(trans, k=5))
        return 0

    if args.cmd == "run":
        cfg = Config.load(args.config)
        if args.tasks is not None:
            cfg.eval.tasks = args.tasks
        if args.env is not None:
            cfg.eval.env = args.env
        if args.no_world_model:
            cfg.orchestrator.use_world_model = False
        metric = run_experiment(cfg, out_dir=args.out)
        print(summarize(metric))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
