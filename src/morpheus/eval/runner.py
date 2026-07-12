"""Runner : instancie les composants depuis la config, joue N tâches, écrit les résultats."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..agents.policy import Policy
from ..agents.surprise import SurpriseRouter
from ..agents.world_model import WorldModel
from ..config import Config
from ..envs import build_env_factory
from ..llm import build_llm
from ..orchestrator.loop import Orchestrator
from .metrics import SuccessVsTurns, summarize


def run_experiment(cfg: Config, out_dir: str | None = None) -> SuccessVsTurns:
    policy_llm = build_llm(cfg.policy)
    wm_llm = build_llm(cfg.world_model)

    policy = Policy(policy_llm, k=cfg.orchestrator.k_candidates)
    world_model = WorldModel(wm_llm)
    orch = Orchestrator(policy, world_model, cfg.orchestrator, SurpriseRouter())

    make_env = build_env_factory(cfg.eval)
    metric = SuccessVsTurns()

    out = Path(out_dir or cfg.eval.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "episodes.jsonl"

    with trace_path.open("w", encoding="utf-8") as f:
        for i in range(cfg.eval.tasks):
            env = make_env(i)
            result = orch.run(env)
            bucket = env.required_turns()
            metric.add(bucket, result.success)
            f.write(json.dumps({
                "task": i,
                "required_turns": bucket,
                "success": result.success,
                "turns": result.turns,
                "total_reward": result.total_reward,
                "trace": [asdict(s) for s in result.trace],
            }, ensure_ascii=False) + "\n")

    (out / "summary.txt").write_text(summarize(metric), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metric
