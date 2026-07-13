"""Runner : instancie les composants depuis la config, joue N tâches, écrit les résultats."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..agents.knowledge import KnowledgeBase, locate_policy
from ..agents.policy import Policy
from ..agents.surprise import SurpriseRouter
from ..agents.world_model import WorldModel
from ..config import Config
from ..envs import build_env_factory
from ..llm import build_llm
from ..orchestrator.loop import Orchestrator
from .metrics import SuccessVsTurns, summarize
from .report import _now_iso, write_reports


def run_experiment(cfg: Config, out_dir: str | None = None) -> SuccessVsTurns:
    started_at = _now_iso()
    policy_llm = build_llm(cfg.policy)
    wm_llm = build_llm(cfg.world_model)

    policy = Policy(policy_llm, k=cfg.orchestrator.k_candidates)
    # World-model : LLM par défaut ; JEPA latent seulement si activé (torch importé à ce moment).
    if cfg.jepa_wm.enabled:
        from ..agents.jepa_world_model import JepaWorldModel

        world_model = JepaWorldModel(cfg.jepa_wm.checkpoint, device=cfg.jepa_wm.device)
    else:
        world_model = WorldModel(wm_llm)

    kb: KnowledgeBase | None = None
    if cfg.orchestrator.use_rag:
        policy_path = locate_policy(
            cfg.eval.domain, cfg.eval.kb_policy_path, cfg.eval.tau2_data_dir
        )
        kb = KnowledgeBase.from_policy_file(policy_path, cfg.eval.domain)

    orch = Orchestrator(policy, world_model, cfg.orchestrator, SurpriseRouter(), kb=kb)

    make_env, n_tasks = build_env_factory(cfg.eval)
    metric = SuccessVsTurns()

    out = Path(out_dir or cfg.eval.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "episodes.jsonl"

    with trace_path.open("w", encoding="utf-8") as f:
        for i in range(n_tasks):
            env = make_env(i)
            try:
                result = orch.run(env)
            finally:
                # τ² : termine le thread orchestrateur si la boucle s'est arrêtée sans `done`.
                close = getattr(env, "close", None)
                if callable(close):
                    close()
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
    # Résultats mis de côté en markdown : rapport par-run + journal cumulatif BENCHMARKS.md.
    write_reports(cfg, metric, out, started_at)
    return metric
