"""Runner : instancie les composants depuis la config, joue N tâches, écrit les résultats."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..agents.knowledge import KnowledgeBase, locate_policy
from ..agents.policy import Policy
from ..agents.surprise import Router, SurpriseRouter
from ..agents.world_model import WorldModel
from ..config import Config, OrchestratorConfig
from ..envs import build_env_factory
from ..llm import build_llm
from ..orchestrator.loop import Orchestrator
from .metrics import SuccessVsTurns, summarize
from .report import _now_iso, write_reports


def build_router(cfg: OrchestratorConfig) -> Router:
    """Règle Phase 1 par défaut ; routeur APPRIS si `router_checkpoint` (Phase 4).

    Refuse un régime de sondage incompatible avec l'entraînement : `as_vector()` épingle à 0
    les signaux non sondés, ce que le modèle lit comme une valeur RÉELLE hors distribution ⇒
    dérive constante du logit. Échouer ICI (avant un run GPU de plusieurs heures) plutôt que
    de produire une mesure biaisée et non comparable.
    """
    if not cfg.router_checkpoint:
        return SurpriseRouter()

    from ..router.model import MAX_REGIME_DRIFT, RouterModel   # numpy : extra [jepa]/[dev]

    model = RouterModel.load(cfg.router_checkpoint)
    if not cfg.use_world_model:
        # Baseline nue : pas de lookahead ⇒ `predicted` reste None ⇒ la boucle n'appelle JAMAIS
        # route(). Le routeur est INERTE : juger son régime n'aurait aucun sens (`direction`
        # n'est pas sondée *par construction*, ce n'est pas un désaccord). On charge quand même
        # — un chemin de checkpoint erroné doit échouer ici — mais sans garde ni bruit. Sans ce
        # court-circuit, `--no-world-model` sur une config PORTANT un router_checkpoint (le cas
        # de la mesure baseline-vs-JEPA-WM, même YAML) échouerait pour un routeur non utilisé.
        print(f"routeur : APPRIS chargé mais INERTE ({cfg.router_checkpoint} — "
              f"--no-world-model : aucune surprise routée)")
        return model

    live = {"use_rag": cfg.use_rag, "use_memory": cfg.use_memory,
            "use_reducibility": cfg.use_reducibility, "use_world_model": cfg.use_world_model}
    drift, detail = model.regime_drift(**live)

    if abs(drift) >= MAX_REGIME_DRIFT:
        toward = "ERREUR" if drift > 0 else "NOUVEAUTÉ"
        per_feat = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(detail.items()))
        raise ValueError(
            f"routeur appris {cfg.router_checkpoint!r} : régime de sondage ≠ entraînement.\n"
            f"  Dérive systématique du logit : {drift:+.2f} vers {toward}, à CHAQUE décision.\n"
            f"  Features épinglées à 0 mais sondées à l'entraînement : {per_feat}\n"
            f"  ⇒ activez les drapeaux correspondants (orchestrator.use_rag / use_memory / …) "
            f"pour retrouver le régime d'entraînement, OU ré-entraînez le routeur sur ce "
            f"régime (scripts/build_router_dataset.py --no-kb / --no-memory)."
        )

    blind = model.unused_live_signals(**live)
    if blind:
        print(f"  ⚠️  routeur appris : signaux sondés en live mais IGNORÉS (poids 0 — jamais "
              f"sondés à l'entraînement) : {', '.join(blind)}. Re-générer le dataset avec "
              f"--checkpoint pour que le routeur les exploite.")
    if drift:
        print(f"  ℹ️  routeur appris : dérive de régime {drift:+.2f} logit "
              f"(< {MAX_REGIME_DRIFT}, sous le seuil de décision).")
    ba = model.meta.get("cv", {}).get("balanced_accuracy")
    print(f"routeur : APPRIS ({cfg.router_checkpoint}"
          + (f", balanced_acc CV {ba}" if ba is not None else "") + ")")
    return model


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

    orch = Orchestrator(policy, world_model, cfg.orchestrator,
                        build_router(cfg.orchestrator), kb=kb)
    # δ n'a pas la même échelle selon le world-model ⇒ le seuil lui appartient. On ANNONCE la
    # valeur résolue et son origine : un run doit dire sous quel régime il a tourné.
    origin = ("config" if cfg.orchestrator.surprise_threshold is not None
              else f"défaut de {type(world_model).__name__}")
    print(f"world-model : {type(world_model).__name__} · seuil de surprise "
          f"{orch.surprise_threshold:.2f} ({origin})")

    make_env, n_tasks = build_env_factory(cfg.eval)
    metric = SuccessVsTurns()

    out = Path(out_dir or cfg.eval.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "episodes.jsonl"

    tag = "WM" if cfg.orchestrator.use_world_model else "baseline"
    with trace_path.open("w", encoding="utf-8") as f:
        for i in range(n_tasks):
            env = make_env(i)
            final_reward = None
            try:
                result = orch.run(env)
            finally:
                # τ² : termine le thread orchestrateur si la boucle s'est arrêtée sans `done`, et
                # récupère le reward FINAL (état DB) — sinon une tâche faite-mais-non-conclue au
                # plafond de tours serait comptée échec à tort.
                close = getattr(env, "close", None)
                if callable(close):
                    final_reward = close()
            bucket = env.required_turns()
            # succès = la boucle a conclu avec succès, OU la terminaison forcée révèle une DB correcte.
            success = bool(result.success or (final_reward is not None and final_reward >= 1.0 - 1e-9))
            metric.add(bucket, success)
            # Progression en direct (suivi/arrêt anticipé) : une ligne par tâche, flushée.
            print(
                f"[{tag}] {i + 1}/{n_tasks} · req={bucket} turns={result.turns} "
                f"ok={success} · réussite {metric.n_success}/{metric.n} "
                f"({metric.overall:.0%})",
                flush=True,
            )
            f.write(json.dumps({
                "task": i,
                "goal": env.goal(),   # persisté : requis pour rejouer score_to_goal (validation étape 4)
                "required_turns": bucket,
                "success": success,
                "turns": result.turns,
                "success_via_close": bool(not result.success and success),
                "total_reward": result.total_reward,
                "reward_breakdown": getattr(env, "reward_breakdown", lambda: None)(),
                "trace": [asdict(s) for s in result.trace],
            }, ensure_ascii=False) + "\n")
            f.flush()

    (out / "summary.txt").write_text(summarize(metric), encoding="utf-8")
    cfg_dump = cfg.to_dict()
    # `surprise_threshold: null` signifie « le world-model décide » : sans la valeur RÉSOLUE, le
    # run ne dirait plus sous quel seuil il a tourné. On la fige à côté (lecture humaine ; rien
    # ne relit ce fichier programmatiquement).
    cfg_dump["orchestrator"]["surprise_threshold_resolved"] = orch.surprise_threshold
    (out / "config.json").write_text(
        json.dumps(cfg_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Résultats mis de côté en markdown : rapport par-run + journal cumulatif BENCHMARKS.md.
    write_reports(cfg, metric, out, started_at)
    return metric
