"""Archivage markdown des résultats de bench.

Deux sorties à chaque `morpheus run` :
- `<out_dir>/results.md` : rapport détaillé d'UN run (métadonnées + courbe réussite-vs-tours).
- `BENCHMARKS.md` (racine repo, versionné) : journal CUMULATIF — une ligne par run, pour garder
  la trace de tous les benchs sans écraser les précédents (« mettre de côté les résultats »).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from .metrics import SuccessVsTurns

BENCHMARKS_FILE = "BENCHMARKS.md"

_HEADER = (
    "# Résultats de bench morpheus\n\n"
    "Journal cumulatif (une ligne par run). La métrique qui tranche = réussite **vs nombre de "
    "tours** ; la thèse veut voir la courbe *world-model* diverger de la baseline à 8+ tours.\n\n"
    "| Date (UTC) | Run | Env / domaine | Mode | Variante | Modèle | K/H/Tmax | Tâches | Réussite | Courbe (tours:réussite) |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _variant(cfg: Config) -> str:
    return "world-model" if cfg.orchestrator.use_world_model else "baseline"


def _mode(cfg: Config) -> str:
    if cfg.eval.env != "tau2":
        return "—"
    return "solo" if cfg.eval.tau2_solo else "user-sim"


def _curve_compact(metric: SuccessVsTurns) -> str:
    return " · ".join(f"{b}:{rate:.0%}(n{n})" for b, rate, n in metric.curve()) or "—"


def render_run_markdown(cfg: Config, metric: SuccessVsTurns, out_dir: str | Path,
                        started_at: str | None = None) -> str:
    """Rapport markdown détaillé d'un run."""
    e, o = cfg.eval, cfg.orchestrator
    lines = [
        f"# Bench — {Path(out_dir).name}",
        "",
        f"- **Date** : {started_at or _now_iso()} UTC",
        f"- **Env / domaine** : `{e.env}` / `{e.domain}`" + (f" ({_mode(cfg)})" if e.env == "tau2" else ""),
        f"- **Variante** : {_variant(cfg)} (`use_world_model={o.use_world_model}`)",
        f"- **Modèle politique** : `{cfg.policy.model}`",
        f"- **K / horizon / max_turns** : {o.k_candidates} / {o.horizon} / {o.max_turns}"
        f" · concurrency={o.concurrency}",
        f"- **Tâches** : {metric.n} · seed={e.seed}",
        f"- **Réussite globale** : **{metric.overall:.1%}**",
        "",
        "## Réussite vs nombre de tours",
        "",
        "| Tours (réf.) | Réussite | n |",
        "|---|---|---|",
    ]
    for bucket, rate, n in metric.curve():
        lines.append(f"| {bucket} | {rate:.1%} | {n} |")
    lines.append(f"| **global** | **{metric.overall:.1%}** | **{metric.n}** |")
    lines.append("")
    return "\n".join(lines)


def append_benchmark_row(cfg: Config, metric: SuccessVsTurns, out_dir: str | Path,
                         started_at: str | None = None,
                         path: str | Path = BENCHMARKS_FILE) -> None:
    """Ajoute une ligne récapitulative au journal cumulatif (crée l'en-tête si absent)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        p.write_text(_HEADER, encoding="utf-8")
    row = (
        f"| {started_at or _now_iso()} | `{Path(out_dir).name}` | {cfg.eval.env}/{cfg.eval.domain} "
        f"| {_mode(cfg)} | {_variant(cfg)} | `{cfg.policy.model}` "
        f"| {cfg.orchestrator.k_candidates}/{cfg.orchestrator.horizon}/{cfg.orchestrator.max_turns} "
        f"| {metric.n} | {metric.overall:.1%} | {_curve_compact(metric)} |\n"
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(row)


def write_reports(cfg: Config, metric: SuccessVsTurns, out_dir: str | Path,
                  started_at: str | None = None,
                  bench_path: str | Path = BENCHMARKS_FILE) -> None:
    """Écrit le rapport par-run ET met à jour le journal cumulatif."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(
        render_run_markdown(cfg, metric, out_dir, started_at), encoding="utf-8"
    )
    append_benchmark_row(cfg, metric, out_dir, started_at, path=bench_path)
