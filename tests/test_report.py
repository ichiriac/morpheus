"""Tests de l'archivage markdown des résultats de bench (eval/report.py)."""

from __future__ import annotations

from morpheus.config import Config
from morpheus.eval.metrics import SuccessVsTurns
from morpheus.eval.report import (BENCH_ANCHOR, append_benchmark_row, render_run_markdown,
                                  write_reports)


def _metric() -> SuccessVsTurns:
    m = SuccessVsTurns()
    for rt, ok in [(4, True), (4, False), (8, True), (12, False)]:
        m.add(rt, ok)
    return m


def test_render_run_markdown_has_curve():
    md = render_run_markdown(Config(), _metric(), "runs/demo", started_at="2026-07-13 12:00")
    assert "Réussite vs nombre de tours" in md
    assert "| 4 |" in md and "| 8 |" in md and "| 12 |" in md
    assert "world-model" in md            # Config() par défaut : use_world_model=True


def test_append_benchmark_row_accumulates(tmp_path):
    bench = tmp_path / "BENCHMARKS.md"
    append_benchmark_row(Config(), _metric(), "runs/a", "2026-07-13 12:00", path=bench)
    append_benchmark_row(Config(), _metric(), "runs/b", "2026-07-13 12:05", path=bench)
    txt = bench.read_text(encoding="utf-8")
    assert "| `a` |" in txt and "| `b` |" in txt   # deux runs cumulés (basenames)
    assert txt.count("world-model |") == 2         # deux lignes de données
    assert txt.startswith("# Résultats de bench morpheus")


def test_row_lands_inside_table_even_with_prose_below(tmp_path):
    """NON-RÉGRESSION : c'est le bug qui a rendu 2 lignes du 13/07 invisibles pendant des
    semaines (dont le 100% telecom). L'ancien `open("a")` écrivait en FIN DE FICHIER ; dès qu'il
    y a de la prose sous le tableau, la ligne atterrit derrière et n'est plus rendue."""
    bench = tmp_path / "BENCHMARKS.md"
    append_benchmark_row(Config(), _metric(), "runs/a", "2026-07-13 12:00", path=bench)
    # quelqu'un ajoute de la prose SOUS le tableau (blocs d'avertissement, section « Signal »…)
    with bench.open("a", encoding="utf-8") as f:
        f.write("\n## Une section ajoutée après coup\n\nDe la prose.\n")
    append_benchmark_row(Config(), _metric(), "runs/b", "2026-07-13 12:05", path=bench)

    txt = bench.read_text(encoding="utf-8")
    assert txt.index("| `b` |") < txt.index("## Une section ajoutée après coup"), (
        "la ligne du 2e run a atterri APRÈS la prose ⇒ hors du tableau (le bug d'origine)"
    )
    assert txt.index("| `a` |") < txt.index("| `b` |")      # ordre chronologique préservé
    assert txt.count(BENCH_ANCHOR) == 1                     # l'ancre ne se duplique pas


def test_append_without_anchor_still_records_and_warns(tmp_path, capsys):
    """Un journal antérieur à l'ancre ne doit JAMAIS faire perdre un résultat (des heures de
    GPU) — on écrit quand même, mais on prévient bruyamment."""
    bench = tmp_path / "BENCHMARKS.md"
    bench.write_text("# Journal legacy\n\n| Date | Run |\n|---|---|\n", encoding="utf-8")
    append_benchmark_row(Config(), _metric(), "runs/legacy", "2026-07-13 12:00", path=bench)
    assert "| `legacy` |" in bench.read_text(encoding="utf-8")
    assert "ancre" in capsys.readouterr().out


def test_write_reports_creates_results_md(tmp_path):
    out = tmp_path / "run1"
    write_reports(Config(), _metric(), out, started_at="2026-07-13 12:00",
                  bench_path=tmp_path / "BENCHMARKS.md")
    assert (out / "results.md").exists()
    assert (tmp_path / "BENCHMARKS.md").exists()
