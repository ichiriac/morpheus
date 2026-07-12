"""Métriques. LA métrique de morpheus : réussite de tâche en fonction du nb de tours.

C'est la courbe que l'architecture doit redresser à 8+ tours (cf. specs/00, specs/01).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SuccessVsTurns:
    """Agrège les épisodes par « longueur de référence » (bucket) de la tâche."""

    by_bucket_total: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    by_bucket_success: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    n: int = 0
    n_success: int = 0

    def add(self, required_turns: int, success: bool) -> None:
        self.n += 1
        self.n_success += int(success)
        self.by_bucket_total[required_turns] += 1
        self.by_bucket_success[required_turns] += int(success)

    def curve(self) -> list[tuple[int, float, int]]:
        """[(bucket, taux_de_réussite, n)] trié par longueur croissante."""
        out = []
        for b in sorted(self.by_bucket_total):
            tot = self.by_bucket_total[b]
            out.append((b, self.by_bucket_success[b] / tot if tot else 0.0, tot))
        return out

    @property
    def overall(self) -> float:
        return self.n_success / self.n if self.n else 0.0


def summarize(metric: SuccessVsTurns) -> str:
    lines = ["Réussite vs nombre de tours (bucket = longueur de référence de la tâche) :"]
    for bucket, rate, n in metric.curve():
        bar = "#" * round(rate * 20)
        lines.append(f"  {bucket:>3} tours | {rate:5.1%} | n={n:<3} {bar}")
    lines.append(f"  Global    | {metric.overall:5.1%} | n={metric.n}")
    return "\n".join(lines)
