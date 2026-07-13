"""Base de connaissance (KB) = référentiel de vérité, récupéré *gated par la surprise*.

Corpus de départ : les **policy documents** de τ²-bench (retail/telecom/airline). Chaque
`policy.md` est une liste d'outils + des règles conditionnelles « dans tel cas, fais ceci /
tu ne peux X que si Y » — exactement la classe de connaissance qui attrape le
*cohérent-mais-faux* que JEPA ne peut pas voir (cf. specs/00 §« Le rôle de la connaissance »).

Deux briques :
- **Chunker** : découpe le markdown de la policy en **règles atomiques** (un paragraphe =
  une règle), en gardant le chemin de titres comme métadonnée (`section`).
- **Retriever lexical pondéré IDF** : score = somme des IDF des tokens partagés entre la
  requête et la règle. Volontairement MINIMAL et sans dépendance — même esprit « proxy
  Phase-1 » que `surprise.divergence` (Jaccard). En Phase 4, remplaçable par un retriever
  dense (sentence-transformers, déjà dispo via l'extra `[jepa]`) sans changer l'interface.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9_]+")

# Paramètres BM25 standards.
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


@dataclass(frozen=True)
class Rule:
    """Une règle atomique de la KB (un paragraphe de la policy + son chemin de titres)."""

    domain: str
    section: str      # chemin de titres, ex. "Cancel pending order"
    text: str

    def as_fact(self) -> str:
        """Rendu compact pour la trace / le prompt de replanification."""
        head = f"[{self.domain}:{self.section}] " if self.section else f"[{self.domain}] "
        return head + " ".join(self.text.split())


class KnowledgeBase:
    """Règles chunkées + retriever lexical IDF. Immuable après construction."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules
        self._tf: list[Counter[str]] = [Counter(_tokens(r.text)) for r in rules]
        self._len: list[int] = [sum(c.values()) for c in self._tf]
        n = len(rules) or 1
        self._avgdl = (sum(self._len) / n) or 1.0
        # IDF BM25 (toujours > 0 grâce au +1) : les termes de domaine rares (cancelled,
        # gift_card, pending…) pèsent plus que les mots vides → matching qui « colle » au fond.
        df: dict[str, int] = {}
        for c in self._tf:
            for t in c:
                df[t] = df.get(t, 0) + 1
        self._idf: dict[str, float] = {
            t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def sections(self) -> list[str]:
        seen: list[str] = []
        for r in self.rules:
            if r.section not in seen:
                seen.append(r.section)
        return seen

    def score(self, query: str) -> list[tuple[float, Rule]]:
        """Score BM25 de chaque règle contre la requête, trié décroissant (score>0 d'abord).
        La normalisation par longueur (paramètre `b`) évite que les règles longues raflent
        tout : les règles conditionnelles nettes (« only … if status is 'pending' ») remontent."""
        q = set(_tokens(query))
        scored: list[tuple[float, Rule]] = []
        for tf, dl, rule in zip(self._tf, self._len, self.rules):
            denom_norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / self._avgdl)
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * (_BM25_K1 + 1) / (f + denom_norm)
            scored.append((s, rule))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def retrieve(self, query: str, k: int = 3) -> list[Rule]:
        """Les `k` règles les plus pertinentes (score > 0). Vide si rien ne matche."""
        return [rule for s, rule in self.score(query) if s > 0.0][:k]

    # --- construction ---

    @classmethod
    def from_text(cls, markdown: str, domain: str) -> "KnowledgeBase":
        return cls(chunk_policy(markdown, domain))

    @classmethod
    def from_policy_file(cls, path: str | Path, domain: str) -> "KnowledgeBase":
        return cls.from_text(Path(path).read_text(encoding="utf-8"), domain)


def chunk_policy(markdown: str, domain: str) -> list[Rule]:
    """Découpe une policy markdown en règles atomiques.

    Un « bloc » = un run de lignes non vides ; les titres (`#`…`####`) ne sont pas des
    règles mais mettent à jour le chemin de section courant. Les listes à puces
    consécutives restent groupées dans le bloc qui les porte (elles forment une règle).
    """
    rules: list[Rule] = []
    heading: dict[int, str] = {}
    block: list[str] = []

    def section_path() -> str:
        # chemin des titres de niveau ≥ 2 (on ignore le H1 = titre du document)
        parts = [heading[lvl] for lvl in sorted(heading) if lvl >= 2 and heading.get(lvl)]
        return " > ".join(parts)

    def flush() -> None:
        if not block:
            return
        text = "\n".join(block).strip()
        block.clear()
        if text:
            rules.append(Rule(domain=domain, section=section_path(), text=text))

    for raw in markdown.splitlines():
        line = raw.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            heading[level] = m.group(2).strip()
            # invalider les sous-titres plus profonds (nouveau contexte)
            for deeper in [lvl for lvl in heading if lvl > level]:
                heading.pop(deeper, None)
            continue
        if not line.strip():
            flush()
            continue
        block.append(line)
    flush()
    return rules


# --- localisation du policy.md τ² ------------------------------------------------------

_DOMAINS = ("retail", "telecom", "airline", "mock")


def locate_policy(domain: str, explicit: str | None = None,
                  data_dir: str | None = None) -> Path:
    """Trouve le contenu KB d'un domaine. Ordre d'essai :
    1. `explicit` (chemin direct fourni en config) ;
    2. **`data/kb/<domaine>.md`** — contenu RAG VERSIONNÉ dans le repo (policy + signatures
       d'outils, généré par `scripts/build_kb.py`) : préféré → reproductible, sans dépendance τ² ;
    3. `data_dir`, puis `$TAU2_DATA_DIR`, puis `./tau2-bench/data` — le `policy.md` τ² brut, en
       repli si le KB versionné n'a pas encore été généré.
    Lève une erreur claire listant les chemins essayés si rien n'est trouvé.
    """
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"kb_policy_path introuvable : {p}")

    tried: list[Path] = []
    # 2. KB versionné (repo) — relatif au CWD ; couvre aussi l'exécution depuis la racine repo.
    versioned = Path("data") / "kb" / f"{domain}.md"
    tried.append(versioned)
    if versioned.is_file():
        return versioned

    # 3. repli : policy.md τ² brut.
    roots = [data_dir, os.environ.get("TAU2_DATA_DIR"), "tau2-bench/data"]
    for root in roots:
        if not root:
            continue
        cand = Path(root) / "tau2" / "domains" / domain / "policy.md"
        tried.append(cand)
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"contenu KB introuvable pour le domaine {domain!r}. Chemins essayés : "
        + ", ".join(str(t) for t in tried)
        + ". Générer `data/kb/<domaine>.md` via `python scripts/build_kb.py`, ou renseigner "
        "eval.kb_policy_path / eval.tau2_data_dir (ou $TAU2_DATA_DIR)."
    )
