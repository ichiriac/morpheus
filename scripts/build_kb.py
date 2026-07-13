#!/usr/bin/env python
"""Génère le contenu RAG versionné : data/kb/<domaine>.md = politique du domaine + signatures
d'outils, dérivés de τ²-bench (MIT © 2025 Sierra Research). Chunké au runtime par
`agents/knowledge.chunk_policy` (BM25) et récupéré *gated par la surprise* (loop.py étape 5).

Usage : python scripts/build_kb.py            # retail + airline
        python scripts/build_kb.py telecom    # un domaine précis (si policy.md dispo)

But : rendre le référentiel de vérité REPRODUCTIBLE et indépendant de l'install τ² (le runner
préfère data/kb/<domaine>.md quand il existe, cf. agents/knowledge.locate_policy).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DOMAINS = ["retail", "airline"]

# Emplacements possibles des données τ² (mêmes que agents/knowledge.locate_policy).
import os

_DATA_ROOTS = [os.environ.get("TAU2_DATA_DIR"), str(REPO / "tau2-bench" / "data"),
               "/workspace/tau2-bench/data"]


def _policy_path(domain: str) -> Path:
    for root in _DATA_ROOTS:
        if not root:
            continue
        p = Path(root) / "tau2" / "domains" / domain / "policy.md"
        if p.is_file():
            return p
    raise FileNotFoundError(f"policy.md introuvable pour {domain!r} (essayé {_DATA_ROOTS})")


def _tool_signatures(domain: str) -> list[str]:
    """`tool(arg1*, arg2) — description` pour chaque outil du domaine (via τ² registry)."""
    from tau2.registry import registry

    tools = registry.get_env_constructor(domain)().get_tools()
    out: list[str] = []
    for t in sorted(tools, key=lambda x: x.name):
        sc = getattr(t, "openai_schema", None) or {}
        fn = sc.get("function", sc)
        params = fn.get("parameters") or {}
        props = list((params.get("properties") or {}).keys())
        req = set(params.get("required") or [])
        sig = ", ".join((f"{p}*" if p in req else p) for p in props)
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        out.append(f"`{t.name}({sig})`" + (f" — {desc[:120]}" if desc else ""))
    # outil synthétique morpheus (non-solo) : parler à l'utilisateur.
    out.append("`respond_to_user(text*)` — [outil morpheus] parler à l'utilisateur "
               "(poser une question, confirmer) plutôt qu'appeler un outil.")
    return out


def build_domain(domain: str) -> Path:
    policy = _policy_path(domain).read_text(encoding="utf-8").rstrip()
    sigs = _tool_signatures(domain)
    header = (
        f"<!-- Contenu RAG dérivé de tau2-bench (MIT © 2025 Sierra Research), domaine "
        f"{domain}. Généré par scripts/build_kb.py — NE PAS éditer à la main. -->\n"
    )
    # Signatures : une puce par ligne SÉPARÉE PAR UNE LIGNE VIDE → chaque outil devient une
    # règle atomique distincte pour le retriever (chunk_policy groupe les lignes contiguës).
    sig_block = "\n\n".join(f"- {s}" for s in sigs)
    body = (
        f"{header}\n{policy}\n\n"
        f"## Signatures des outils\n\n"
        f"Arguments EXACTS de chaque outil (`*` = requis) — utiliser exactement ces noms.\n\n"
        f"{sig_block}\n"
    )
    dst = REPO / "data" / "kb" / f"{domain}.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(body, encoding="utf-8")
    return dst


def main(argv=None) -> int:
    domains = argv or DEFAULT_DOMAINS
    sys.path.insert(0, str(REPO / "src"))
    from morpheus.agents.knowledge import KnowledgeBase

    for domain in domains:
        dst = build_domain(domain)
        kb = KnowledgeBase.from_policy_file(dst, domain)
        print(f"✓ {dst.relative_to(REPO)} — {len(kb)} règles, {len(kb.sections)} sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
