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
DEFAULT_DOMAINS = ["retail", "airline", "telecom"]

# Emplacements possibles des données τ² (mêmes que agents/knowledge.locate_policy).
import os

_DATA_ROOTS = [os.environ.get("TAU2_DATA_DIR"), str(REPO / "tau2-bench" / "data"),
               "/workspace/tau2-bench/data"]

# Fichiers de connaissance par domaine (concaténés). retail/airline ont un `policy.md` ; telecom
# (dual-control, pas de policy.md unique) → policy solo + le MANUEL de dépannage (« si X, fais Y »,
# le vrai contenu à récupérer sur surprise).
DOMAIN_SOURCES = {
    "retail": ["policy.md"],
    "airline": ["policy.md"],
    "telecom": ["main_policy_solo.md", "tech_support_manual.md"],
}
# Domaines joués en SOLO : leurs outils (agent + device) viennent d'un reset gym solo, pas du
# constructeur d'env (qui n'expose que les outils AGENT non-solo). En solo, pas de respond_to_user.
SOLO_DOMAINS = {"telecom"}


def _domain_dir(domain: str) -> Path:
    for root in _DATA_ROOTS:
        if root and (Path(root) / "tau2" / "domains" / domain).is_dir():
            return Path(root) / "tau2" / "domains" / domain
    raise FileNotFoundError(f"domaine τ² {domain!r} introuvable (essayé {_DATA_ROOTS})")


def _policy_text(domain: str) -> str:
    d = _domain_dir(domain)
    files = DOMAIN_SOURCES.get(domain, ["policy.md"])
    parts = [(d / f).read_text(encoding="utf-8").rstrip() for f in files if (d / f).is_file()]
    if not parts:
        raise FileNotFoundError(f"aucune source de policy pour {domain!r} (cherché {files} dans {d})")
    return "\n\n".join(parts)


def _domain_tools(domain: str):
    """Outils du domaine. Solo → reset gym (agent + device) ; sinon → constructeur d'env."""
    from tau2.registry import registry

    if domain in SOLO_DOMAINS:
        from tau2.gym.gym_agent import AgentGymEnv

        tasks = [t for t in registry.get_tasks_loader(domain)() if getattr(t, "ticket", None)]
        env = AgentGymEnv(domain=domain, task_id=tasks[0].id, solo_mode=True, max_steps=5)
        _obs, info = env.reset()
        return info.get("tools", [])
    return registry.get_env_constructor(domain)().get_tools()


def _tool_signatures(domain: str) -> list[str]:
    """`tool(arg1*, arg2) — description` pour chaque outil du domaine."""
    out: list[str] = []
    for t in sorted(_domain_tools(domain), key=lambda x: x.name):
        sc = getattr(t, "openai_schema", None) or {}
        fn = sc.get("function", sc)
        params = fn.get("parameters") or {}
        props = list((params.get("properties") or {}).keys())
        req = set(params.get("required") or [])
        sig = ", ".join((f"{p}*" if p in req else p) for p in props)
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        out.append(f"`{t.name}({sig})`" + (f" — {desc[:120]}" if desc else ""))
    if domain not in SOLO_DOMAINS:  # outil synthétique morpheus (non-solo uniquement).
        out.append("`respond_to_user(text*)` — [outil morpheus] parler à l'utilisateur "
                   "(poser une question, confirmer) plutôt qu'appeler un outil.")
    return out


def build_domain(domain: str) -> Path:
    policy = _policy_text(domain)
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
