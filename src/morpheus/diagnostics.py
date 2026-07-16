"""Diagnostics de branchement LLM (`morpheus check-llm`).

Objectif de l'étape 1 : valider qu'un vrai Qwen (servi par vLLM) répond, et surtout que la
POLITIQUE produit un format d'action parsable sur une tâche mock. On imprime la sortie
BRUTE puis les candidats PARSÉS, pour diagnostiquer d'un coup d'œil format/thinking/snap.
"""

from __future__ import annotations

from .agents.policy import Policy, _parse_actions
from .config import Config
from .envs import build_env_factory
from .llm import build_llm
from .llm.base import system, user
from .orchestrator.types import State


def check_llm(cfg: Config) -> int:
    print(f"[1/3] Ping backend policy : kind={cfg.policy.kind} model={cfg.policy.model} "
          f"base_url={cfg.policy.base_url}")
    llm = build_llm(cfg.policy)
    ping = llm.complete([user("Réponds exactement : OK")])
    print(f"      réponse brute : {ping!r}\n")

    print("[2/3] Construction d'une tâche mock (retail-lite) et appel de la politique...")
    make_env, _n = build_env_factory(cfg.eval)
    env = make_env(0)
    obs = env.reset()
    # `turn`/`max_turns` renseignés : sans eux, `build_prompt` n'émet PAS le bloc [BUDGET] et
    # `check-llm` validerait un prompt que la boucle n'envoie jamais. Le diagnostic doit exercer
    # le prompt RÉEL — c'est tout son intérêt, surtout pour valider le parse sous `enable_thinking`.
    state = State(goal=env.goal(), observation=obs, turn=1,
                  max_turns=cfg.orchestrator.max_turns)
    tools = env.tool_names()
    policy = Policy(llm, k=cfg.orchestrator.k_candidates)

    prompt = policy.build_prompt(state, tools)
    print("      --- PROMPT ENVOYÉ (doit contenir [BUDGET]) ---")
    print(_indent(prompt))
    raw = llm.complete([system_prompt(), user(prompt)])
    print("      --- SORTIE BRUTE DE LA POLITIQUE ---")
    print(_indent(raw))
    print("      --- FIN SORTIE BRUTE ---\n")

    print("[3/3] Parsing (strip raisonnement + snap sur whitelist)...")
    actions = _parse_actions(raw, tools)
    if not actions:
        print("      ⚠️  AUCUNE action parsée. Vérifie le format / le mode thinking.")
        return 1
    for a in actions:
        print(f"      • {a}")
    expected = tools[0]  # 1re étape attendue par le mock
    ok = any(a.tool == expected for a in actions)
    print(f"\n      1re étape attendue par le mock : {expected}")
    print(f"      → la politique la propose : {'OUI ✅' if ok else 'non (à investiguer)'}")
    return 0 if ok else 2


def system_prompt() -> object:
    from .agents.policy import _SYS

    return system(_SYS)


def _indent(text: str, pad: str = "      | ") -> str:
    return "\n".join(pad + line for line in text.splitlines())
