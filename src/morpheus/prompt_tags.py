"""Balises des blocs de prompt — source UNIQUE, partagée producteurs ⇄ lecteurs.

Les prompts (`agents/policy.py`, `agents/world_model.py`) écrivent des blocs
``[BALISE]contenu[/BALISE]`` que le backend déterministe (`llm/stub.py`) relit pour décider.
Producteur et lecteur doivent donc nommer la balise à l'identique — d'où ce module.

Pourquoi il existe : la politique émettait `[ÉTAT COURANT]` pendant que le stub lisait
`[STATE]`. Deux littéraux dupliqués qui ont divergé sans que rien n'échoue : l'état lu était
vide ⇒ l'indice « Prochaine étape attendue : X » ne matchait jamais ⇒ le stub retombait sur sa
branche à UNE seule action ⇒ `loop.py` (qui exige `len(candidates) > 1`) ne déclenchait jamais
le lookahead. Toute config à politique stub tournait donc à K=1, sans MPC, sans divergence et
sans routage de surprise, silencieusement. Redupliquer un littéral ici rejouerait ce bug.

`STATE` et `POLICY_STATE` restent DEUX balises distinctes à dessein : ces littéraux partent
dans les prompts des vrais LLM, et les uniformiser changerait les prompts de production — donc
la comparabilité des runs de BENCHMARKS.md — pour une raison purement cosmétique.
"""

from __future__ import annotations

# --- Blocs de contenu (écrits via `block()`, relus via le lecteur du stub) ---
GOAL = "GOAL"
STATE = "STATE"                  # état vu par le world-model (predict / score_to_goal)
POLICY_STATE = "ÉTAT COURANT"    # état vu par la politique (PROPOSER)
BUDGET = "BUDGET"                # tour courant / budget de tours (PROPOSER seulement)
ACTION = "ACTION"
PREDICTED = "PREDICTED"
REAL = "REAL"
CANDIDATE_TOOLS = "CANDIDATE_TOOLS"

# --- Marqueurs de mode : en-tête nu qui dit au stub quel rôle on lui demande ---
PREDICT_NEXT_STATE = "[PREDICT_NEXT_STATE]"
EXPLAIN_GAP = "[EXPLAIN_GAP]"
SCORE_GOAL_DISTANCE = "[SCORE_GOAL_DISTANCE]"


def block(tag: str, content: object) -> str:
    """Rend `[tag]content[/tag]` — le pendant écriture du lecteur de `llm/stub.py`."""
    return f"[{tag}]{content}[/{tag}]"
