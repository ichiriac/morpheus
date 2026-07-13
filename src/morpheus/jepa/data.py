"""Normalisation des trajectoires vers des transitions `(obs, action, obs')`. SANS torch.

Le prédicteur JEPA a besoin de l'ÉTAT RÉSULTANT (la réponse de l'outil / l'observation
suivante) — pas seulement de l'appel. On convertit donc des conversations outillées en
transitions atomiques.

Sources gérées :
- `synthetic` : générateur jouet (chaîne d'outils) — aucun téléchargement, pour smoke tests.
- `messages`  : conversations façon OpenAI (`role`+`tool_calls`, `role="tool"`) OU ShareGPT
                (`from`/`value` avec from ∈ {gpt, function_call, observation, tool, human…}).
                Couvre APIGen-MT-5k et la plupart des datasets multi-tours.
- `alfworld`  : étapes (observation, action) → transitions consécutives.
- `jsonl`     : fichier de dicts {obs, action, next_obs, ...} déjà normalisés.

⚠️ Les schémas HF exacts peuvent varier : `from_messages` est défensif et documenté.
Vérifier sur quelques exemples réels (`describe_records`) avant un gros run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Transition:
    obs: str                       # état AVANT l'action (contexte accumulé)
    action: str                    # appel d'outil (nom + args), sérialisé
    next_obs: str                  # état résultant (réponse d'outil / observation suivante)
    reward: float = 0.0
    done: bool = False
    # --- signal goal-relative (Phase 2, fix du signal goal) ---
    goal: str = ""                 # objectif de la trajectoire (requête user / état résolu visé)
    progress: float = 0.0          # position normalisée dans la trajectoire ∈ [0,1] (0=début, 1=résolu)
    traj_id: int = -1              # identité de la trajectoire (groupe les pas ; négatifs InfoNCE)
    meta: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return bool(self.action.strip()) and bool(self.next_obs.strip())


# --------------------------------------------------------------------------- #
# Générateur synthétique (smoke tests, aucune dépendance)
# --------------------------------------------------------------------------- #

_CHAIN = [
    "authenticate_user", "lookup_order", "check_refund_policy",
    "verify_payment_method", "compute_refund_amount", "issue_refund",
]


def synthetic_transitions(n_episodes: int = 50, seed: int = 0) -> list[Transition]:
    """Trajectoires jouet type retail-lite. Déterministe (varie par index, pas de random)."""
    out: list[Transition] = []
    for ep in range(n_episodes):
        length = 2 + (ep % (len(_CHAIN) - 1))  # 2..len
        chain = _CHAIN[:length]
        # objectif de la trajectoire : l'état résolu visé (dernière étape). Diffère par
        # chaîne d'outils → le signal goal-relative ne peut PAS se réduire au n° de ticket.
        goal = f"Objectif ticket #{ep} : exécuter {chain[-1]} et résoudre la demande."
        obs = f"Ticket #{ep}. Prochaine étape attendue : {chain[0]}."
        for i, tool in enumerate(chain):
            done = i == length - 1
            nxt = (
                f"Étape {tool} OK. Demande résolue."
                if done else
                f"Étape {tool} OK. Prochaine étape attendue : {chain[i + 1]}."
            )
            out.append(Transition(
                obs=obs, action=f"{tool}()", next_obs=nxt,
                reward=1.0 if done else 0.1, done=done,
                goal=goal, progress=(i + 1) / length, traj_id=ep,
                meta={"episode": ep, "step": i},
            ))
            obs = nxt
    return out


# --------------------------------------------------------------------------- #
# Normalisation de conversations outillées (OpenAI / ShareGPT) → transitions
# --------------------------------------------------------------------------- #

_ASSISTANT_ROLES = {"assistant", "gpt", "function_call"}
_OBS_ROLES = {"tool", "observation", "function_response", "tool_response", "function"}
_CONTEXT_ROLES = {"user", "human", "system"}


def _role_and_content(msg: dict) -> tuple[str, str, Any]:
    """Retourne (role, texte, tool_calls) en tolérant OpenAI et ShareGPT."""
    role = msg.get("role") or msg.get("from") or ""
    content = msg.get("content")
    if content is None:
        content = msg.get("value", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    tool_calls = msg.get("tool_calls")
    return role.lower(), content, tool_calls


def _extract_action(role: str, content: str, tool_calls: Any) -> str | None:
    """Sérialise l'appel d'outil d'un message assistant, s'il y en a un."""
    if tool_calls:
        parts = []
        for tc in tool_calls:
            fn = (tc.get("function") or tc) if isinstance(tc, dict) else {}
            name = fn.get("name", "tool")
            args = fn.get("arguments", "")
            parts.append(f"{name}({args})")
        return " ; ".join(parts)
    if role == "function_call":
        # ShareGPT : le contenu EST l'appel (souvent un JSON {name, arguments})
        return content.strip() or None
    return None


def from_messages(messages: Iterable[dict], reward_key: str | None = None,
                  traj_id: int = -1) -> list[Transition]:
    """Parcourt une conversation et émet une transition par (contexte → appel → réponse).

    Le `goal` de la trajectoire = la 1re requête utilisateur (façon τ² : l'objectif est donné
    en tête, PAS l'état terminal → pas de token-leak). `progress` = position normalisée du pas
    dans la conversation (0=début … 1=résolu), pour l'alignement goal-relative de la perte."""
    transitions: list[Transition] = []
    context: list[str] = []
    pending_action: str | None = None
    pending_ctx: str = ""
    goal: str = ""

    for msg in messages:
        role, content, tool_calls = _role_and_content(msg)

        if role in _CONTEXT_ROLES and role in {"user", "human"} and not goal and content.strip():
            goal = content.strip()          # 1re requête user = objectif de la trajectoire

        if role in _ASSISTANT_ROLES:
            action = _extract_action(role, content, tool_calls)
            if action:
                pending_action = action
                pending_ctx = "\n".join(context[-8:])  # contexte = 8 derniers tours
            if content:
                context.append(f"assistant: {content}")
            continue

        if role in _OBS_ROLES:
            if pending_action is not None:
                transitions.append(Transition(
                    obs=pending_ctx, action=pending_action, next_obs=content,
                ))
                pending_action = None
            context.append(f"tool: {content}")
            continue

        if role in _CONTEXT_ROLES:
            context.append(f"{role}: {content}")

    # progress croissant sur l'état RÉSULTANT (next_obs) : dernier pas = état résolu (1.0).
    n = len(transitions)
    for k, t in enumerate(transitions):
        t.goal = goal
        t.progress = (k + 1) / n if n else 0.0
        t.traj_id = traj_id
        t.done = (k == n - 1)
    return transitions


def from_alfworld_steps(steps: Iterable[dict], obs_key: str = "observation",
                        act_key: str = "action", traj_id: int = -1) -> list[Transition]:
    """Étapes (observation, action) consécutives → transitions (obs_i, act_i, obs_{i+1})."""
    steps = list(steps)
    out: list[Transition] = []
    n = max(1, len(steps) - 1)
    goal = str(steps[0].get(obs_key, "")) if steps else ""   # 1re observation = brief/tâche
    for i in range(len(steps) - 1):
        out.append(Transition(
            obs=str(steps[i].get(obs_key, "")),
            action=str(steps[i].get(act_key, "")),
            next_obs=str(steps[i + 1].get(obs_key, "")),
            done=i + 1 == len(steps) - 1,
            goal=goal, progress=(i + 1) / n, traj_id=traj_id,
        ))
    return out


# --------------------------------------------------------------------------- #
# Chargement haut-niveau
# --------------------------------------------------------------------------- #

def _from_jsonl(path: str) -> list[Transition]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Transition(**{k: d[k] for k in ("obs", "action", "next_obs") if k in d},
                                   reward=d.get("reward", 0.0), done=d.get("done", False),
                                   goal=d.get("goal", ""), progress=float(d.get("progress", 0.0)),
                                   traj_id=int(d.get("traj_id", -1)), meta=d.get("meta", {})))
    return out


def _messages_field(record: dict) -> list[dict] | None:
    for key in ("messages", "conversations", "conversation"):
        if isinstance(record.get(key), list):
            return record[key]
    return None


def load_transitions(source: str, *, hf_split: str = "train", limit: int | None = None,
                     **kwargs) -> list[Transition]:
    """Dispatch. `source` ∈ {synthetic, jsonl:<path>, hf:<dataset_name>}.

    Pour HF, on charge via `datasets` (import paresseux) et on normalise chaque enregistrement
    avec `from_messages` (ou `from_alfworld_steps` si `kwargs['alfworld']=True`)."""
    if source == "synthetic":
        return synthetic_transitions(**{k: v for k, v in kwargs.items()
                                        if k in ("n_episodes", "seed")})
    if source.startswith("jsonl:"):
        return _from_jsonl(source[len("jsonl:"):])
    if source.startswith("hf:"):
        return _load_hf(source[len("hf:"):], hf_split, limit, **kwargs)
    raise ValueError(f"source inconnue : {source!r}")


def _load_hf(name: str, split: str, limit: int | None, alfworld: bool = False,
             steps_key: str | None = None, **_) -> list[Transition]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("chargement HF : `pip install datasets`") from e

    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out: list[Transition] = []
    for tid, rec in enumerate(ds):        # tid = identité de trajectoire (négatifs InfoNCE)
        if alfworld and steps_key and isinstance(rec.get(steps_key), list):
            out += from_alfworld_steps(rec[steps_key], traj_id=tid)
            continue
        msgs = _messages_field(rec)
        if msgs:
            out += from_messages(msgs, traj_id=tid)
    return [t for t in out if t.is_valid()]


def describe_records(transitions: list[Transition], k: int = 3) -> str:
    """Aperçu lisible pour vérifier la normalisation avant un gros run."""
    lines = [f"{len(transitions)} transitions. Exemples :"]
    for t in transitions[:k]:
        lines += [
            "  ── obs      : " + t.obs[:90].replace("\n", " ⏎ "),
            "     action   : " + t.action[:90],
            "     next_obs : " + t.next_obs[:90].replace("\n", " ⏎ "),
        ]
    return "\n".join(lines)


def to_jsonl(transitions: list[Transition], path: str) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for t in transitions:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
