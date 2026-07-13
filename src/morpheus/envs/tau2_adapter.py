"""Adaptateur τ²-bench (Sierra) → interface `Env` de morpheus.

τ²-bench expose une interface **gymnasium** (`tau2.gym.AgentGymEnv`) où un contrôleur
externe pilote l'agent pas-à-pas : `reset()` démarre l'orchestrateur τ² (dans un thread),
`step(action_str)` exécute UN coup de l'agent et renvoie `(obs, reward, terminated, ...)`.
C'est le point de raccord idéal pour la boucle MPC de morpheus (un pas réel par tour).

Correspondance (cf. specs/02) :
- `Action(tool, args)`  → chaîne JSON `{"name": tool, "arguments": args}` passée à `gym.step`.
- observation τ²       → `Observation.text` (résultat d'outil / message).
- fin d'épisode        → l'agent appelle l'outil `done` (auto-ajouté par τ²) ⇒ `terminated`.
- reward               → `evaluate_simulation` (0.0 tant que non terminé, valeur réelle à la fin).
- `success`            → `reward >= 1.0` (convention pass@1 de τ²).
- `required_turns`     → nb d'actions **agent** de la trajectoire de référence
                         (`evaluation_criteria`) ⇒ bucket « longueur de tâche » des métriques.

Deux modes :
- **solo** (`tau2_solo: true`) : `DummyUser`, hors-ligne, tool-only. N'accepte que les tâches
  avec un `ticket` (telecom, mock). L'agent reçoit aussi les *user tools*.
- **non-solo** (défaut, retail/airline) : utilisateur simulé par LLM. Renseigner `tau2_user_*`
  (le simulateur peut pointer sur le même vLLM Qwen). ⚠️ la politique de morpheus n'émet que
  des appels d'outils : les tâches exigeant un vrai dialogue avec l'utilisateur seront limitées
  (voir TODO — capacité « répondre à l'utilisateur » à ajouter à l'orchestrateur).

Import PARESSEUX : tant que `tau2` n'est pas installé, seule la construction lève un message
clair, la Phase 1 mock reste exécutable.
"""

from __future__ import annotations

import json

from ..config import EvalConfig
from ..orchestrator.types import Action, Observation, StepResult

# Marqueurs heuristiques d'échec d'outil (instrumentation du routeur de surprise, Phase 1).
_ERROR_MARKERS = ("error", "erreur", "not found", "invalid", "failed", "exception", "cannot")

# Outil SYNTHÉTIQUE « répondre à l'utilisateur » (étape 3). En mode non-solo (retail/airline),
# certaines tâches exigent un vrai dialogue : demander l'id de commande, confirmer un
# remboursement… La politique de morpheus n'émet que des appels d'outils ; on lui expose donc
# cet outil supplémentaire. `step()` l'intercepte et envoie le TEXTE (pas un appel d'outil) à
# `gym.step`, que τ² route vers le simulateur d'utilisateur — dont la réponse devient
# l'observation suivante. `loop.py` ne change pas : la capacité vit à la frontière env, là où un
# utilisateur existe réellement (donc PAS en solo).
RESPOND_TOOL = "respond_to_user"
_TEXT_KEYS = ("text", "message", "content", "reply", "msg")


class Tau2Env:
    """Enveloppe une tâche τ²-bench derrière l'interface `Env` de morpheus."""

    def __init__(
        self,
        task_id: str,
        domain: str,
        *,
        solo: bool = False,
        max_steps: int = 30,
        required_turns: int = -1,
        user_llm: str | None = None,
        user_llm_args: dict | None = None,
    ) -> None:
        from tau2.gym.gym_agent import AgentGymEnv  # import paresseux

        self._domain = domain
        self._task_id = task_id
        self._solo = solo
        self._required_turns = required_turns
        # « répondre à l'utilisateur » : pertinent uniquement quand un user-sim existe (non-solo).
        self._extra_tools: list[str] = [] if solo else [RESPOND_TOOL]
        self._gym = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            solo_mode=solo,
            max_steps=max_steps,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
        )
        self._tools: list[str] = []
        self._goal: str = ""
        self._done: bool = False

    # --- API Env ---
    def reset(self) -> Observation:
        obs, info = self._gym.reset()
        self._tools = [t.name for t in info.get("tools", [])]
        self._goal = _build_goal(info.get("task"), self._solo, self._domain)
        self._done = False
        # En solo, l'observation initiale est vide (le ticket est porté par goal()).
        return Observation(text=obs or "(nouvelle tâche — voir l'objectif)")

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(Observation("épisode terminé"), 0.0, True, {"success": False})

        if action.tool == RESPOND_TOOL and not self._solo:
            # message texte à l'utilisateur : τ² route le CONTENU (pas un appel d'outil) vers le
            # simulateur d'utilisateur ; sa réponse revient comme observation suivante.
            gym_action = _extract_text(action)
        else:
            gym_action = json.dumps({"name": action.tool, "arguments": action.args or {}})

        obs, reward, terminated, _truncated, info = self._gym.step(gym_action)

        self._done = bool(terminated)
        reward = float(reward)
        success = bool(terminated and reward >= 1.0 - 1e-9)
        return StepResult(
            observation=Observation(text=obs or "", tool_error=_looks_like_error(obs)),
            reward=reward,
            done=bool(terminated),
            info={"success": success, "reward": reward},
        )

    def goal(self) -> str:
        return self._goal

    def tool_names(self) -> list[str]:
        return self._tools + self._extra_tools

    def required_turns(self) -> int:
        return self._required_turns

    def close(self) -> None:
        """Termine proprement l'orchestrateur τ² si morpheus s'est arrêté sans appeler `done`
        (sinon le thread τ² reste bloqué en attente d'action)."""
        gym = getattr(self, "_gym", None)
        if gym is None or self._done:
            return
        try:
            gym.step('{"name": "done", "arguments": {}}')
        except Exception:
            pass
        self._done = True


def _looks_like_error(obs: str | None) -> bool:
    if not obs:
        return False
    low = obs.lower()
    return any(m in low for m in _ERROR_MARKERS)


def _extract_text(action: Action) -> str:
    """Texte du message `respond_to_user`. On lit une clé d'args plausible (text/message/…),
    sinon le rationale, sinon les valeurs d'args concaténées. On garantit un contenu NON vide et
    qui ne ressemble pas à un appel d'outil (sinon τ² le re-parserait en tool call)."""
    args = action.args or {}
    text = ""
    for k in _TEXT_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            break
    if not text and action.rationale.strip():
        text = action.rationale.strip()
    if not text and args:
        text = " ".join(str(v) for v in args.values() if str(v).strip())
    if not text:
        text = "Pouvez-vous préciser votre demande ?"
    # évite qu'un texte de forme `foo(...)` soit interprété comme un appel d'outil par τ².
    if text.endswith(")") and "(" in text:
        text = f"Message : {text}"
    return text


# Instruction générique de l'agent en NON-SOLO (aucune fuite du besoin utilisateur). Miroir de
# l'AGENT_INSTRUCTION de τ² : le vrai agent ne voit que la policy + les outils, jamais le scénario.
_NONSOLO_GOAL = (
    "Tu es un agent de service client du domaine {domain}. Aide l'utilisateur à résoudre sa "
    "demande en respectant la politique du domaine et en utilisant les outils. Son besoin n'est "
    "PAS connu d'avance : découvre-le par le dialogue (pose des questions via `respond_to_user`), "
    "vérifie l'identité si nécessaire, puis agis. Termine avec l'outil `done` une fois la demande "
    "résolue."
)


def _build_goal(task, solo: bool, domain: str = "") -> str:
    """Objectif présenté à la politique.

    - **solo** : le `ticket` — brief LÉGITIMEMENT donné à l'agent seul (comme dans τ²).
    - **non-solo** : instruction GÉNÉRIQUE uniquement. ⚠️ NE JAMAIS injecter `user_scenario` :
      c'est le brief PRIVÉ de l'utilisateur (persona, reason_for_call, known/unknown_info,
      task_instructions) — « All the information that will be sent to the user simulator ».
      L'exposer à la politique = fuite : le but contiendrait la réponse, `score_to_goal` serait
      trivialement aligné, et la mesure ne serait pas comparable au leaderboard τ². Le besoin
      doit émerger du dialogue (observations), pas du scénario.
    """
    if solo:
        ticket = getattr(task, "ticket", None) if task is not None else None
        return str(ticket) if ticket else "Résoudre le ticket."
    return _NONSOLO_GOAL.format(domain=domain or "service client")


def _num_agent_actions(task) -> int:
    ec = getattr(task, "evaluation_criteria", None)
    if ec is None:
        return -1
    try:
        return int(ec.info().get("num_agent_actions", -1))
    except Exception:
        return -1


def build_tau2_factory(cfg: EvalConfig):
    """Charge le jeu de tâches τ² une fois, sélectionne un sous-ensemble, et renvoie une
    fabrique `make(task_index) -> Tau2Env`. Filtre les tâches solo-invalides (sans ticket)."""
    try:
        from tau2.registry import registry
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "τ²-bench non installé. Cloner github.com/sierra-research/tau2-bench puis "
            "`pip install -e /chemin/tau2-bench` (+ `pip install gymnasium`). "
            "En attendant, utiliser `env: mock`."
        ) from e

    # τ² logue chaque message en INFO via loguru → on coupe pour ne pas noyer la sortie morpheus.
    try:
        from loguru import logger as _tau2_logger

        _tau2_logger.disable("tau2")
    except Exception:
        pass

    split = cfg.tau2_split
    tasks = registry.get_tasks_loader(cfg.domain)(split) if split else registry.get_tasks_loader(cfg.domain)()

    if cfg.tau2_solo:
        tasks = [t for t in tasks if getattr(t, "ticket", None)]
        if not tasks:
            raise ValueError(
                f"Aucune tâche solo (avec ticket) pour le domaine {cfg.domain!r}. "
                "Le solo n'est dispo que pour telecom/mock ; pour retail/airline, mettre "
                "tau2_solo: false et renseigner tau2_user_llm."
            )

    # sous-ensemble déterministe : les N premières tâches (pas de mélange → reproductible).
    n = min(cfg.tasks, len(tasks))
    selected = tasks[:n]
    ids_turns = [(t.id, _num_agent_actions(t)) for t in selected]

    # user-sim (non-solo) : args litellm pour pointer sur un endpoint OpenAI-compatible.
    user_llm = cfg.tau2_user_llm
    user_llm_args: dict | None = None
    if not cfg.tau2_solo and (cfg.tau2_user_base_url or cfg.tau2_user_llm):
        import os

        user_llm_args = {}
        if cfg.tau2_user_base_url:
            user_llm_args["api_base"] = cfg.tau2_user_base_url
        key = os.environ.get(cfg.tau2_user_api_key_env) if cfg.tau2_user_api_key_env else None
        user_llm_args["api_key"] = key or "EMPTY"
        # user-sim Qwen : couper le mode « thinking » (sinon les réponses user sont polluées par
        # des blocs <think>). extra_body est transmis tel quel par litellm à l'endpoint vLLM.
        user_llm_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    def make(task_index: int) -> Tau2Env:
        task_id, req_turns = ids_turns[task_index % len(ids_turns)]
        return Tau2Env(
            task_id=task_id,
            domain=cfg.domain,
            solo=cfg.tau2_solo,
            max_steps=cfg.tau2_max_steps,
            required_turns=req_turns,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
        )

    return make, n
