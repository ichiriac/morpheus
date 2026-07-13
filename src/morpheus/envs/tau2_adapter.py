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
        self._goal = _build_goal(info.get("task"))
        self._done = False
        # En solo, l'observation initiale est vide (le ticket est porté par goal()).
        return Observation(text=obs or "(nouvelle tâche — voir l'objectif)")

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(Observation("épisode terminé"), 0.0, True, {"success": False})

        payload = json.dumps({"name": action.tool, "arguments": action.args or {}})
        obs, reward, terminated, _truncated, info = self._gym.step(payload)

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
        return self._tools

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


def _build_goal(task) -> str:
    """Objectif présenté à la politique. En solo : le `ticket` (format pensé pour l'agent seul).
    Sinon : les instructions du scénario utilisateur. La *policy* du domaine (longue) n'est PAS
    injectée ici pour ne pas gonfler chaque prompt K·H — piste de réglage ultérieure."""
    if task is None:
        return "Résoudre la tâche."
    ticket = getattr(task, "ticket", None)
    if ticket:
        return str(ticket)
    scenario = getattr(task, "user_scenario", None)
    return str(scenario) if scenario is not None else "Résoudre la tâche."


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
