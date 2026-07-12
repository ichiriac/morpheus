"""Boucle fermée MPC (implémentation directe de l'algorithme de specs/01).

    PROPOSER (Qwen) → LOOKAHEAD (world-model) → EXÉCUTER 1 pas (réalité)
    → DIVERGENCE → ROUTER LA SURPRISE → RÉ-ANCRER sur l'état vrai.

`use_world_model=False` court-circuite le lookahead et exécute la 1re action proposée :
c'est la baseline ReAct nue (Phase 0), à comparer contre la version world-model (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agents.policy import Policy
from ..agents.surprise import SurpriseRouter, divergence
from ..agents.world_model import WorldModel
from ..config import OrchestratorConfig
from ..envs.base import Env
from .types import Action, State, TraceStep


@dataclass
class EpisodeResult:
    success: bool
    turns: int
    total_reward: float
    trace: list[TraceStep] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        cfg: OrchestratorConfig,
        router: SurpriseRouter | None = None,
    ) -> None:
        self.policy = policy
        self.wm = world_model
        self.cfg = cfg
        self.router = router or SurpriseRouter()

    def run(self, env: Env) -> EpisodeResult:
        obs = env.reset()
        tools = env.tool_names()
        state = State(goal=env.goal(), observation=obs)
        total_reward = 0.0
        trace: list[TraceStep] = []

        for turn in range(1, self.cfg.max_turns + 1):
            state.turn = turn

            # 1. PROPOSER
            candidates = self.policy.propose(state, tools)

            # 2. LOOKAHEAD (MPC) — sinon baseline nue
            if self.cfg.use_world_model and len(candidates) > 1:
                scored = [
                    (self.wm.rollout(self.policy, state, c, tools, self.cfg.horizon), c)
                    for c in candidates
                ]
                best_score, chosen = max(scored, key=lambda t: t[0])
                predicted = self.wm.predict(state, chosen)
                score_before = self.wm.score_to_goal(state.goal, state.text)
            else:
                chosen = candidates[0]
                predicted = None
                best_score = 0.0
                score_before = 0.0

            # 3. EXÉCUTER un seul pas (réalité)
            step = env.step(chosen)
            total_reward += step.reward

            # 4. DIVERGENCE
            delta = divergence(predicted, step.observation.text) if predicted else 0.0

            # 5. ROUTER LA SURPRISE
            route = None
            if predicted is not None and delta > self.cfg.surprise_threshold:
                score_after = self.wm.score_to_goal(state.goal, step.observation.text)
                route = self.router.route(
                    delta=delta,
                    tool_error=step.observation.tool_error,
                    score_before=score_before,
                    score_after=score_after,
                )
                # Phase 3+ : ici, si route == ERROR/NOVELTY → RAG gated + replanification.
                # Phase 1 : on trace le routage sans agir dessus (instrumentation seule).

            trace.append(
                TraceStep(
                    turn=turn,
                    candidates=[str(c) for c in candidates],
                    chosen=str(chosen),
                    predicted_state=predicted,
                    real_state=step.observation.text,
                    divergence=delta,
                    surprise_route=route,
                    reward=step.reward,
                    done=step.done,
                )
            )

            # 6. RÉ-ANCRER sur l'état VRAI (jamais sur le prédit)
            state.observation = step.observation
            state.history.append(str(chosen))

            if step.done:
                return EpisodeResult(
                    success=step.info.get("success", step.reward > 0),
                    turns=turn,
                    total_reward=total_reward,
                    trace=trace,
                )

        return EpisodeResult(False, self.cfg.max_turns, total_reward, trace)
