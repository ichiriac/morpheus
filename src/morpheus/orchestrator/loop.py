"""Boucle fermée MPC (implémentation directe de l'algorithme de specs/01).

    PROPOSER (Qwen) → LOOKAHEAD (world-model) → EXÉCUTER 1 pas (réalité)
    → DIVERGENCE → ROUTER LA SURPRISE → RÉ-ANCRER sur l'état vrai.

`use_world_model=False` court-circuite le lookahead et exécute la 1re action proposée :
c'est la baseline ReAct nue (Phase 0), à comparer contre la version world-model (Phase 1).
"""

from __future__ import annotations

import concurrent.futures as _cf
from dataclasses import dataclass, field

from ..agents.knowledge import KnowledgeBase
from ..agents.policy import Policy
from ..agents.surprise import SurpriseRouter
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
        kb: KnowledgeBase | None = None,
    ) -> None:
        self.policy = policy
        self.wm = world_model
        self.cfg = cfg
        self.router = router or SurpriseRouter()
        self.kb = kb  # référentiel de vérité, interrogé seulement sur surprise (RAG gated)

    def _rollout_all(self, state: State, candidates: list[Action],
                     tools: list[str]) -> list[tuple[float, str]]:
        """Évalue les K candidats par rollout. `concurrency>1` lance les rollouts en
        parallèle (threads) : les appels LLM partent concurremment → vLLM les batche.
        `ThreadPoolExecutor.map` préserve l'ordre → départage des ex æquo identique au séquentiel."""
        H = self.cfg.horizon

        def one(c: Action) -> tuple[float, str]:
            return self.wm.rollout(self.policy, state, c, tools, H)

        if self.cfg.concurrency <= 1 or len(candidates) <= 1:
            return [one(c) for c in candidates]
        workers = min(self.cfg.concurrency, len(candidates))
        with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, candidates))

    def run(self, env: Env) -> EpisodeResult:
        obs = env.reset()
        tools = env.tool_names()
        state = State(goal=env.goal(), observation=obs)
        # Manuel LÉGITIME de l'agent (policy du domaine τ²) : injecté au vrai PROPOSER seulement,
        # PAS dans les rollouts imaginés du world-model (prompts K·H bornés).
        sys_ctx = getattr(env, "system_context", lambda: None)()
        # Scratchpad ReAct : mémoire des couples (action → résultat réel). Sans lui, la politique
        # n'a que la DERNIÈRE observation et oublie les résultats d'outils passés (amnésie fatale
        # en tool-use multi-tours). Passé au vrai PROPOSER seulement, PAS aux rollouts imaginés.
        transcript: list[tuple[str, str]] = []
        if obs.text:
            transcript.append(("(ouverture)", obs.text))
        total_reward = 0.0
        trace: list[TraceStep] = []

        for turn in range(1, self.cfg.max_turns + 1):
            state.turn = turn

            # 1. PROPOSER
            candidates = self.policy.propose(
                state, tools, system_context=sys_ctx, transcript=transcript
            )

            # 2. LOOKAHEAD (MPC) — sinon baseline nue
            if self.cfg.use_world_model and len(candidates) > 1:
                # K rollouts indépendants → parallélisables (vLLM batche les requêtes).
                results = self._rollout_all(state, candidates, tools)   # [(score, ŝ'_1er), ...]
                best_i = max(range(len(candidates)), key=lambda i: results[i][0])
                chosen = candidates[best_i]
                best_score, predicted = results[best_i]                  # ŝ' réutilisé (pas de re-predict)
                score_before = self.wm.score_to_goal(state.goal, state.text)
            else:
                chosen = candidates[0]
                predicted = None
                best_score = 0.0
                score_before = 0.0

            # 3. EXÉCUTER un seul pas (réalité)
            step = env.step(chosen)
            total_reward += step.reward

            # 4. DIVERGENCE (déléguée au world-model : texte pour LLM, cosinus latent pour JEPA)
            delta = self.wm.divergence(predicted, step.observation.text) if predicted else 0.0

            # 5. ROUTER LA SURPRISE  +  RAG *gated par la surprise* (Phase 3)
            route = None
            facts: list[str] = []
            if predicted is not None and delta > self.cfg.surprise_threshold:
                score_after = self.wm.score_to_goal(state.goal, step.observation.text)
                route = self.router.route(
                    delta=delta,
                    tool_error=step.observation.tool_error,
                    score_before=score_before,
                    score_after=score_after,
                )
                # Récupération déclenchée UNIQUEMENT par la surprise (économie du RAG gated) :
                # on interroge la KB avec l'état vrai + l'action qui a surpris.
                if self.cfg.use_rag and self.kb is not None:
                    query = f"{chosen} {step.observation.text}"
                    facts = [r.as_fact() for r in self.kb.retrieve(query, self.cfg.rag_top_k)]
                # Couture Phase 3+ : selon `route`, replanifier (ERROR) / assimiler (NOVELTY)
                # en réinjectant `facts` dans la politique. Ici on récupère + trace ; agir sur
                # `facts` (replanification) est l'incrément suivant.

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
                    retrieved_facts=facts,
                )
            )

            # 6. RÉ-ANCRER sur l'état VRAI (jamais sur le prédit)
            state.observation = step.observation
            state.history.append(str(chosen))
            transcript.append((str(chosen), step.observation.text))

            if step.done:
                return EpisodeResult(
                    success=step.info.get("success", step.reward > 0),
                    turns=turn,
                    total_reward=total_reward,
                    trace=trace,
                )

        return EpisodeResult(False, self.cfg.max_turns, total_reward, trace)
