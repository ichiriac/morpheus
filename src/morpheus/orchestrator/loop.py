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
from ..agents.memory import FactMemory
from ..agents.policy import Policy
from ..agents.surprise import (DIALOGUE_TOOL, Router, SurpriseRouter, SurpriseSignals,
                               familiarity)
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
        router: Router | None = None,
        kb: KnowledgeBase | None = None,
    ) -> None:
        self.policy = policy
        self.wm = world_model
        self.cfg = cfg
        self.router = router or SurpriseRouter()
        self.kb = kb  # référentiel de vérité, interrogé seulement sur surprise (RAG gated)
        # Le seuil de surprise appartient au WORLD-MODEL, pas à l'orchestrateur : δ change de
        # grandeur avec l'implémentation (Jaccard texte vs (1−cos)/2 latent), donc un seuil porté
        # ici se transmettrait d'un WM à l'autre en gardant une calibration qui n'est plus la
        # bonne. La config PRIME si elle en fixe un (échappatoire explicite) ; sinon on demande au
        # WM ; le repli 0.5 couvre les world-models tiers/doubles de test qui n'en déclarent pas.
        self.surprise_threshold: float = (
            cfg.surprise_threshold if cfg.surprise_threshold is not None
            else float(getattr(world_model, "surprise_threshold", 0.5))
        )

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
        # `max_turns` porté par l'état ⇒ rendu dans le prompt du PROPOSER (cf. policy.build_prompt).
        state = State(goal=env.goal(), observation=obs, max_turns=self.cfg.max_turns)
        # Manuel LÉGITIME de l'agent (policy du domaine τ²) : injecté au vrai PROPOSER seulement,
        # PAS dans les rollouts imaginés du world-model (prompts K·H bornés).
        sys_ctx = getattr(env, "system_context", lambda: None)()
        # Scratchpad ReAct : mémoire des couples (action → résultat réel). Sans lui, la politique
        # n'a que la DERNIÈRE observation et oublie les résultats d'outils passés (amnésie fatale
        # en tool-use multi-tours). Passé au vrai PROPOSER seulement, PAS aux rollouts imaginés.
        transcript: list[tuple[str, str]] = []
        if obs.text:
            transcript.append(("(ouverture)", obs.text))
        # Faits KB récupérés sur surprise au tour t → réinjectés au PROPOSER du tour t+1
        # (replanification ERROR / assimilation NOVELTY). La boucle re-ancre sur l'état vrai
        # entre-temps : c'est la replanification MPC, pas une exécution à l'aveugle.
        pending_facts: list[str] = []
        pending_route: str | None = None
        # Mémoire épisodique de faits atomiques (LWM-Planner) : accumule les faits des
        # observations réelles ; interrogée sur surprise en complément de la KB statique.
        # NON redondante avec le system_context (qui n'a que la policy, pas ce qui a été observé).
        memory = FactMemory()
        total_reward = 0.0
        trace: list[TraceStep] = []
        # Rubrique `loop_no_progress` (data/annotations) : répéter l'outil du pas précédent
        # (qui n'avait PAS erré) = signal d'agent qui tourne en rond.
        prev_tool: str | None = None
        prev_tool_error = False

        for turn in range(1, self.cfg.max_turns + 1):
            state.turn = turn

            # 1. PROPOSER (informé par la KB récupérée au tour précédent, s'il y a eu surprise)
            candidates = self.policy.propose(
                state, tools, system_context=sys_ctx, transcript=transcript,
                facts=pending_facts or None, route=pending_route,
            )
            pending_facts, pending_route = [], None   # consommés

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
                score_before = None   # pas de lookahead ⇒ proximité au but non sondée

            # 3. EXÉCUTER un seul pas (réalité)
            step = env.step(chosen)
            total_reward += step.reward

            # 4. DIVERGENCE (déléguée au world-model : texte pour LLM, cosinus latent pour JEPA)
            delta = self.wm.divergence(predicted, step.observation.text) if predicted else 0.0

            # 5. ROUTER LA SURPRISE  +  RAG *gated par la surprise* (Phase 3)
            route = None
            facts: list[str] = []
            signals: SurpriseSignals | None = None
            if predicted is not None and delta > self.surprise_threshold:
                score_after = self.wm.score_to_goal(state.goal, step.observation.text)
                # Récupération déclenchée UNIQUEMENT par la surprise (économie du gated) : on
                # interroge, avec l'état vrai + l'action qui a surpris, la KB statique (policy) ET
                # la mémoire épisodique (faits observés). Cette dernière sort du régime redondant.
                # Les scores de récupération sont capturés au passage : signal « cohérence RAG ».
                query = f"{chosen} {step.observation.text}"
                kb_top: float | None = None
                kb_hits: int | None = None
                if self.cfg.use_rag and self.kb is not None:
                    top = [(s, r) for s, r in self.kb.score(query) if s > 0.0][: self.cfg.rag_top_k]
                    kb_top = top[0][0] if top else 0.0
                    kb_hits = len(top)
                    facts += [r.as_fact() for _, r in top]
                mem_hits: int | None = None
                if self.cfg.use_memory:
                    mem_rules = memory.retrieve(query, self.cfg.memory_top_k)
                    mem_hits = len(mem_rules)
                    facts += [r.as_fact() for r in mem_rules]
                # Sonde « réductibilité » (opt-in : +1 appel LLM). Une prédiction latente (JEPA)
                # n'est pas verbalisable → sonde sans objet (reste None).
                reducibility: float | None = None
                if self.cfg.use_reducibility and isinstance(predicted, str):
                    probe = getattr(self.wm, "explain_gap", None)
                    if callable(probe):
                        reducibility = probe(predicted, step.observation.text, chosen)
                # Vecteur de signaux (tableau specs/01 + rubrique data/annotations) : journalisé
                # dans la trace (matière première du routeur appris) ET consommé par `route`.
                # Qui route dépend de la config (`router_checkpoint`) : règle Phase 1 par défaut,
                # routeur appris sinon — les deux lisent le MÊME vecteur, donc les runs restent
                # comparables à signaux identiques.
                signals = SurpriseSignals(
                    delta=delta,
                    tool_error=step.observation.tool_error,
                    score_before=score_before,
                    score_after=score_after,
                    kb_top_score=kb_top,
                    kb_hits=kb_hits,
                    memory_hits=mem_hits,
                    familiarity=familiarity(step.observation.text,
                                            (res for _, res in transcript)),
                    repeated_tool=(chosen.tool == prev_tool and chosen.tool != DIALOGUE_TOOL
                                   and not prev_tool_error),
                    is_user_turn=(chosen.tool == DIALOGUE_TOOL),
                    reducibility=reducibility,
                )
                route = self.router.route(signals)
                if facts:
                    # ARMÉS pour le PROPOSER du tour suivant : replanifier (ERROR) / assimiler (NOVELTY).
                    pending_facts, pending_route = facts, route

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
                    tool_error=step.observation.tool_error,
                    score_before=score_before,
                    signals=signals.to_dict() if signals is not None else None,
                )
            )

            # 6. RÉ-ANCRER sur l'état VRAI (jamais sur le prédit)
            state.observation = step.observation
            state.history.append(str(chosen))
            transcript.append((str(chosen), step.observation.text))
            prev_tool, prev_tool_error = chosen.tool, step.observation.tool_error
            if self.cfg.use_memory:   # mémorise les faits atomiques de l'observation réelle
                memory.observe(str(chosen), step.observation)

            if step.done:
                return EpisodeResult(
                    success=step.info.get("success", step.reward > 0),
                    turns=turn,
                    total_reward=total_reward,
                    trace=trace,
                )

        return EpisodeResult(False, self.cfg.max_turns, total_reward, trace)
