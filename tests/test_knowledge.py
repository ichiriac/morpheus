"""KB (référentiel de vérité) : chunking des policies τ², retriever BM25, gating par surprise."""

from __future__ import annotations

import pytest

from morpheus.agents.knowledge import KnowledgeBase, chunk_policy, locate_policy
from morpheus.agents.policy import Policy
from morpheus.agents.surprise import SurpriseRouter
from morpheus.agents.world_model import WorldModel
from morpheus.config import LLMConfig, OrchestratorConfig
from morpheus.envs.mock_env import make_mock_env
from morpheus.llm import build_llm
from morpheus.orchestrator.loop import Orchestrator
from morpheus.orchestrator.types import Action

# Mini-policy au format τ² (titres + paragraphes = règles atomiques).
SAMPLE = """# Retail agent policy

At the beginning you must authenticate the user identity.

## Cancel pending order

An order can only be cancelled if its status is 'pending'.

The user must confirm the reason, either 'no longer needed' or 'ordered by mistake'.

## Modify pending order

### Modify items

This action can only be called once, and changes the status to 'pending (items modified)'.
"""


def test_chunker_splits_rules_and_tracks_sections():
    rules = chunk_policy(SAMPLE, domain="retail")
    # 1 intro + 2 cancel + 1 modify-items = 4 règles ; le H1 n'est pas une règle.
    assert len(rules) == 4
    assert rules[0].section == ""                              # intro (avant tout H2)
    assert "authenticate" in rules[0].text
    sections = [r.section for r in rules]
    assert "Cancel pending order" in sections
    # chemin de titres imbriqués conservé
    assert "Modify pending order > Modify items" in sections


def test_as_fact_is_compact_and_labelled():
    kb = KnowledgeBase.from_text(SAMPLE, domain="retail")
    fact = kb.retrieve("cancel order status pending", k=1)[0].as_fact()
    assert fact.startswith("[retail:Cancel pending order]")
    assert "\n" not in fact                                    # aplati


def test_retriever_surfaces_the_violated_rule():
    kb = KnowledgeBase.from_text(SAMPLE, domain="retail")
    # état surprenant : on a tenté d'annuler une commande livrée
    hits = kb.retrieve("cancel_order but the order status is delivered not pending", k=2)
    assert hits, "le retriever doit renvoyer au moins une règle"
    assert hits[0].section == "Cancel pending order"


def test_retriever_empty_query_returns_nothing():
    kb = KnowledgeBase.from_text(SAMPLE, domain="retail")
    assert kb.retrieve("", k=3) == []
    assert kb.retrieve("zzz qqq xyzzy", k=3) == []             # aucun token partagé


def test_locate_policy_explicit_and_missing(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text(SAMPLE, encoding="utf-8")
    assert locate_policy("retail", explicit=str(p)) == p
    with pytest.raises(FileNotFoundError):
        locate_policy("retail", explicit=str(tmp_path / "nope.md"))


def _orch(use_rag: bool, threshold: float, kb: KnowledgeBase | None) -> Orchestrator:
    llm = build_llm(LLMConfig(kind="stub"))
    cfg = OrchestratorConfig(
        k_candidates=4, horizon=2, max_turns=12, use_world_model=True,
        surprise_threshold=threshold, use_rag=use_rag, rag_top_k=3,
    )
    return Orchestrator(Policy(llm, k=4), WorldModel(llm), cfg, SurpriseRouter(), kb=kb)


def test_rag_is_gated_by_surprise():
    """Invariant : on ne récupère la KB QUE sur surprise (facts ⟹ route déclenché)."""
    kb = KnowledgeBase.from_text(SAMPLE, domain="retail")
    result = _orch(use_rag=True, threshold=0.0, kb=kb).run(  # seuil bas → surprise fréquente
        make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    )
    for step in result.trace:
        if step.retrieved_facts:
            assert step.surprise_route is not None             # jamais de RAG hors surprise


def test_rag_disabled_never_retrieves():
    kb = KnowledgeBase.from_text(SAMPLE, domain="retail")
    result = _orch(use_rag=False, threshold=0.0, kb=kb).run(
        make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    )
    assert all(not step.retrieved_facts for step in result.trace)


def test_policy_prompt_injects_kb_facts_by_route():
    """La KB récupérée entre dans le prompt PROPOSER, avec la consigne selon le routage."""
    from morpheus.orchestrator.types import Observation, State

    pol = Policy(build_llm(LLMConfig(kind="stub")), k=4)
    st = State(goal="g", observation=Observation(text="état"))
    facts = ["[retail:Cancel] An order can only be cancelled if pending."]
    err = pol.build_prompt(st, ["cancel_order"], facts=facts, route="ERROR")
    nov = pol.build_prompt(st, ["cancel_order"], facts=facts, route="NOVELTY")
    assert "CONNAISSANCE RÉCUPÉRÉE" in err and "can only be cancelled" in err
    assert "FAUTÉ" in err and "ASSIMILE" in nov               # consigne différente par route
    # sans faits : pas de bloc KB
    assert "CONNAISSANCE RÉCUPÉRÉE" not in pol.build_prompt(st, ["cancel_order"])


# KB dont le vocabulaire recouvre les observations du mock (noms d'outils underscore compris).
_KB_MOCK = """# Procédures retail (mock)

## Ticket
Pour un nouveau ticket, appeler authenticate_user puis lookup_order avant tout remboursement.

## Remboursement
Vérifier check_refund_policy et verify_payment_method avant issue_refund.
"""


class _TwoCandPolicy(Policy):
    """Renvoie 2 candidats fixes (→ la branche world-model tourne) et espionne les `facts`."""

    def __init__(self, llm, tools):
        super().__init__(llm, k=2)
        self._tools = tools
        self.facts_seen: list = []

    def propose(self, state, tools, system_context=None, transcript=None, facts=None, route=None):
        self.facts_seen.append(facts)
        return [Action(tool=tools[0]), Action(tool=tools[min(1, len(tools) - 1)])]


def test_rag_facts_reinjected_at_next_proposer():
    """Câblage : les faits récupérés sur surprise au tour t arment le PROPOSER du tour t+1
    (les rollouts imaginés, eux, ne reçoivent JAMAIS de faits)."""
    llm = build_llm(LLMConfig(kind="stub"))
    env = make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    kb = KnowledgeBase.from_text(_KB_MOCK, domain="retail")
    pol = _TwoCandPolicy(llm, env.tool_names())
    # seuil négatif → la surprise se déclenche dès que le world-model a prédit (predicted≠None).
    cfg = OrchestratorConfig(k_candidates=2, horizon=1, max_turns=6, use_world_model=True,
                             surprise_threshold=-1.0, use_rag=True, rag_top_k=3)
    result = Orchestrator(pol, WorldModel(llm), cfg, SurpriseRouter(), kb=kb).run(env)

    assert any(s.retrieved_facts for s in result.trace), "au moins une récupération attendue"
    assert any(f for f in pol.facts_seen if f), "les faits doivent être réinjectés au PROPOSER"
    # invariant : un PROPOSER n'a des faits que si un tour PRÉCÉDENT a récupéré (jamais au tour 1).
    assert pol.facts_seen[0] is None
