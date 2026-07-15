"""Instrumentation du routeur de surprise (Phase 4) : features `SurpriseSignals`,
proxy `familiarity`, sonde `explain_gap`, et collecte/journalisation par la boucle."""

from __future__ import annotations

from morpheus.agents.policy import Policy
from morpheus.agents.surprise import SurpriseSignals, familiarity
from morpheus.agents.world_model import WorldModel
from morpheus.config import LLMConfig, OrchestratorConfig
from morpheus.envs.mock_env import make_mock_env
from morpheus.llm import build_llm
from morpheus.llm.base import Message
from morpheus.orchestrator.loop import Orchestrator
from morpheus.orchestrator.types import Action


# --------------------------------------------------------------------------- #
# Features (dataclass)
# --------------------------------------------------------------------------- #

def test_familiarity_bounds():
    assert familiarity("a b c", []) == 0.0                        # aucun passé
    assert familiarity("a b c", ["a b c"]) == 1.0                 # déjà-vu à l'identique
    assert familiarity("a b c", ["x y z"]) == 0.0                 # rupture totale
    assert 0.0 < familiarity("a b c", ["x y z", "a b q"]) < 1.0   # max sur le passé


def test_direction_none_without_scores():
    assert SurpriseSignals(delta=0.9, tool_error=False).direction is None
    s = SurpriseSignals(delta=0.9, tool_error=False, score_before=0.2, score_after=0.5)
    assert abs(s.direction - 0.3) < 1e-9


def test_signals_dict_and_vector_are_stable():
    s = SurpriseSignals(delta=0.7, tool_error=False, score_before=0.4, score_after=0.6)
    d = s.to_dict()
    assert d["delta"] == 0.7
    assert d["reducibility"] is None                  # non sondé = null journalisé tel quel
    v = s.as_vector()
    assert len(v) == len(SurpriseSignals.VECTOR_FIELDS)   # shape FIXE (classifieur Phase 4)
    assert all(isinstance(x, float) for x in v)
    # l'indicateur « sondé » distingue None de 0.0 (l'absence est une information)
    idx_probed = SurpriseSignals.VECTOR_FIELDS.index("reducibility_probed")
    assert v[idx_probed] == 0.0
    s2 = SurpriseSignals(delta=0.7, tool_error=False, reducibility=0.0)
    assert s2.as_vector()[idx_probed] == 1.0


# --------------------------------------------------------------------------- #
# Sonde « réductibilité » (WorldModel.explain_gap)
# --------------------------------------------------------------------------- #

class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, messages: list[Message], **kwargs) -> str:
        return self.reply


def test_explain_gap_parses_score_clamps_or_none():
    assert WorldModel(_FakeLLM("REDUCTIBLE: 8")).explain_gap("p", "r", Action("t")) == 0.8
    assert WorldModel(_FakeLLM("REDUCTIBLE: 42")).explain_gap("p", "r", "t") == 1.0  # clampé
    assert WorldModel(_FakeLLM("aucune idée")).explain_gap("p", "r", "t") is None    # imparsable


def test_stub_answers_explain_gap():
    wm = WorldModel(build_llm(LLMConfig(kind="stub")))
    # prédit == réel → écart nul → parfaitement réductible pour l'heuristique du stub
    assert wm.explain_gap("commande trouvée", "commande trouvée", Action("lookup_order")) == 1.0
    out = wm.explain_gap("commande trouvée", "erreur totale inconnue", Action("lookup_order"))
    assert out is not None and out < 0.5


# --------------------------------------------------------------------------- #
# Collecte par la boucle (gating + journalisation dans TraceStep)
# --------------------------------------------------------------------------- #

class _TwoCand(Policy):
    """2 candidats fixes (→ branche world-model active) — recette de test_memory.py : fixer
    les candidats rend ces tests de signaux indépendants de l'heuristique de la politique."""

    def propose(self, state, tools, system_context=None, transcript=None, facts=None, route=None):
        return [Action(tool=tools[0]), Action(tool=tools[min(1, len(tools) - 1)])]


def _orch(use_reducibility: bool = False) -> Orchestrator:
    llm = build_llm(LLMConfig(kind="stub"))
    cfg = OrchestratorConfig(k_candidates=2, horizon=1, max_turns=6, use_world_model=True,
                             surprise_threshold=-1.0,   # toute divergence ⇒ surprise (test)
                             use_reducibility=use_reducibility)
    return Orchestrator(_TwoCand(llm, k=2), WorldModel(llm), cfg)


def test_loop_records_signals_on_surprise_only():
    result = _orch().run(make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12]))
    assert any(s.surprise_route is not None for s in result.trace), "seuil -1 ⇒ surprises attendues"
    for s in result.trace:
        # signaux ⟺ surprise : la collecte suit exactement le gating du routeur
        assert (s.signals is not None) == (s.surprise_route is not None)
        if s.signals is not None:
            assert s.signals["delta"] == s.divergence
            assert s.signals["score_after"] is not None          # sondé sur surprise
            assert s.signals["score_before"] is not None         # sondé au lookahead
            assert 0.0 <= s.signals["familiarity"] <= 1.0
            assert isinstance(s.signals["repeated_tool"], bool)
            assert s.signals["kb_top_score"] is None             # use_rag off ⇒ non sondé
            assert s.signals["memory_hits"] is None              # use_memory off ⇒ non sondé
            assert s.signals["reducibility"] is None             # sonde off par défaut
        # les champs bon marché sont journalisés à CHAQUE tour
        assert isinstance(s.tool_error, bool)


def test_reducibility_probed_when_enabled():
    result = _orch(use_reducibility=True).run(
        make_mock_env(task_index=0, seed=0, buckets=[4, 8, 12])
    )
    probed = [s.signals["reducibility"] for s in result.trace if s.signals is not None]
    assert probed, "surprises attendues"
    assert all(p is not None and 0.0 <= p <= 1.0 for p in probed)  # le stub répond REDUCTIBLE: n
