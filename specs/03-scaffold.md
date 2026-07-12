# 03 — Scaffold Phase 1

> État du code au démarrage. La **Phase 1** (LLM-as-world-model, boucle fermée) est
> implémentée et **exécutable sans GPU ni clé API** (backend `stub` + env `mock`).

## Démarrage rapide

```bash
pip install -e .            # ou : pip install pyyaml pytest
export PYTHONIOENCODING=utf-8            # Windows : évite le crash d'encodage console

# smoke test complet (stub + mock), écrit runs/phase1/
morpheus run --config configs/phase1.yaml
# équivalent sans install :
PYTHONPATH=src python -m morpheus.cli run --config configs/phase1.yaml

# baseline ReAct nue (Phase 0) vs world-model (Phase 1)
python -m morpheus.cli run --config configs/phase1.yaml --no-world-model --out runs/baseline
python -m morpheus.cli run --config configs/phase1.yaml --out runs/phase1

pytest -q                   # 7 smoke tests
```

Sortie : `runs/<nom>/` contient `episodes.jsonl` (trace par tour), `summary.txt`
(courbe réussite-vs-tours) et `config.json`.

## Arborescence

```
src/morpheus/
├── config.py                 # dataclasses de config + chargement YAML
├── cli.py                    # `morpheus run …`
├── llm/                      # UNE interface, plusieurs backends
│   ├── base.py               #   LLMClient (Protocol), Message
│   ├── stub.py               #   déterministe, hors-ligne (CI / smoke)
│   ├── openai_compat.py      #   Qwen local via vLLM/llama.cpp (endpoint OpenAI)
│   └── anthropic_client.py   #   ligne de référence (API Sonnet 4.6)
├── orchestrator/
│   ├── types.py              #   Action, Observation, State, StepResult, TraceStep
│   └── loop.py               #   BOUCLE FERMÉE MPC (algo de specs/01)
├── agents/
│   ├── policy.py             #   Qwen : propose K actions candidates
│   ├── world_model.py        #   Phase 1 : LLM-as-world-model (predict + rollout + score)
│   └── surprise.py           #   δ (divergence) + routeur ERREUR/NOUVEAUTÉ
├── envs/
│   ├── base.py               #   interface Env (style Gym/τ²)
│   ├── mock_env.py           #   « retail-lite » multi-tours, longueur paramétrable
│   └── tau2_adapter.py       #   SQUELETTE τ²-bench (TODO(tau2) à câbler)
└── eval/
    ├── metrics.py            #   réussite vs nb de tours (LA métrique)
    └── runner.py             #   joue N tâches, écrit traces + résumé
```

## Correspondance code ↔ specs/01

| Étape de l'algo (specs/01) | Code |
|---|---|
| PROPOSER (Qwen, K actions) | `agents/policy.py::Policy.propose` |
| LOOKAHEAD (MPC, horizon H) | `agents/world_model.py::WorldModel.rollout` + `loop.py` |
| EXÉCUTER 1 pas (réalité) | `envs/*.step` appelé dans `loop.py` |
| DIVERGENCE δ | `agents/surprise.py::divergence` |
| ROUTER LA SURPRISE | `agents/surprise.py::SurpriseRouter.route` |
| RÉ-ANCRER sur l'état vrai | `loop.py` (`state.observation = step.observation`) |

## Ce qui est réel vs stubbé

**Réel et fonctionnel :**
- la boucle fermée MPC complète (propose → lookahead → exécute → divergence → route → ré-ancre) ;
- le calcul de la métrique réussite-vs-tours + traces JSONL ;
- l'env mock multi-tours avec injection de nouveauté et d'erreurs d'outil ;
- les connecteurs Qwen (OpenAI-compat) et Anthropic — prêts, testés à l'import près.

**Volontairement minimal (Phase 1), à faire évoluer :**
- `stub` : fausse politique/WM déterministe, sans vocation de performance (plomberie/CI). La courbe n'a de **sens qu'avec un vrai LLM** — avec le stub, baseline et world-model sont équivalents.
- `divergence` = Jaccard inversé de tokens (proxy) → **Phase 2** : `dist(ŝ', E_state(obs))` en latent JEPA.
- `WorldModel` = LLM-as-world-model → **Phase 2** : prédicteur JEPA `P(E_state(s), E_action(a))`.
- `SurpriseRouter` = règle à 2 signaux → **Phase 4** : classifieur appris sur les 5 signaux de specs/01.
- gating RAG **non branché** en Phase 1 : le routage est **tracé sans agir** (`loop.py`, étape 5). **Phase 3** : replanification + retrieval déclenchés par la surprise.
- `tau2_adapter.py` : squelette, `TODO(tau2)` à câbler une fois la version/API figée.

## Prochaines étapes (dans l'ordre)

1. **Brancher Qwen réel** : lancer vLLM (`Qwen3-32B`), passer `policy.kind/world_model.kind = openai`, refaire tourner mock pour valider le format de sortie de la politique.
2. **Câbler `tau2_adapter`** (retail) : reset/step/tools/goal → basculer `eval.env = tau2`.
3. **Mesurer la baseline** : courbe réussite-vs-tours Qwen nu (`--no-world-model`) vs Qwen + LLM-as-world-model. C'est la Phase 1 du plan de mesure.
4. **Ajouter la ligne de référence Sonnet 4.6** (`configs/reference_sonnet.yaml`, `kind: anthropic`).
5. Seulement ensuite : implémenter le JEPA latent (Phase 2).

## Décisions encore ouvertes (cf. specs/01)

- **Runtime Qwen** : vLLM vs llama.cpp + quantization (Q4/Q5/AWQ) sous 32 Go.
- **Stack de la boucle** : rester en PyTorch/maison, ou intégrer LangGraph/DSPy pour l'orchestration et réserver PyTorch au seul JEPA.
