# TODO — reprise de session (RunPod)

> Point de reprise pour continuer morpheus sur un pod RunPod (Linux + GPU).
> Contexte complet dans [`specs/`](specs/) ; commencer par [`specs/README.md`](specs/README.md).

## Où on en est

- **Specs écrites** : contexte de l'expérience, architecture orchestrateur JEPA+Qwen,
  benchmark de référence (τ²-bench figé), scaffold, procédure RunPod. Voir `specs/00`→`04`.
- **Scaffold Phase 1 codé et testé** (10 smoke tests verts, sans GPU) : boucle fermée MPC
  + LLM-as-world-model, env mock multi-tours, métrique réussite-vs-tours, connecteurs
  LLM (stub / Qwen via vLLM / Anthropic), adaptateur τ²-bench (squelette).
- **Branchement Qwen prêt** : `configs/qwen_local.yaml`, `scripts/serve_qwen_vllm.sh`,
  `scripts/runpod_setup.sh`, commande `morpheus check-llm`.
- **Pas encore fait** : exécuter avec un VRAI Qwen (besoin GPU) ; câbler τ²-bench ; JEPA latent.

## Rappels clés (ne pas rediscuter)

- **Objectif** : agentique multi-tours (10+ tours) sur **1 GPU < 5000 € (~32 Go)**, Qwen
  (politique) + world-model latent JEPA (planificateur), boucle fermée + RAG *gated* par
  la surprise. Nœud scientifique = **routeur de surprise** (ERREUR vs NOUVEAUTÉ).
- **Bench figé** : **τ²-bench** (retail d'abord). Ligne de référence supérieure = **API Sonnet 4.6**.
  Cible réaliste single-GPU = **approcher Sonnet 4.6** (pas Opus 4.8 / Fable 5 / Mythos 5).
- **Métrique qui tranche** : réussite de tâche **vs nombre de tours** ; la thèse veut voir
  la courbe world-model diverger de la baseline **à partir de 8+ tours**.
- **Modèles single-GPU visés** : `Qwen/Qwen3-32B-AWQ` (~22 Go, RTX 4090 24G) ou
  `Qwen/Qwen3-Coder-30B-A3B` ; A100 80G en bf16 pour aller vite sans quant.
- Ne PAS réentraîner le backbone de Qwen en JEPA (mauvaise piste). JEPA = module annexe.

## Étape 1 — brancher Qwen réel (EN COURS)   → détail : specs/04-runpod-qwen.md

```bash
git clone https://github.com/ichiriac/morpheus.git && cd morpheus
bash scripts/runpod_setup.sh                                   # venv + vllm + morpheus
MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh       # terminal 1 (tmux)
source .venv/bin/activate                                      # terminal 2
morpheus check-llm --config configs/qwen_local.yaml            # doit finir par "OUI ✅"
```

- [ ] `check-llm` OK : la politique propose bien `authenticate_user` en format parsable.
      Si Qwen dévie → regarder le bloc « SORTIE BRUTE », ajuster `enable_thinking`,
      `policy.temperature`, ou le prompt `_SYS` dans `src/morpheus/agents/policy.py`.
- [ ] Sanity check pipeline sur le mock avec Qwen réel :
      ```bash
      morpheus run --config configs/qwen_local.yaml --no-world-model --out runs/qwen_baseline
      morpheus run --config configs/qwen_local.yaml --out runs/qwen_wm
      ```
      (Sur le mock, baseline ≈ world-model : c'est normal, le mock est trop simple pour
      départager — la vraie mesure c'est τ²-bench.)

## Étape 2 — câbler τ²-bench (retail)

- [ ] Installer τ²-bench sur le pod (confirmer le nom du paquet / repo Sierra).
- [ ] Implémenter les `TODO(tau2)` dans `src/morpheus/envs/tau2_adapter.py` :
      `reset() / step() / goal() / tool_names()` + mapping `Action`↔outils τ²-bench,
      reward de tâche, flag `done`, et `tool_error` sur échec d'outil.
- [ ] Basculer `eval.env: tau2` dans une nouvelle config `configs/qwen_tau2.yaml`.
- [ ] Vérifier que la métrique bucketise bien par longueur de tâche (sinon adapter
      `required_turns()` ou le bucketing dans `eval/metrics.py`).

## Étape 3 — mesures de référence

- [ ] Courbe **réussite-vs-tours** sur τ²-bench retail : `Qwen nu (--no-world-model)`
      vs `Qwen + LLM-as-world-model`.
- [ ] Ligne de référence **Sonnet 4.6** : créer `configs/reference_sonnet.yaml`
      (`kind: anthropic`, `ANTHROPIC_API_KEY`), même sous-ensemble de tâches.
- [ ] Consigner les courbes (les 3) — c'est le livrable de la Phase 1.

## Étape 4 — JEPA latent (Phase 2)   → détail : specs/05-jepa-training.md

**Pipeline d'entraînement DÉJÀ scaffoldé et testé** (`src/morpheus/jepa/`, 16 tests, 2 skip torch) :
prédicteur `P` + VICReg (`model.py`/`losses.py`), encodeur gelé (`encoders.py`), normalisation
des trajectoires HF→`(obs,action,next_obs)` (`data.py`), boucle (`train.py`), CLI
`morpheus train-jepa` + `morpheus inspect-data`.

- [ ] Sur le pod : `pip install -e ".[jepa]"` (torch + sentence-transformers + datasets).
- [ ] Smoke : `morpheus train-jepa --config configs/jepa.yaml` (synthetic+hashing, doit
      converger la perte de prédiction).
- [ ] **Vérifier la normalisation** d'un vrai dataset AVANT gros run :
      `morpheus inspect-data --source hf:Salesforce/APIGen-MT-5k --limit 20`
      (ajuster `from_messages` dans `jepa/data.py` si l'aperçu est faux).
- [ ] Vrai run : éditer `configs/jepa.yaml` → `source: hf:Salesforce/APIGen-MT-5k`,
      `encoder: sentence_transformer`, `epochs: 50`. Puis fine-tune sur rollouts τ²-bench
      (export `to_jsonl` → `source: jsonl:<path>`).
- [ ] **Wiring** : écrire `JepaWorldModel` (même contrat que `agents/world_model.py`) branché
      sur `jepa.pt` ; remplacer le proxy Jaccard de `surprise.py` par `1 - cos(ŝ', E_state(obs))`.
      L'orchestrateur (`loop.py`) ne change pas.
- [ ] Phase 3 : **RAG gated par la surprise** (aujourd'hui tracé sans agir, `loop.py` étape 5).
      Phase 4 : routeur de surprise appris.

## Commandes utiles

```bash
pytest -q                                          # 10 smoke tests (sans GPU)
morpheus run --config configs/phase1.yaml          # tout en stub+mock (sanity)
export PYTHONIOENCODING=utf-8                       # si sortie console casse (surtout Windows)
```

## Décisions encore ouvertes

- **Runtime Qwen** : vLLM (défaut) vs llama.cpp ; quantization AWQ/GPTQ sous 32 Go.
- **Stack de boucle** : rester maison/PyTorch, ou intégrer LangGraph/DSPy pour
  l'orchestration et réserver PyTorch au seul JEPA.
