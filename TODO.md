# TODO — reprise de session (RunPod)

> Point de reprise pour continuer morpheus sur un pod RunPod (Linux + GPU).
> Contexte complet dans [`specs/`](specs/) ; commencer par [`specs/README.md`](specs/README.md).

## Quickstart RunPod

```bash
git clone https://github.com/ichiriac/morpheus.git && cd morpheus
bash scripts/runpod_setup.sh                 # venv + vllm + morpheus[openai,anthropic,dev]
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # ← choisir la quant selon la VRAM
```

**Choix du modèle selon le GPU attribué** (décision runtime figée = vLLM) :

| VRAM | Commande serveur (terminal 1, tmux) |
|---|---|
| **24 Go** (RTX 4090) — cible projet | `bash scripts/serve_qwen_vllm.sh` (défaut Qwen3-32B-AWQ, MAX_LEN 16k) |
| 24 Go, variante débit | `MODEL=Qwen/Qwen3-Coder-30B-A3B bash scripts/serve_qwen_vllm.sh` |
| 48 Go (A6000/L40S) | `MODEL=Qwen/Qwen3-32B-AWQ MAX_LEN=32768 bash scripts/serve_qwen_vllm.sh` |
| 80 Go (A100/H100) | `MODEL=Qwen/Qwen3-32B MAX_LEN=32768 bash scripts/serve_qwen_vllm.sh` (bf16, sans quant) |

Deux pistes **indépendantes** peuvent démarrer en parallèle : **A (brancher Qwen)** et
**B (smoke JEPA)**. Elles ne se bloquent pas.

## État actuel

- **Specs** `specs/00`→`05` (contexte, archi, bench, scaffold, RunPod+Qwen, entraînement JEPA).
- **Code testé sans GPU** : **16 tests verts + 2 skip torch**. Boucle fermée MPC + LLM-as-world-model,
  env mock, métrique réussite-vs-tours, connecteurs LLM (stub/vLLM/Anthropic), pipeline JEPA
  (data/encoders/model/losses/train), adaptateur τ²-bench (squelette `TODO(tau2)`).
- **Pas encore fait sur GPU** : run avec vrai Qwen, câblage τ²-bench, entraînement JEPA réel, wiring JepaWorldModel.

## Décisions figées (ne pas rediscuter)

- **Objectif** : agentique multi-tours (10+ tours) sur **1 GPU < 5000 € (~24-32 Go)**. Qwen
  (politique) + world-model latent JEPA (planificateur), boucle fermée + RAG *gated* par la
  surprise. Nœud scientifique = **routeur de surprise** (ERREUR vs NOUVEAUTÉ).
- **Bench** : **τ²-bench** (retail d'abord). Réf. supérieure = **API Sonnet 4.6**. Cible
  réaliste single-GPU = **approcher Sonnet 4.6** (pas Opus 4.8 / Fable 5 / Mythos 5).
- **Métrique qui tranche** : réussite **vs nombre de tours** ; la thèse veut voir la courbe
  world-model diverger de la baseline **à partir de 8+ tours** (pas le score agrégé).
- **Runtime** : **vLLM**, cible **RTX 4090 24 Go**, **Qwen3-32B-AWQ** (ou MoE Qwen3-Coder-30B-A3B).
- **Stack boucle** : **maison** (loop.py), pas LangGraph. **DSPy plus tard** (optim. prompt politique).
- **JEPA** : module annexe, encodeur **gelé** (pas H-JEPA v0), **orchestrateur-pilote** (pas Qwen-pilote).
  Ne PAS réentraîner le backbone de Qwen en JEPA.

---

## Piste A — brancher Qwen réel (étape 1)   → specs/04-runpod-qwen.md

```bash
# terminal 1 : serveur (voir table Quickstart selon la VRAM)
bash scripts/serve_qwen_vllm.sh
# terminal 2 :
morpheus check-llm --config configs/qwen_local.yaml            # doit finir par "OUI ✅"
```

- [ ] `check-llm` OK : la politique propose `authenticate_user` en format parsable.
      Si Qwen dévie → lire le bloc « SORTIE BRUTE », ajuster `enable_thinking`,
      `policy.temperature`, ou le prompt `_SYS` dans `src/morpheus/agents/policy.py`.
- [ ] Sanity pipeline sur le mock avec Qwen réel :
      ```bash
      morpheus run --config configs/qwen_local.yaml --no-world-model --out runs/qwen_baseline
      morpheus run --config configs/qwen_local.yaml --out runs/qwen_wm
      ```
      (Sur le mock, baseline ≈ world-model : normal, trop simple — la vraie mesure c'est τ²-bench.)

## Piste B — smoke + entraînement JEPA (Phase 2)   → specs/05-jepa-training.md

```bash
pip install -e ".[jepa]"                                       # sentence-transformers + datasets (torch déjà via vllm)
morpheus train-jepa --config configs/jepa.yaml                 # smoke synthetic+hashing : la perte pred doit baisser
pytest -q                                                      # doit passer les 2 tests torch (skip hors GPU)
```

- [ ] Smoke `train-jepa` converge (perte `pred` décroît).
- [ ] **Vérifier la normalisation d'un vrai dataset AVANT gros run** :
      `morpheus inspect-data --source hf:Salesforce/APIGen-MT-5k --limit 20`
      (si l'aperçu est faux → ajuster `from_messages` dans `src/morpheus/jepa/data.py`).
- [ ] Vrai run : éditer `configs/jepa.yaml` → `source: hf:Salesforce/APIGen-MT-5k`,
      `encoder: sentence_transformer`, `epochs: 50`.

---

## Étape 2 — câbler τ²-bench (retail)   *(après piste A)*

- [ ] Installer τ²-bench sur le pod (confirmer le nom du paquet / repo Sierra).
- [ ] Implémenter les `TODO(tau2)` dans `src/morpheus/envs/tau2_adapter.py` :
      `reset() / step() / goal() / tool_names()` + mapping `Action`↔outils, reward, `done`,
      et `tool_error` sur échec d'outil.
- [ ] Nouvelle config `configs/qwen_tau2.yaml` (`eval.env: tau2`).
- [ ] Vérifier le bucketing par longueur de tâche (`required_turns()` / `eval/metrics.py`).

## Étape 3 — mesures de référence (livrable Phase 1)

- [ ] Courbe **réussite-vs-tours** sur τ²-bench retail : `Qwen nu (--no-world-model)` vs
      `Qwen + LLM-as-world-model`.
- [ ] Ligne de référence **Sonnet 4.6** : `configs/reference_sonnet.yaml` (`kind: anthropic`,
      `ANTHROPIC_API_KEY`), même sous-ensemble de tâches.
- [ ] Consigner les 3 courbes.
- [ ] **Exporter les rollouts τ²-bench** (`to_jsonl`) → fine-tune JEPA dessus (`source: jsonl:<path>`).

## Étape 4 — wiring JEPA dans la boucle + Phases 3/4

- [ ] Écrire `JepaWorldModel` (même contrat que `agents/world_model.py`) branché sur `jepa.pt` :
      `predict` → `model.predict_next(...)`, `divergence` → `1 - cos(ŝ', E_state(obs))`,
      `score_to_goal` → distance latente à `E_state(goal)`. **loop.py ne change pas.**
- [ ] Re-mesurer la courbe réussite-vs-tours avec le JEPA latent.
- [ ] **Phase 3** : brancher le **RAG gated par la surprise** (aujourd'hui tracé sans agir,
      `loop.py` étape 5).
- [ ] **Phase 4** : routeur de surprise appris (5 signaux de specs/01) ; POC **DSPy** sur
      `agents/policy.py` (optimiseur avec la réussite-vs-tours comme métrique).

---

## Commandes utiles

```bash
pytest -q                                          # 16 tests + 2 skip torch (sans GPU)
morpheus run --config configs/phase1.yaml          # tout en stub+mock (sanity, sans GPU)
morpheus inspect-data --source synthetic           # aperçu des transitions JEPA
export PYTHONIOENCODING=utf-8                       # si la sortie console casse (Windows surtout)
```

## Décisions encore ouvertes

- Aucune bloquante. Seul ajustement : modèle/quant selon le GPU réel (voir table Quickstart).
