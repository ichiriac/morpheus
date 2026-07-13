# TODO — reprise de session (RunPod)

> Point de reprise pour continuer morpheus sur un pod RunPod (Linux + GPU).
> Contexte complet dans [`specs/`](specs/) ; commencer par [`specs/README.md`](specs/README.md).

## Quickstart RunPod

```bash
git clone https://github.com/ichiriac/morpheus.git && cd morpheus
bash scripts/install_pinned.sh               # pile FIGÉE (vllm 0.10.2 + torch cu128 + transformers<5)
                                             # ⚠️ PAS runpod_setup.sh (vllm>=0.6.0 → torch cu130, casse : cf. Journal §1)
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # ← choisir la quant selon la VRAM
```

**Choix du modèle selon le GPU attribué** (décision runtime figée = vLLM) :

| VRAM | Commande serveur (terminal 1, tmux) |
|---|---|
| **48 Go (NVIDIA A40)** — GPU retenu | `bash scripts/serve_qwen_vllm.sh` (défaut : Qwen3-32B-AWQ, MAX_LEN 32768) |
| 48 Go, montée en qualité | `MODEL=Qwen/Qwen3-32B-GPTQ-Int8 bash scripts/serve_qwen_vllm.sh` (~34 Go, plus lent) |
| 24 Go (RTX 4090) | `MAX_LEN=16384 bash scripts/serve_qwen_vllm.sh` |
| 80 Go (A100/H100) | `MODEL=Qwen/Qwen3-32B bash scripts/serve_qwen_vllm.sh` (bf16, sans quant) |

> **A40 = Ampere (GA102), 48 Go** : pas de FP8 natif → rester en **AWQ/GPTQ entier**. bf16 32B
> (~64 Go) ne rentre pas dans 48 Go. Démarrer AWQ 4-bit (débit) ; passer GPTQ 8-bit si le jugement long est juste.

Deux pistes **indépendantes** peuvent démarrer en parallèle : **A (brancher Qwen)** et
**B (smoke JEPA)**. Elles ne se bloquent pas.

## État actuel

- **Specs** `specs/00`→`05` (contexte, archi, bench, scaffold, RunPod+Qwen, entraînement JEPA).
- **Code testé** : **29 tests verts** (les 2 tests torch tournent une fois torch installé). Boucle fermée MPC + LLM-as-world-model,
  env mock, métrique réussite-vs-tours, connecteurs LLM (stub/vLLM/Anthropic), pipeline JEPA
  (data/encoders/model/losses/train), adaptateur τ²-bench (squelette `TODO(tau2)`),
  **KB/RAG gated par la surprise** (`agents/knowledge.py`, policies τ² → BM25).
- **Piste A FAITE sur GPU (session 2026-07-12)** : Qwen3-32B-AWQ servi par vLLM sur A40,
  `check-llm` = OUI ✅, sanity baseline mock 5 tâches = 100 %.
- **Piste A étape 1 TERMINÉE (session 2026-07-13, après réinstall serveur)** : sanity world-model
  `qwen_mock_fast.yaml` = **100 % (3 tâches, bucket 4 tours, ~69 s)**. Chemin WM validé sans erreur.
  (Baseline ≈ WM sur le mock : attendu, trop simple — la vraie mesure c'est τ²-bench.)
- **Piste B avancée (2026-07-13)** : smoke `train-jepa` **converge** (val pred 0.254→0.062,
  `checkpoints/jepa/jepa.pt`) ; `inspect-data` sur `hf:Salesforce/APIGen-MT-5k` OK (44 transitions,
  normalisation `from_messages` correcte).
- **Pas encore fait sur GPU** : câblage τ²-bench, vrai run JEPA (sentence_transformer + APIGen), wiring JepaWorldModel.

## ⚠️ Journal d'environnement — NE PAS re-découvrir

> **MAJ 2026-07-13 (réinstall serveur)** : le pod A40 est reprovisionné avec un **driver CUDA 13.0**
> (`nvidia-smi` → 580.126.20), plus récent que le 12.8 d'origine. Un driver plus récent est
> **rétro-compatible** : la pile pinned cu128 ci-dessous tourne telle quelle (vérifié `torch.cuda`=True,
> vllm 0.10.2 sert Qwen sans souci). Le cache HF `/workspace/.hf-cache` a été perdu à la réinstall →
> re-download des poids (rapide ici, ~20 s de load). **Install : utiliser `scripts/install_pinned.sh`**
> (pile figée du journal), **PAS** `runpod_setup.sh` (`vllm>=0.6.0` casse, cf. §1).

Le pod A40 d'origine avait un **driver CUDA 12.8** (`nvidia-smi` → 570.211.01). Pièges rencontrés et résolus :

1. **`vllm>=0.6.0` installe vLLM 0.25 → torch cu130 (CUDA 13)** → crash `NVIDIA driver too old (12080)`.
   **Fix** : pile figée en `vllm==0.10.2` + `torch==2.8.0+cu128` (match exact driver 12.8) :
   ```bash
   pip install "vllm==0.10.2" --extra-index-url https://download.pytorch.org/whl/cu128
   ```
2. **`transformers 5.x` casse le tokenizer de vLLM 0.10.2** (`Qwen2Tokenizer has no attribute
   all_special_tokens_extended`). **Fix** : `pip install "transformers>=4.55.2,<5"` (→ 4.57.6).
   ⚠️ toute install qui tire `transformers` (ex. `[jepa]`/sentence-transformers) doit garder `<5`.
3. **AWQ lent (~3 tok/s !)** : le script forçait `--quantization awq` (kernel legacy). **Corrigé
   dans `scripts/serve_qwen_vllm.sh`** → `awq_marlin` (~27 tok/s, ×9). Idem GPTQ → `gptq_marlin`.
4. **Cache HF** : `HF_HOME=/workspace/.hf-cache` (volume PERSISTANT) est maintenant gravé dans le
   script. Sinon HF retombe sur l'overlay `/` éphémère et **re-télécharge 19 Go** à chaque restart.
   Le `.venv` et les poids (19 Go) sont sur `/workspace` → un restart ne re-télécharge RIEN.
5. **Redémarrer le serveur demain** (download-free, ~1 min de load + CUDA graphs) :
   ```bash
   cd /workspace/morpheus && source .venv/bin/activate
   MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh        # HF_HOME + marlin déjà gérés
   ```
6. **Coût du world-model** : ~35 appels LLM/tour (k=4 × horizon=3). ⇒ était **~3 min/tour** en
   série. **CORRIGÉ depuis** : `orchestrator.concurrency>1` lance les K rollouts en parallèle
   (vLLM batche) + `rollout` réutilise le ŝ' (un `predict`/tour en moins). Config rapide de
   sanity : `configs/qwen_mock_fast.yaml`. Flags : `--k --horizon --concurrency`. Le vrai run
   se fera sur τ²-bench (avec `concurrency: 8` dans `qwen_local.yaml`), pas sur le mock.

`scripts/serve_qwen_vllm.sh` (marlin + HF_HOME) est **déjà commité** (`006abf5`). Reste juste ce
`TODO.md` à committer.

## Décisions figées (ne pas rediscuter)

- **Objectif** : agentique multi-tours (10+ tours) sur **1 GPU < 5000 € (~24-32 Go)**. Qwen
  (politique) + world-model latent JEPA (planificateur), boucle fermée + RAG *gated* par la
  surprise. Nœud scientifique = **routeur de surprise** (ERREUR vs NOUVEAUTÉ).
- **Bench** : **τ²-bench** (retail d'abord). Réf. supérieure = **API Sonnet 4.6**. Cible
  réaliste single-GPU = **approcher Sonnet 4.6** (pas Opus 4.8 / Fable 5 / Mythos 5).
- **Métrique qui tranche** : réussite **vs nombre de tours** ; la thèse veut voir la courbe
  world-model diverger de la baseline **à partir de 8+ tours** (pas le score agrégé).
- **Runtime** : **vLLM**, GPU retenu **A40 48 Go**, **Qwen3-32B-AWQ** (kernels `awq_marlin`).
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

- [x] `check-llm` OK (2026-07-12) : format `ACTION/ARGS` parsable, pas de bloc `<think>`,
      `authenticate_user` bien proposé en 1er. ⚠️ ARGS = placeholders `{"clef":"valeur"}` —
      sans effet sur le mock, mais **à corriger dans `_SYS` avant τ²-bench** (args réels requis).
- [x] Sanity baseline mock (2026-07-12) : `--no-world-model --tasks 5` = **100 %** (~2 min).
      Boucle MPC end-to-end validée avec Qwen réel.
- [x] **FAIT (2026-07-13)** : sanity world-model via la config RAPIDE = **100 % (3 tâches, ~69 s)**.
      ```bash
      morpheus run --config configs/qwen_mock_fast.yaml --out runs/qwen_wm_fast   # K=2, H=1, 3 tâches, concurrency=4
      ```
      Depuis le §6, deux correctifs ont été ajoutés : les **K rollouts tournent en parallèle**
      (`orchestrator.concurrency>1` → vLLM batche, plus de série), et `rollout` réutilise le ŝ'
      (un `predict`/tour en moins). Coût/tour = `3 + K·(3H−1)` ⇒ K=2,H=1 ≈ 5 appels batché ⇒
      sanity en quelques secondes au lieu de ~1-2 h. Surcharges dispo : `--k --horizon --concurrency`.
      (Sur le mock, baseline ≈ WM : normal, trop simple — la vraie mesure c'est τ²-bench. Ne PAS
      y passer du temps : valider que ça tourne, puis attaquer l'étape 2.)

## Piste B — smoke + entraînement JEPA (Phase 2)   → specs/05-jepa-training.md

```bash
pip install -e ".[jepa]"                                       # sentence-transformers + datasets (torch déjà via vllm)
morpheus train-jepa --config configs/jepa.yaml                 # smoke synthetic+hashing : la perte pred doit baisser
pytest -q                                                      # doit passer les 2 tests torch (skip hors GPU)
```

- [x] Smoke `train-jepa` converge (2026-07-13) : val pred 0.254→0.062, `checkpoints/jepa/jepa.pt`.
- [x] **Normalisation d'un vrai dataset vérifiée (2026-07-13)** :
      `morpheus inspect-data --source hf:Salesforce/APIGen-MT-5k --limit 20` = 44 transitions,
      `from_messages` correct (obs=message / action=tool call / next_obs=résultat d'outil).
- [ ] Vrai run : éditer `configs/jepa.yaml` → `source: hf:Salesforce/APIGen-MT-5k`,
      `encoder: sentence_transformer`, `epochs: 50`.

---

## Étape 2 — câbler τ²-bench   *(FAIT 2026-07-13, sauf le point « mesure équitable retail »)*

- [x] **Installé** : paquet `tau2` (v1.0.0, repo Sierra `sierra-research/tau2-bench`, MIT, py3.12).
      Depuis source (pas sur PyPI) : `git clone … && pip install -e /workspace/tau2-bench` +
      `pip install gymnasium`. N'a PAS bousculé la pile vllm (dry-run vérifié). Clone persistant
      sur `/workspace/tau2-bench` → survit à un restart ; à re-`pip install -e` après réinstall venv.
- [x] **Adaptateur câblé sur l'interface gym de τ²** (`tau2.gym.AgentGymEnv`, step-based) :
      `reset/step/goal/tool_names/required_turns/close`. `Action`→JSON `{"name","arguments"}`,
      reward via `evaluate_simulation` (0 jusqu'à `done`), `success = reward≥1`, `done` = l'agent
      appelle l'outil `done` (auto-exposé). `required_turns` = nb d'actions **agent** de référence.
      Deux modes : **solo** (DummyUser, hors-ligne, tool-only, exige un `ticket`) et **non-solo**
      (user simulé par LLM). Testé : 3 tests (`tests/test_tau2.py`, skip si τ² absent) — **22 verts**.
- [x] Configs : `configs/qwen_tau2_telecom_solo.yaml` (smoke solo qui tourne tout de suite) et
      `configs/qwen_tau2.yaml` (cible retail, non-solo, user-sim sur le vLLM Qwen).
- [x] **Bucketing vérifié** : sur retail, `required_turns` couvre {5,6,7,8,9,10,11,12,13} (régime
      8+ tours bien peuplé) ; `SuccessVsTurns` agrège correctement (test dédié).
- [x] **Smoke end-to-end** : `morpheus run … qwen_tau2_telecom_solo` pilote τ² de bout en bout,
      Qwen émet de vrais appels d'outils τ² (args réels), reward calculé, threads nettoyés.

### ⚠️ Reste pour une MESURE ÉQUITABLE sur retail (bloc de l'étape 3)

Découverte de câblage : **retail n'a aucun `ticket`** (0/114) → pas de solo ; il faut le
**user-sim** (non-solo). Or la politique de morpheus **n'émet que des appels d'outils**, jamais de
message texte à l'utilisateur → les tâches retail qui exigent un dialogue (demander l'id de
commande, confirmer un remboursement…) sont pénalisées. Le pipeline tourne (`configs/qwen_tau2.yaml`)
mais la mesure n'est pas encore juste. Le seul domaine solo clé-en-main (telecom) a des tâches
trop courtes (0–2 actions) → courbe plate. **Décision à prendre pour l'étape 3** :

- [ ] **(recommandé)** Ajouter à l'orchestrateur une action « répondre à l'utilisateur » (message
      texte, en plus des appels d'outils) → débloque retail non-solo, fidèle à la thèse 8+ tours.
      Touche `policy.py` (émettre un message) + `loop.py`/`Action` (type message) + adaptateur
      (passer le texte à `gym.step`). ~sous-chantier dédié.
- [ ] Alternative de repli : mesurer d'abord en tool-only sur retail (biais assumé, borne basse)
      pour valider les courbes baseline-vs-WM avant d'investir la capacité dialogue.
- [ ] Corriger aussi le placeholder `ARGS` dans `policy._SYS` (args réels) — visible dès telecom.

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
- [x] **Phase 3 — KB v0 FAITE (2026-07-13)** : `agents/knowledge.py` charge les **policy.md
      de τ²** (chunk par règle atomique + retriever **BM25** sans dépendance) ; `loop.py` étape 5
      **récupère la KB *gated par la surprise*** (uniquement quand δ>seuil) et trace les règles
      dans `TraceStep.retrieved_facts`. Flags : `orchestrator.use_rag`, `rag_top_k` ;
      `eval.kb_policy_path` / `eval.tau2_data_dir`. Inspecter : `morpheus inspect-kb --domain
      retail --data-dir <τ²>/data --query "<état surprenant>"`. 7 tests verts (`test_knowledge.py`).
- [ ] **Phase 3 — reste** : agir sur `route` (réinjecter `retrieved_facts` dans la politique →
      `replanifie` si ERROR / `assimile` si NOVELTY). Aujourd'hui la KB est récupérée + tracée,
      pas encore consommée. Retriever dense (sentence-transformers) = optionnel, Phase 4.
- [ ] **Phase 4** : routeur de surprise appris (5 signaux de specs/01) ; POC **DSPy** sur
      `agents/policy.py` (optimiseur avec la réussite-vs-tours comme métrique).

---

## Commandes utiles

```bash
pytest -q                                          # 16 tests + 2 skip torch (sans GPU)
morpheus run --config configs/phase1.yaml          # tout en stub+mock (sanity, sans GPU)
morpheus inspect-data --source synthetic           # aperçu des transitions JEPA
morpheus inspect-kb --domain retail --data-dir <τ²>/data --query "cancel delivered order"  # KB (RAG)
export PYTHONIOENCODING=utf-8                       # si la sortie console casse (Windows surtout)
```

## Décisions encore ouvertes

- Aucune bloquante. Seul ajustement : modèle/quant selon le GPU réel (voir table Quickstart).
