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
- **Pas encore fait sur GPU** : vrai run JEPA (sentence_transformer + APIGen) puis run τ²-bench
  avec `jepa_wm.enabled` (le `JepaWorldModel` est écrit et testé, reste à l'alimenter d'un vrai checkpoint).

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

### Mesure équitable retail — capacité dialogue FAITE (2026-07-13)

Découverte de câblage : **retail n'a aucun `ticket`** (0/114) → pas de solo ; il faut le
**user-sim** (non-solo). La politique de morpheus n'émettait que des appels d'outils.

- [x] **Action « répondre à l'utilisateur » ajoutée** (`respond_to_user`) : outil SYNTHÉTIQUE
      exposé par l'adaptateur en mode non-solo (`tool_names += respond_to_user`). `step()`
      l'intercepte et envoie le TEXTE (pas un appel d'outil) à `gym.step` → τ² le route vers le
      user-sim, dont la réponse devient l'observation suivante. **`loop.py` inchangé** (la capacité
      vit à la frontière env). Helpers testés offline (`_extract_text`). Hint ajouté à `policy._SYS`.
- [x] **User-sim câblé sur le vLLM Qwen** (litellm `openai/…` + `api_base`) ; `<think>` coupé côté
      user-sim (`extra_body.chat_template_kwargs.enable_thinking=false`).
- [x] **Vérifié end-to-end** : smoke retail non-solo → Qwen appelle `respond_to_user`, le user-sim
      répond (`user: …`, sans `<think>`), dialogue multi-tours réel. (0% sur 2-3 tâches K=2/H=1 :
      normal, ce n'est pas encore la vraie mesure — juste la preuve que le dialogue tourne.)
- [x] `policy._SYS` : placeholder `ARGS` remplacé par « vrais arguments » (le point telecom).
- [x] **FUITE D'INFO CORRIGÉE (2026-07-13)** : `_build_goal` renvoyait `str(user_scenario)` en
      non-solo. Or `UserScenario` = « All the information that will be sent to the user simulator »
      (persona, reason_for_call, **unknown_info** = ce que l'agent doit découvrir, task_instructions).
      Le vrai agent τ² ne voit QUE `domain_policy + agent_instruction`, jamais le scénario → l'injecter
      **gonflait** la mesure (le but contenait la réponse ; `score_to_goal` trivialement aligné) et la
      rendait non comparable au leaderboard. **Fix** : non-solo `goal()` = instruction GÉNÉRIQUE
      (aucun besoin utilisateur) ; le besoin émerge du dialogue (observations). Solo garde le `ticket`
      (brief légitime). Test de non-régression `test_nonsolo_goal_does_not_leak_user_scenario`.

### ⚠️ À TRANCHER avant la mesure de l'étape 3

Deux questions couplées, ouvertes par le fix ci-dessus :

1. **Policy du domaine** — le vrai agent τ² a la policy complète en système ; morpheus ne l'a plus
   (goal générique). Sans elle, morpheus est *sous-informé* (règles retail inconnues) → toujours pas
   comparable au leaderboard, biais inverse. Options : (a) **contexte système dédié** dans `policy.py`
   = policy du domaine (fidèle à τ², la garder HORS du prompt world-model pour borner le coût) ;
   (b) s'appuyer sur le **RAG existant** (`knowledge.py`, policy.md → BM25, gated surprise) ;
   (c) policy entière dans `goal()` (simple mais gonfle les prompts K·H). *Reco : (a).*
2. **Formation du but latent `g` en non-solo** — le world-model veut un `g` concret, mais en non-solo
   il n'existe aucun but légitime a priori : il doit **émerger du dialogue**. Options : (a) `g`
   statique générique (signal `score_to_goal` faible mais honnête) ; (b) `g` **dynamique** distillé de
   la conversation une fois le besoin révélé (met à jour `state.goal` en cours d'épisode → vrai signal
   MPC, fidèle à « le but émerge du dialogue » ; touche `loop.py`/l'orchestrateur). *Reco : (a)
   d'abord pour débloquer une mesure, (b) comme raffinement une fois JEPA branché.*

## Étape 3 — mesures de référence (livrable Phase 1)

> **Archivage des résultats FAIT** : chaque `morpheus run` écrit `<out_dir>/results.md` (courbe +
> métadonnées) et **ajoute une ligne à `BENCHMARKS.md`** (journal cumulatif versionné, racine repo).
> Cf. `eval/report.py`. Il suffit de lancer les runs ci-dessous, les courbes s'y consignent seules.

- [ ] Courbe **réussite-vs-tours** sur τ²-bench retail : `Qwen nu (--no-world-model)` vs
      `Qwen + LLM-as-world-model` (config `qwen_tau2.yaml`, éventuellement policy domaine injectée).
- [ ] Ligne de référence **Sonnet 4.6** : `configs/reference_sonnet.yaml` (`kind: anthropic`,
      `ANTHROPIC_API_KEY`), même sous-ensemble de tâches.
- [ ] Consigner les 3 courbes (auto-agrégées dans `BENCHMARKS.md`).
- [ ] **Exporter les rollouts τ²-bench** (`to_jsonl`) → fine-tune JEPA dessus (`source: jsonl:<path>`).

## Étape 4 — wiring JEPA dans la boucle + Phases 3/4

- [x] **JepaWorldModel FAIT (2026-07-13)** : `agents/jepa_world_model.py` charge `jepa.pt`
      (encodeur gelé + modèle reconstruits depuis le checkpoint), même contrat que
      `world_model.py` → **drop-in**. `predict`→`predict_next` (ŝ' latent), `score_to_goal`→cos
      latent (proj état, proj but), `divergence`→`(1-cos(ŝ', proj(E_state(obs))))/2`.
      **Intégration OPTIONNELLE** : `jepa_wm.enabled` (défaut `false`) — off ⇒ LLM WM, torch
      jamais importé, tests intacts. 3 tests (`test_jepa_wm.py`, skip sans torch), suite verte.
      ⚠️ **1 ligne changée dans loop.py** (`divergence()` → `self.wm.divergence()`, délégation à
      comportement identique pour le LLM WM) — nécessaire pour une divergence latente ; le TODO
      disait « loop.py ne change pas » mais c'était incompatible avec `divergence` côté WM.
      **Limite v0** : lookahead latent à **1 pas** (pas de décodage latent→texte pour proposer
      plus loin via Qwen) ; `horizon>1` ⇒ politique en espace latent, chantier ultérieur.
- [ ] **⛔ GATE étape 4 — valider le signal goal-relative de `score_to_goal`** (sinon c'est un
      PROXY, déjà documenté comme tel dans `jepa_world_model.py`). Harnais **écrit** :
      `scripts/validate_goal_signal.py` (H1 monotonie sur succès via Spearman t↑vs score↑ ;
      H2 pente succès > échec ; H3 `score_after<score_before` vs annotation ERREUR/NOUVEAUTÉ).
      Stats par permutation (numpy pur), vérifié sur jeux plantés (signal→PASS, nul→FAIL).
      **BLOQUÉ sur 3 pré-requis manquants** (état 2026-07-13) :
      1. **0 trajectoire τ² RÉSOLUE** : les 6 épisodes réels (`runs/qwen_tau2_*`) sont tous
         succès=0, échec sur le **bug ARGS placeholder** (`_SYS` : `{"clef":"valeur"}` → outils en
         erreur). ⇒ corriger `_SYS`/politique pour des args réels PUIS relancer un vrai run τ²
         (assez de résolus ET d'échoués pour H1/H2).
      2. **Pas de JEPA sémantique** : seul `jepa.pt` = hashing/synthetic (bruit + token-leak).
         ⇒ entraîner sur **sentence_transformer + APIGen** (⚠️ garder `transformers<5`, cf.
         Journal §2 — un mauvais install casse le tokenizer vLLM du serveur en cours).
      3. **`goal` désormais persisté** dans `episodes.jsonl` (fait : 1 ligne dans `runner.py`) —
         les 6 runs existants sont d'AVANT ce fix, donc sans goal → à re-générer.
      Lancer ensuite : `python scripts/validate_goal_signal.py --episodes runs/<run>/episodes.jsonl
      --checkpoint checkpoints/jepa/jepa.pt --labels <annotations_manuelles>.jsonl`.
      Si H1/H2 FAIL ⇒ entraîner `P(z,a,g)` conditionné OU terme d'alignement but↔état-terminal.
- [ ] Re-mesurer la courbe réussite-vs-tours avec le JEPA latent (après le gate ci-dessus).
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
