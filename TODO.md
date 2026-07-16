# TODO — reprise de session (RunPod)

> Point de reprise pour continuer morpheus sur un pod RunPod (Linux + GPU).
> Contexte complet dans [`specs/`](specs/) ; commencer par [`specs/README.md`](specs/README.md).

## Quickstart RunPod

```bash
git clone https://github.com/ichiriac/morpheus.git && cd morpheus
bash scripts/install_pinned.sh               # pile FIGÉE (vllm 0.10.2 + torch cu128 + transformers<5)
                                             # venv → /root/.venv-morpheus : HORS QUOTA (Journal §7).
                                             # runpod_setup.sh y délègue désormais.
source /root/.venv-morpheus/bin/activate
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
- **Code testé** : **48 tests verts** (2026-07-13). Boucle fermée MPC + LLM-as-world-model,
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
## 🔬 Bilan session 2026-07-13 (étape 3 + Phase 2) — LIRE AVANT DE REPRENDRE

Longue session de mesure. Beaucoup d'infra + des **résultats négatifs solides** qui cadrent la suite.

**A. Harnais τ²-bench rendu ÉQUITABLE (5 correctifs, tous committés + testés)** — trouvés en mesurant :
1. Schémas d'outils exposés à la politique (`_tool_signatures`, vrais noms d'args) — sinon Qwen
   invente les args → tout en erreur.
2. Fuite de scénario supprimée (`goal()` non-solo = générique, plus `user_scenario`).
3. Policy du domaine injectée en contexte système (au vrai PROPOSER, hors world-model).
4. Mémoire multi-tours (scratchpad action→résultat) — sinon amnésie des résultats d'outils.
5. Comptage du reward `close()` (état DB à la fin, même sur cap de tours).

**B. Phase 1 — LLM-as-world-model = AUCUN GAIN (attendu, cf. specs « baseline à battre »).**
   Politique ET WM = le même Qwen ⇒ le lookahead ≈ Qwen qui raisonne, ne bat pas le glouton.
   Mock planning (obs ne révèle plus l'étape) : **baseline 100 % à 4/8/12** (Qwen lit le plan du goal),
   WM **moins efficace** (10 tours pour une tâche de 4). Retail : **baseline ET WM ≈ 0 %**.

**C. Retail = trop dur pour Qwen-32B nu : baseline 0/8** (harnais équitable, comptage close vérifié
   `via_close=False`). L'agent atteint la phase d'action mais ses mutations ne produisent pas l'état
   DB cible. ⇒ pas de trajectoires τ² réussies disponibles.

**D. Phase 2 — JEPA entraîné + câblé, MAIS signal goal DÉGÉNÉRÉ (le vrai nœud).**
   - `configs/jepa_apigen.yaml` : JEPA réel sur APIGen-MT-5k (21106 transitions, 40 ép., val pred
     0.016). Checkpoint `checkpoints/jepa_apigen/jepa.pt` (compat `JepaWorldModel`).
   - `JepaWorldModel` fonctionnel en drop-in (`configs/jepa_wm_smoke.yaml`) : predict 256d,
     score/divergence ∈ [0,1], rollout OK, `loop.py` inchangé.
   - **PROBLÈME** : `score_to_goal` **ne discrimine rien** — étendue **0.0086** sur textes très
     contrastés (état-but 0.993 … charabia 0.984). Espace latent NON goal-relative (embeddings
     sentence-transformer anisotropes + `proj` entraîné seulement sur la perte de prédiction).
     ⇒ sélection MPC au niveau du bruit. C'est ce que `scripts/validate_goal_signal.py` (H1/H2)
     prédit en échec.

### ➡️ Reprise Phase 2 — deux chantiers couplés (décision : consolidé, à attaquer à froid)

- [x] **FIX DU SIGNAL GOAL — mécanisme IMPLÉMENTÉ + validé en-distribution (2026-07-13)** :
      terme d'**alignement but↔état** ajouté à la perte (choix : alignement, PAS `P(z,a,g)` — car
      `score_to_goal = cos(proj(état), proj(but))` n'utilise PAS le prédicteur ; c'est `proj` qu'il
      faut rendre goal-relative). `jepa/losses.py::goal_alignment_loss` = **régression**
      `cos(proj(s), proj(g)) ≈ 2·progress−1` (monotonie + étendue franche) + **InfoNCE** état→but
      (discrimination inter-buts, pondérée par progress). Données : `Transition` porte désormais
      `goal`/`progress`/`traj_id` (`data.py`) ; but = requête user (APIGen), progress = position
      normalisée. Câblé dans `train.py` (sélection du checkpoint sur `g_align`). Réentraîné :
      `checkpoints/jepa_apigen_goal/jepa.pt` (config `jepa_apigen_goal.yaml`, ~4 min GPU).
      - **En-distribution (sanity synthétique `scripts/check_goal_discrimination.py`)** : étendue
        **0.20** (checkpoint MiniLM) — voire **0.64** (checkpoint hashing dédié — vs **0.0086**
        dégénéré) ; monotonie Spearman **+0.96** ; discrimination terminal own−autres **+0.12**. ✅
      - **Probe direct τ² (`scripts/probe_tau2_goal_range.py`)** : étendue intra-épisode **médiane
        0.176** (vs 0.0086). Le signal n'est PLUS dégénéré.
      - **Gate officiel `validate_goal_signal.py` sur `data/tau2_replay/retail.jsonl`** (112 succès /
        57 échecs) : **H2 PASS** (p=0.0003 — pente succès −0.019 > pente échec −0.034 ; AVANT le fix
        H2 FAILait, pentes ≈ 0.0001) **MAIS magnitude faible et séparation de NIVEAU ≈ 0**.
        **H1 FAIL** (mean_rho **−0.46**, INVERSÉ). ⇒ verdict gate toujours ❌.
      - **DIAGNOSTIC du non-transfert τ² (le vrai constat)** : mismatch de distribution
        APIGen→τ²-retail. (1) Le `goal` retail est une **instruction générique en prose FR**, PAS un
        état-cible/requête d'issue → aucun chemin sémantique monotone vers la résolution. (2) Les
        `states` retail sont des **blobs JSON bruts** d'outils (`{"order_id":...}`, IDs nus), pas des
        observations NL comme à l'entraînement APIGen. Le latent est goal-relative LÀ OÙ il a été
        entraîné, pas encore sur τ²-retail.
- [x] **SUITE DU FIX (transfert τ²) — H1 RÉPARÉ sur held-out (2026-07-13)** : alignement entraîné
      EN-DOMAINE sur les transitions τ² rejouées (states JSON + but générique retail), split
      **held-out par trajectoire** (`scripts/build_tau2_alignment_data.py` : 101 traj train / 33 val
      held-out ; anti-leak). But unique ⇒ `w_goal_nce=0` (InfoNCE dégénéré) : on apprend un AXE DE
      PROGRESSION conditionné sur le but. Config `jepa_tau2_align.yaml`, checkpoint
      `checkpoints/jepa_tau2_align/jepa.pt`. **Gate held-out** (`retail_align_val.jsonl`, 21 succ /
      12 échecs) :
      - **H1 PASS** : mean_rho **+0.567** (médiane 0.7, frac>0 0.857, **p=0.0001**) — INVERSION
        CORRIGÉE (avant : rho −0.46 avec le checkpoint APIGen). Latent τ² monotone goal-relative
        sur des trajectoires JAMAIS VUES.
      - **H2 FAIL mais MARGINAL** : pente succès 0.059 > pente échec 0.027, **p=0.064** (juste
        au-dessus de α=0.05, direction CORRECTE). Limité par 12 négatifs seulement, tous « dernière
        action tronquée » → ils progressent presque autant que les succès.
      ⇒ Verdict script encore ❌ (H2 < 0.05) mais **H1 (la monotonie, le nœud) est réparé sur held-out**.
- [x] **H2 < 0.05 ATTEINT + JEPA-WM OPÉRATIONNEL (2026-07-13)** — gate held-out **✅ H1+H2 PASS** :
      1. **Négatifs variés** : `replay_reference_trajectories.py --neg-fracs 0.4,0.65,0.9` → troncatures
         à plusieurs points (échecs précoces ET tardifs) → **112 pos / 153 nég** (avant 57).
      2. **H2 rendu length-robust** (correctif de MÉTHODE, pas p-hacking) : la pente OLS est confondue
         par la longueur (un négatif court comprime la montée → pente/pas plus raide qu'un long succès
         qui plafonne ⇒ pente_échec > pente_succès à tort). `validate_goal_signal.py::h2_separation`
         teste désormais le **NIVEAU moyen** de `score_to_goal` (length-robust ET c'est ce que la
         sélection MPC exploite ; la pente reste en diagnostic).
      3. **Prédicteur à VRAIES actions** : le replay persiste `actions`, `build_tau2_alignment_data.py`
         construit (états[k−1] --action[k]--> états[k]) → P(z,a) apprend une vraie dynamique (avant :
         actions factices `step_k` → rollout/divergence creux). Alignement `w_goal=3.0`, epochs 100.
      **Gate held-out** (57 traj, 17 succ / 40 échecs) : **H1 PASS** (rho +0.634, p≈0) · **H2 PASS**
      (niveau succès 0.637 > échec 0.545, **p≈0**) → verdict **✅ score_to_goal n'est plus un proxy**.
      **JEPA-WM opérationnel dans le loop** (`configs/jepa_wm_tau2.yaml`) : predict→latent 256d,
      divergence discrimine la vraie action vs une mauvaise (0.13–0.24 vs 0.27–0.44, 3/4 pas), chemin
      MPC exercé avec >1 candidats (test `test_jepa_wm_lookahead_fires_with_multiple_candidates`).
      **⚠️ Caveat honnête** : le classement `rollout` à 1 pas (proximité au but de l'état PRÉDIT) reste
      bruité sous but générique — c'est une EXTRAPOLATION au-delà de ce que H1/H2 valident (états
      RÉELS). Pour un MPC tranchant : but d'issue par tâche + horizon>1 (chantier ultérieur).
- [ ] Ensuite : mesure baseline vs JEPA-WM sur un régime discriminant (avec la politique Qwen, qui
      produit de vrais candidats multiples ; relancer le serveur vLLM).
- [x] **DÉBLOCAGE DONNÉES FAIT (2026-07-13)** : `scripts/replay_reference_trajectories.py` rejoue
      les `evaluation_criteria.actions` de référence contre l'env τ² → **112 positifs retail**
      (db_reward=1.0 vérifié par l'évaluateur τ² officiel ; 77 exploitables ≥3 états) + **57 négatifs**
      (rejeu tronqué de la dernière action, db_reward<1). Aucune dépendance au serveur LLM (arrêté).
      `goal` = instruction générique non-solo (pas de fuite `user_scenario`). Sortie :
      `data/tau2_replay/retail.jsonl`. **`validate_goal_signal.py` a TOURNÉ sur du VRAI** :
      **H1 PASS** (mean_rho 0.121, p=0.029 — ordre de rang faiblement correct sur les succès),
      **H2 FAIL** (pente succès ≈ pente échec ≈ 0.0001, p=1.0 — magnitude négligeable, aucune
      séparation), H3 N/A (0 annotation manuelle). **Verdict gate = ❌** : espace NON goal-relative
      **confirmé empiriquement** (plus une prédiction). ⇒ le vrai chantier reste le FIX DU SIGNAL
      GOAL ci-dessus (P(z,a,g) conditionné OU terme d'alignement but↔état-terminal).
- [ ] Ensuite seulement : mesure baseline vs JEPA-WM sur un régime discriminant.

- [ ] **EN COURS (2026-07-13, committé) — juge NL-assertions retail câblé sur le vLLM local** :
      sans lui le reward retail = 0 (défaut τ² = `gpt-4.1` → 404 ; 112/114 tâches retail ont
      `NL_ASSERTION` dans `reward_basis` ⇒ reward = db × nl = 0). `tau2_adapter._wire_nl_judge`
      patche `DEFAULT_LLM_NL_ASSERTIONS` (au niveau du module evaluator, pas `tau2.config`),
      coupe le mode *thinking* et force `response_format=json_object` (sinon `json.loads` casse).
      Config : `tau2_judge_{llm,base_url,api_key_env}` (`configs/qwen_tau2.yaml` + `_jepawm.yaml`).
      Aussi : `reward_breakdown` (DB vs NL) exposé par l'env et journalisé par le runner ; rejeu
      `--solo` (telecom) pour un scoring DB fiable. ⚠️ **Qwen-juge-Qwen = mesure indicative, pas
      une référence** — cf. mémoire [[tau2-retail-scoring-needs-llm-judge]]. **Non encore mesuré
      end-to-end** (serveur vLLM à relancer).
- [ ] **Procédure d'install consolidée** : voir [`INSTALL.md`](INSTALL.md) (from scratch). Le
      checkpoint validé `checkpoints/jepa_tau2_align/jepa.pt` (H1+H2 PASS) **+** `data/tau2_replay/`
      sont désormais **versionnés** (exceptions `.gitignore`) → survivent à la perte du serveur.

### ⏭️ REPRISE session suivante (2026-07-13 soir) — état exact au moment de l'arrêt

- **JEPA-WM retail = validé + opérationnel + versionné.** Gate held-out ✅ (H1 rho +0.634,
  H2 niveau 0.637>0.545, p≈0). Checkpoint + data committés (survivent à une nouvelle machine).
- **⛔ TELECOM = IMPASSE (constat de cette session, corrige la note « `--solo` scoring DB fiable »
  ci-dessus qui est FAUSSE)** : telecom n'est PAS scoré en DB mais en **`ENV_ASSERTION`**, et ses
  `evaluation_criteria.actions` sont **entrelacées user/assistant** (actions device de l'utilisateur
  + actions agent) dépendantes de l'état device initial. Le rejeu depuis un env frais **échoue les
  assertions** (`enable_roaming`→« enabled » vs attendu « already enabled ») → `--solo` produit
  **0 positif/12**, `--no-solo` 2/12 inexploitables. De plus les tâches telecom-solo font **0–2
  actions agent** → trop courtes pour H1/H2 (≥3 états) ET courbe vs-tours plate. ⇒ **ne PAS
  ré-essayer le rejeu telecom** ; le « telecom-solo fiable » vaut pour les runs LIVE, pas le rejeu.
- **➡️ PROCHAINE ACTION = mesure baseline vs JEPA-WM sur RETAIL** (long-horizon, juge câblé, dialogue
  `respond_to_user`/`done` OK). Config prête : [`configs/qwen_tau2_jepawm.yaml`](configs/qwen_tau2_jepawm.yaml)
  (JEPA-WM validé + `tau2_judge_llm` local). Procédure :
  1. relancer le serveur vLLM (`bash scripts/serve_qwen_vllm.sh`) ;
  2. `morpheus run --config configs/qwen_tau2_jepawm.yaml --no-world-model --out runs/tau2_retail_baseline`
  3. `morpheus run --config configs/qwen_tau2_jepawm.yaml               --out runs/tau2_retail_jepawm`
  4. comparer réussite-vs-tours (buckets 4/8/12). Le run baseline a été LANCÉ puis STOPPÉ (fin de
     session) → aucun résultat exploitable encore.
  **⚠️ Caveats à garder en tête** : (a) rollout JEPA 1-pas bruité sous but générique (le classement
  des candidats peut ne PAS battre le glouton — on mesure pour savoir) ; (b) Qwen-juge-Qwen = mesure
  indicative (cf. [[tau2-retail-scoring-needs-llm-judge]]).

**État env en fin de session** : le **serveur vLLM est ARRÊTÉ** (je l'ai stoppé pour libérer la VRAM
et entraîner JEPA sur GPU). Le relancer pour toute mesure LLM (cf. Journal §5). Le checkpoint JEPA et
le clone `tau2-bench` sont sur `/workspace` (persistants).

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
   Les poids (19 Go) sont sur `/workspace` → un restart ne re-télécharge RIEN.
   ⚠️ **Le `.venv`, LUI, N'EST PAS sur `/workspace`** — il ne peut pas y être (§7). Cette ligne a
   longtemps affirmé le contraire ; c'était faux, et ça a fait chercher une panne de restart là
   où il y avait une panne de quota.
5. **Redémarrer le serveur** (download-free, ~1 min de load + CUDA graphs) :
   ```bash
   cd /workspace/morpheus && source /root/.venv-morpheus/bin/activate   # venv HORS quota (§7)
   MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh        # HF_HOME + marlin déjà gérés
   ```
   Si le conteneur a redémarré, le venv a disparu (il est sur l'éphémère) : `bash
   scripts/install_pinned.sh` d'abord (~10 min). Les 19 Go, eux, sont toujours là.
6. **Coût du world-model** : ~35 appels LLM/tour (k=4 × horizon=3). ⇒ était **~3 min/tour** en
   série. **CORRIGÉ depuis** : `orchestrator.concurrency>1` lance les K rollouts en parallèle
   (vLLM batche) + `rollout` réutilise le ŝ' (un `predict`/tour en moins). Config rapide de
   sanity : `configs/qwen_mock_fast.yaml`. Flags : `--k --horizon --concurrency`. Le vrai run
   se fera sur τ²-bench (avec `concurrency: 8` dans `qwen_local.yaml`), pas sur le mock.
7. **QUOTA `/workspace` — plafond MESURÉ le 2026-07-16 (après relèvement) :
   `36119 Mio ≈ 35,3 Gio`.** Quota provisionné = **38 Go décimaux** (36240 Mio nominal ; mesuré
   à −0,33 %). *Historique* : `31376 Mio ≈ 30,6 Gio` (≈33 Go) avant le relèvement.
   **Méthode** : chunks `dd` jusqu'à `Disk quota exceeded`. Deux méthodes indépendantes
   concordent à **3 Mio près** (base+écrit = 36116 ; `du` au refus = 36119), sur deux
   granularités (128/8 Mio et 256/16 Mio).
   ⚠️ **Le relèvement avait été ANNONCÉ à « 35 Go » — c'était faux** (+8,2 % d'écart au mesuré ;
   38 Go tombe à −0,33 %, exactement la signature de la mesure précédente). Troisième fois qu'un
   chiffre de quota circule sans mesure. **Ne JAMAIS citer ce nombre de mémoire ni sur parole :
   re-mesurer.** Le « 10 Go » d'avant avait traîné dans 2 fichiers sans avoir jamais été mesuré,
   et il était faux (le seul `.hf-cache` en pèse 19).
   **Mesurer SANS tuer un run en cours** : `runner.py` fait `f.write()`+`f.flush()` par épisode,
   sans capture — un `Errno 122` pendant un flush tue le run. Sonder dans la fenêtre juste APRÈS
   une écriture d'épisode (~2,2 min de marge ; la sonde prend ~20 s), et `rm -rf` la sonde
   immédiatement (`trap cleanup EXIT`).
   **`df` et `statvfs` sont INUTILISABLES ici** : ils annoncent 756 Tio, la taille du cluster
   MooseFS, pas la part du pod. Pas de binaire `mfs*`, pas de `getfattr`, `fallocate` non
   supporté. La mesure empirique est la seule.
   **Conséquence — inchangée malgré le relèvement** : le venv (**9,4 Go**) NE RENTRE TOUJOURS
   PAS. Occupation ~29,9/35,3 Gio (30612/36119 Mio) ⇒ **~5,4 Gio libres**. `python3 -m venv .venv` depuis le dépôt
   échoue en `Errno 122` **à mi-install, après plusieurs minutes de download**. Les deux scripts
   d'install avaient ce bug.
   **Piste si on veut un venv PERSISTANT** : `/workspace/.pip-cache` (6,8 Go) est du poids mort —
   il ne contient que la pile NON figée (vllm 0.25 / torch cu130) et rien ne le lit (le cache
   utile est sur `/root`). Le purger rendrait ~12,2 Gio ⇒ le venv y tiendrait. MAIS `/workspace`
   est un FS **réseau** : imports Python lents (vllm/torch pèsent). À mesurer avant de décider.
   Échappatoire prévue : `ALLOW_WORKSPACE_VENV=1`.
   ⇒ **Layout** : `/workspace` (persistant, sous quota) = dépôt + données + poids HF + tau2-bench.
   `/root` (overlay 50 Go, éphémère, sans quota) = venv + cache pip. `VENV_DIR` surchargeable,
   défaut `/root/.venv-morpheus` ; `install_pinned.sh` REFUSE un `VENV_DIR` sous `/workspace`.
   ⚠️ `PIP_CACHE_DIR` aussi doit rester hors quota : une seule install ratée y a gonflé
   `/workspace/.pip-cache` de 1,4 à 6,8 Go.
8. **`set -o pipefail` dans tout script d'install.** Sans lui, `bash install.sh | tee log` renvoie
   le code de `tee` : un `Disk quota exceeded` en pleine install ressort en **exit 0**. C'est
   comme ça que le bug de quota est passé pour un succès. Cousin shell du 404 silencieux.
   Corollaire : **ne jamais conclure d'un code retour** — exercer la pile (`import vllm, torch`,
   `torch.cuda.is_available()`), ce que fait maintenant la vérif finale d'`install_pinned.sh`.
9. **`[jepa]` n'est pas optionnel pour le banc.** `eval/runner.py:86` construit le
   `JepaWorldModel` dès que `jepa_wm.enabled` — **y compris sous `--no-world-model`**. Sans
   sentence-transformers (extra `[jepa]`), le banc canonique crashe à la 45e seconde.
   Idem `tau2-bench[gym]` : `tau2.registry` importe `gymnasium` sans condition.
   Les deux sont désormais dans `install_pinned.sh`.

`scripts/serve_qwen_vllm.sh` (marlin + HF_HOME) est **déjà commité** (`006abf5`). Reste juste ce
`TODO.md` à committer.

## 🔬 SESSION 2026-07-16 — LE RANKER EST MORT (mesuré, pas conçu). LIRE AVANT DE REPRENDRE.

> Deux architectures tuées le même soir : le **vérificateur d'arguments (« Q »)** puis le
> **ranker**. Les deux avaient été conçues sur la décomposition d'**un seul run à temperature
> 0.7**. Ce qui suit est mesuré et re-exécutable ; les instruments sont versionnés.

### A. Le JEPA-WM comme ranker est SOUS LE HASARD (`scripts/replay_ranker_offline.py`)

Le rollout JEPA à H=1 **n'appelle jamais la politique** (`jepa_world_model.py::rollout` = pure
arithmétique latente sur `(goal, state.text, str(action))`), et `State.text == observation.text`
(`types.py:46`) que la trace persiste en `real_state`. ⇒ **la décision du ranker est rejouable
hors-ligne, sans vLLM** : `state.text(tour t) == trace[t-2].real_state`. Répondre à « le ranker
préfère-t-il l'écriture à la relecture ? » coûte des MINUTES, pas des heures de GPU.

Sur `runs/retail74_baseline` (39 ép., 25,6 %), 272 pas à K>1, 26 pas décisifs / 14 épisodes :

| | élit l'écriture experte |
|---|---|
| Ranker JEPA (format live) | **2/26 (8 %)** |
| Ranker JEPA (format d'entraînement) | 4/26 (15 %) |
| Hasard uniforme | 7,5 attendus · **p(hasard ≤ 2) = 0,0079** |
| **Baseline `candidates[0]`** | **14/26 (54 %)** |

- **Ce n'est pas du bruit** : étendue médiane intra-candidate-set **0,099** (vs 0,176 intra-épisode,
  vs 0,0086 du dégénéré historique). Le ranker choisit avec ASSURANCE, contre la cible.
- **Mécanisme** : il fuit `[0]` (17 % vs 25 % à l'uniforme ; distribution `{0:45, 1:81, 2:57, 3:89}`)
  alors que l'écriture experte EST en `[0]` dans **14/26** des pas décisifs (`{0:14, 1:7, 2:4, 3:4}`).
- **Downside** : dévie sur **93 %** des pas des **10** épisodes qui réussissent déjà.
- **Marché adressable réel : 3/39 = 7,7 points** (tasks 0, 16, 24 — écriture proposée, jamais
  exécutée), PAS 26. Le ranker en sauve **0/3**. ⇒ **A/B baseline-vs-JEPAWM ANNULÉ** : il
  mesurerait une régression dont la cause est déjà connue.
- ⚠️ Les 26 pas ne sont pas indépendants (3 tours corrélés viennent de l'ep0) ⇒ le p est
  optimiste. Au niveau ÉPISODE : 2/14.

### B. ⛔ « L'écriture experte n'est JAMAIS en candidates[0] » = ARTEFACT DE FILTRE

Le constat fondateur du renversement (« Qwen échantillonne la bonne action et ne la sélectionne
pas ») venait d'un script filtrant `key(parse(c)) != key(chosen)` pour « lister les candidats non
choisis ». **La baseline choisit TOUJOURS `[0]`** (`loop.py:122`) ⇒ ce filtre exclut par
CONSTRUCTION tous les `[0]`. Il ne pouvait produire que « jamais en [0] » : c'était la définition
du filtre, pas une mesure. Vérifié des deux côtés : 9/19 pas décisifs sur 30 ép. (47 %),
14/26 sur 39 ép. (54 %) — **et les 9 ont été exécutés**.

⇒ **Qwen n'a pas un problème de sélection : son premier échantillon est l'écriture experte une
fois sur deux aux pas qui comptent.** La baseline sans ranker est déjà bonne là.

### C. Le mode d'échec DOMINANT : l'écriture prématurée fausse (`scripts/decompose_failures.py`)

Décomposition **corrigée de `tool_error`** (compter `chosen` sans lui donnait « 3 épisodes ont
tout écrit et échoué » — réel : **0**, les 3 avaient été REJETÉS par l'env) :

| mode | n / 29 échecs |
|---|---|
| **écriture PRÉMATURÉE FAUSSE qui RÉUSSIT** | **12** |
| aucune écriture tentée | 8 |
| écritures expertes partielles | 8 |
| **écritures expertes complètes** | **0** |
| pas d'écriture requise | 1 |

**τ²-retail est IRRÉVERSIBLE** (`policy.md:84` « Exchange or modify order tools can only be called
once per order » ; `:110` « The agent will not be able to modify or cancel the order anymore » ;
`:118`/`:130` return/exchange exigent `delivered`). Sur **4 épisodes (2, 20, 27, 34)** la mutation
fausse **FORCLÔT** l'experte : task 2 → `exchange` réussi au t12, puis le `return` expert au t16
→ `Error: Non-delivered order cannot be returned`.

⇒ **L'hypothèse « la clôture prématurée est accidentellement alignée sur le fix » était à
l'envers : la clôture prématurée EST le mode d'échec.** Et le ranker préfère mesurablement les
écritures (0,685 vs 0,667 pour les lectures, Δ=0,018 — noyé sous une étendue de 0,099). Il
déclencherait les écritures PLUS TÔT dans un domaine où écrire trop tôt est définitif.

### D. Ce n'est PAS un problème d'information (donc le RAG ne le sauve pas)

L'agent reçoit la policy τ² COMPLÈTE en system_context (`tau2_adapter.py:97` → `:139`), règle
one-shot incluse. **Il a la règle sous les yeux et la viole.** Par ailleurs `use_rag` vaut
`False` par défaut (`config.py:47`) et le banc ne l'active pas ⇒ la KB tronquée est **inerte**
sur ce run. ⇒ pathologie de la POLITIQUE, pas du contexte ni de la sélection.

### E. 5e BUG — mismatch de format d'action, dans le chemin du ranker

Le prédicteur est entraîné sur `name({"order_id": "#W123"})`
(`replay_reference_trajectories.py:95`, `json.dumps`) et reçoit `name(order_id='#W123')` en live
(`types.py:23`, `repr()`). **Tout appel JEPA-WM est hors distribution** : `predict`, `divergence`,
`rollout`. Corollaire : le **seuil δ=0,20** (`jepa_world_model.py:53`) a été calibré (sonde F5)
sur un format qui ne se produit JAMAIS en live ⇒ le routage de surprise tourne sur une
calibration fantôme. Corriger fait passer le ranker de 2/26 à 4/26 : réel, mais ne sauve rien.
- [ ] **À corriger** : aligner les deux formats (choisir `json.dumps` partout, ou re-générer
      l'alignement au format `repr`), PUIS re-calibrer δ, PUIS re-jouer `replay_ranker_offline.py`.

### ➡️ PROCHAINE MESURE (tranche le sort de la thèse sur 12/29, pas sur 3/39)

- [ ] **Aux 12 pas fatals, le candidate set contenait-il une action NON fatale ?** Si oui (un
      `respond_to_user` de confirmation, la lecture de statut manquante), un ranker avait de quoi
      éviter la catastrophe et le marché est > 3. Si le set n'a que des variantes de l'écriture
      fatale, **le ranker est définitivement hors jeu et c'est le PROPOSEUR**. C'est une mesure,
      pas un design.
- [ ] **Finir les 74**, puis **une 2e graine sur les mêmes 74** →
      `decompose_failures.py --compare` chiffre la stabilité de la décomposition. **Ne PLUS
      concevoir contre une taxonomie dont la variance n'est pas chiffrée** : les tâches 0/1/5
      donnaient C1/A/C2 sur un run et D/D/D sur le suivant. C'est ce qui a coûté deux jours.
- ⚠️ **Ce qui est robuste tient déjà** (272 points de décision, peu sensibles au run) : « sous le
      hasard », « 93 % de déviation », « 0/29 écriture experte complète ».

### ⚠️ Pile réellement installée ≠ pile « figée » du Journal §1/§2

Mesuré le 2026-07-16 sur le pod qui SERT : **vllm 0.25.1 · torch 2.11.0+cu130 · transformers
5.14.1** — soit exactement ce que le Journal interdit. Ça tourne parce que le pod a été
reprovisionné avec un **driver CUDA 13.0** : le pin cu128 existait pour un driver 12.8, et le
`transformers<5` pour un bug de tokenizer de vllm **0.10.2** (que 0.25.1 n'a pas). ⇒ ne pas
« réparer » la pile en la downgradant vers le Journal : on casserait un serveur qui marche.
`sentence-transformers` s'y ajoute sans rien bouger (vérifié : install additive, serveur intact).

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

### Décisions tranchées (2026-07-13)

1. **Policy du domaine → contexte système dédié FAIT.** `Env.system_context()` renvoie la policy
   du domaine τ² (le manuel légitime, ce que le vrai agent a en système) ; `Policy.propose` la reçoit
   au **vrai pas PROPOSER seulement**. Les rollouts imaginés du world-model appellent `propose()` sans
   elle → prompts K·H **bornés** (mock renvoie `None`). Testé (`tests/test_system_context.py` :
   injectée au vrai propose, ABSENTE des prompts world-model). Vérifié en réel : run retail 1 tâche
   avec policy complète → OK, pas d'erreur de contexte, l'agent applique les règles + dialogue.
2. **But latent `g` → statique générique** (retenu). `score_to_goal` reste un signal faible assumé en
   non-solo (le world-model s'appuie surtout sur la divergence des observations). `g` **dynamique**
   distillé du dialogue = raffinement différé (touche `loop.py`), à faire une fois JEPA branché.

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
      **DÉBLOQUÉ (2026-07-13)** — voir la puce « DÉBLOCAGE DONNÉES FAIT » ci-dessus. Le validateur
      a tourné sur 112 positifs + 57 négatifs (rejeu des actions de référence) : **H1 PASS, H2 FAIL**
      ⇒ gate ❌, signal NON goal-relative confirmé sur du vrai. Les pré-requis 1-3 restants sont soit
      levés, soit sans objet pour ce constat :
      1. ~~**0 trajectoire τ² RÉSOLUE**~~ : contourné sans le LLM via le rejeu expert
         (`scripts/replay_reference_trajectories.py`). Un vrai run Qwen (args réels) reste utile plus
         tard pour des négatifs « naturels », mais n'est plus bloquant pour H1/H2.
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
- [ ] **Phase 4** : routeur de surprise appris ; POC **DSPy** sur `agents/policy.py`
      (optimiseur avec la réussite-vs-tours comme métrique).
      - [x] **Signaux INSTRUMENTÉS (2026-07-14)** : `SurpriseSignals` (`agents/surprise.py`) =
            les 5 signaux de specs/01 (amplitude δ, signature outil, direction score_before/after,
            cohérence RAG `kb_top_score`/`kb_hits` + `memory_hits`, localité `familiarity`,
            réductibilité) **+ 2 signaux tirés de la rubrique d'annotation réelle**
            (`data/annotations` : `repeated_tool` ← loop_no_progress, `is_user_turn` ←
            user_new_info). Collecté à chaque surprise, journalisé dans `TraceStep.signals`
            (+ `tool_error`/`score_before` à chaque tour) → joignable aux 109 annotations par
            (episode, turn). Sonde « réductibilité » opt-in (`orchestrator.use_reducibility`,
            +1 appel LLM/surprise, sans objet en JEPA-WM — latent non verbalisable).
            **Routage Phase 1 INCHANGÉ** (2 signaux) : comparabilité des runs préservée ;
            `as_vector()` fige l'ordre des features (None ≠ 0 : indicateurs sondé/non-sondé).
            Tests : `tests/test_signals.py` (7) ; suite complète 49 verts.
      - [x] **ROUTEUR APPRIS v0 ENTRAÎNÉ (2026-07-14) — bat la règle Phase 1** :
            pipeline complet `scripts/build_router_dataset.py` (jointure traces↔annotations,
            recompute hors-ligne fidèle à loop.py via `router/features.py` — même détecteur
            d'erreur τ², même requête KB/mémoire, replay FactMemory) → `morpheus train-router`
            (`router/train.py` : régression logistique numpy pur, gradient À LA MAIN, CV
            stratifiée 5 plis, baselines majoritaire + heuristique Phase 1, p par permutation).
            **Résultat sur les 109 annotations** : balanced accuracy **0.977** out-of-fold
            (29/30 ERREUR, 78/79 NOUVEAUTÉ) vs heuristique Phase 1 **0.817** (19/30 ERREUR —
            aveugle aux boucles sans progrès), **Δ +0.160, p≈0**. Poids lisibles :
            tool_error +2.29 · is_user_turn −1.47 (user_new_info→NOVELTY) · repeated_tool
            +1.32 (loop_no_progress→ERROR) · familiarity +1.16 (⚠️ SENS INVERSE de l'intuition
            specs/01 : le déjà-vu lexical ⇒ ERREUR/stagnation, pas extension). Artefacts
            versionnés : `data/router/dataset.jsonl` + `checkpoints/router/router.json`.
            **Caveats honnêtes** : (1) annotateur = modèle (relecture humaine recommandée) et
            la rubrique d'annotation UTILISE des features observables (tool_error,
            loop_no_progress) → le score mesure d'abord « le vecteur reconstruit la rubrique » ;
            (2) 9 épisodes seulement → durcir avec CV leave-one-episode-out + labels frais ;
            (3) direction (score_before/after) NON sondée localement (sans torch) — re-générer
            le dataset avec `--checkpoint checkpoints/jepa_tau2_align/jepa.pt` sur le pod.
            **Reste pour boucler Phase 4** : brancher `RouterModel.route` dans loop.py derrière
            un flag (`router_checkpoint`), mesurer en live vs règle Phase 1 sur τ²-retail.
      - [x] **REVUE HUMAINE DES ANNOTATIONS (2026-07-14) — 3 découvertes traitées** :
            (1) **Circularité QUANTIFIÉE** : 108/109 labels v1 = fonction mécanique de 3 features
            (tool_error→ERR ; respond_to_user→NOV ; outil répété hors dialogue→ERR ; sinon NOV).
            Traitement : baseline `[RUBRIQUE]` ajoutée à `train-router` — le verdict affiche
            désormais appris-vs-rubrique (le plafond de circularité) en permanence.
            (2) **`retail_postfix#5` t16 RETIRÉ** (109→108, 29 ERREUR / 79 NOUVEAUTÉ) : artefact
            de troncature harnais (observation d'outil recyclée après un respond_to_user au
            plafond de tours — cas unique, vérifié). Documenté dans data/annotations/README.md.
            (3) **`repeated_tool` corrigé** (loop.py + router/features.py, constante partagée
            `DIALOGUE_TOOL`) : les respond_to_user consécutifs = dialogue normal, exclus du
            signal boucle (18/28 répétitions du corpus étaient du dialogue légitime).
            **Chiffres après correction** : appris **0.976** (28/29 ERR, 78/79 NOV) ·
            rubrique **0.983** · heuristique Phase 1 **0.810** · Δ appris-rubrique **−0.007**
            ⇒ le routeur RECONSTRUIT la rubrique, il ne la dépasse pas encore (attendu).
            Preuve par les 2 seules fautes du modèle : FN = retail#1 t15 `coherent_but_wrong`
            (LE cas de la thèse — invisible sans direction/sémantique) ; FP = retail#0 t11
            transfer réussi (sur-généralisation de la monoculture transfer-loops).
            Poids sains post-fix : is_user_turn −1.47→−0.80 (plus besoin de compenser le bruit
            dialogue), repeated_tool +1.32→+1.40. **Le prochain lot d'annotations doit viser
            les cas hors-rubrique** : cohérent-mais-faux, détours légitimes, boucles non-transfer.
      - [x] **ROUTEUR BRANCHÉ dans la boucle derrière un flag (2026-07-14)** :
            `orchestrator.router_checkpoint` (défaut `None` = règle Phase 1 → runs existants
            inchangés). `loop.py` n'a PAS bougé : `RouterModel.route` a déjà le contrat de
            `SurpriseRouter.route` — le contrat est désormais explicite (`agents/surprise.py::Router`,
            Protocol) et la substitution vit dans `eval/runner.py::build_router`. numpy importé
            paresseusement (extra `[jepa]`/`[dev]`), pas de torch à l'inférence.
            **GARDE DE RÉGIME (le vrai piège, trouvé en câblant)** : `as_vector()` épingle à 0 un
            signal non sondé, valeur que le modèle lit comme RÉELLE et hors distribution ⇒ dérive
            CONSTANTE du logit. Le dataset est bâti avec KB+mémoire sondées (défauts de
            `build_router_dataset.py`) alors que `qwen_tau2*.yaml` laissent `use_rag`/`use_memory`
            à **False** ⇒ **−2.4 à −2.7 logit vers NOUVEAUTÉ à chaque décision**, en silence. Les
            indicateurs `*_probed` étaient censés couvrir ce cas mais sont CONSTANTS au train ⇒
            gradient nul ⇒ poids 0 ⇒ inertes : le modèle ne peut PAS détecter qu'il est hors régime.
            `RouterModel.regime_drift()` le chiffre hors-ligne (le `mu` d'une colonne indicatrice
            EST le taux de sondage du train — lisible sur les checkpoints déjà versionnés) et
            `build_router` **refuse le run** au-delà de `MAX_REGIME_DRIFT=1.0` logit, AVANT le run
            GPU. ⇒ **pour mesurer en live : `use_rag: true` + `use_memory: true`** dans la config,
            ou ré-entraîner sur le régime visé. Tests : 6 de plus (`test_router.py`), suite **80 verts**.
      - [x] **DATASET AVEC DIRECTION MESURÉ (2026-07-14) — 2 résultats qui CHANGENT le plan** :
            `data/router/dataset.jsonl` re-généré avec `--checkpoint jepa_tau2_align` (direction
            sondée 99/108 ; committer cet artefact). `direction` prend enfin un poids réel
            **−0.639**, du BON signe (s'éloigner du but ⇒ ERREUR). Mais :
            1. **⛔ La direction N'ATTRAPE PAS `coherent_but_wrong`** — hypothèse du TODO
               **RÉFUTÉE** sur le seul exemple disponible. `retail_postfix#1` t15 :
               score_before 0.6206 → score_after 0.6251, **direction = +0.0045** — signe INVERSE
               de ce qu'il faudrait, et magnitude dans le bruit (~2,5 % de l'étendue intra-épisode
               de 0.176). L'action cohérente-mais-fausse se RAPPROCHE du but dans le latent.
               Toujours FN out-of-fold. Idem le FP (retail#0 t11) : direction −0.0148. **Les deux
               seuls cas qui comptent ont |direction| < 0.015** ⇒ sous but GÉNÉRIQUE, la direction
               à 1 pas est inexploitable (cohérent avec le caveat « rollout 1-pas bruité » déjà
               noté). ⇒ le nœud n'est pas le sondage : c'est le **but d'issue par tâche**
               (+ horizon>1), sinon un signal sémantique dédié.
            2. **⚠️ La règle Phase 1 S'EFFONDRE quand la direction est sondée : 0.810 → 0.646**
               (FP **0 → 26**, recall_novelty 1.0 → 0.671). Son branchement `d < 0 ⇒ ERREUR`
               seuille DUR un signal bruité ; il ne marchait que parce que la direction n'était
               JAMAIS sondée (la règle dégénérait en « tool_error seul »). Le routeur appris, lui,
               n'en fait qu'un poids doux (−0.639) et encaisse le bruit. ⇒ **le « 0.810 » du TODO
               ne décrit PAS la baseline live** : en JEPA-WM, `loop.py` sonde score_before/after,
               donc la règle Phase 1 live vaut ~0.646 et Δ(appris − Phase 1) = **+0.330**, pas
               +0.166. À énoncer tel quel : l'écart vient autant de l'effondrement de la baseline
               que du mérite du routeur.
            **Inchangé** : appris 0.976 vs rubrique 0.983 (Δ −0.007) — la circularité reste le
            plafond, et seules des annotations hors-rubrique la lèveront (cf. puce précédente).
            **Checkpoint versionné RÉ-ENTRAÎNÉ sur ce dataset** (`checkpoints/router/router.json`) :
            il est désormais reproductible depuis `data/router/dataset.jsonl` (même régime :
            kb+mémoire+direction sondées). Son `meta` reflète les chiffres ci-dessus (heuristique
            Phase 1 = 0.646, pas 0.810 : voir point 2). L'ancien checkpoint (kb+mémoire seules,
            sans direction) reste dans l'historique git (`fea09ca`).

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
