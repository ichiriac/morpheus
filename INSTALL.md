# INSTALL — repartir de zéro (RunPod / Linux + 1 GPU)

> Procédure complète pour reconstruire l'environnement morpheus après **perte du serveur**.
> Résumé opérationnel : `TODO.md` (Quickstart + Journal d'environnement). Ce fichier est le
> point d'entrée unique « from scratch ». Contexte scientifique : [`specs/`](specs/).

## 0. Ce qui survit à la perte du serveur (dans le repo git)

| Artefact | Statut | Restauration |
|---|---|---|
| Code, specs, configs, scripts | versionné | `git clone` |
| **Checkpoint JEPA validé** `checkpoints/jepa_tau2_align/jepa.pt` (gate held-out **H1+H2 PASS**) | **versionné** (exception `.gitignore`) | `git clone` |
| **Donnée de replay τ²** `data/tau2_replay/*.jsonl` | **versionné** (exception `.gitignore`) | `git clone` |
| KB/RAG `data/kb/*.md`, annotations `data/annotations/` | versionné | `git clone` |
| Poids Qwen (~19 Go), cache HF | **PERDU** | re-téléchargés au 1er lancement vLLM (~20 s de load) |
| `.venv`, autres checkpoints (`jepa/`, `jepa_apigen*/`) | **PERDU** | réinstall (§2) / réentraînement (§6) |

Rien de scientifiquement critique n'est perdu : le **seul checkpoint validé** et sa donnée source
sont dans git. Les checkpoints intermédiaires (APIGen) sont régénérables mais non requis pour la suite.

## 1. Cloner

```bash
git clone git@github.com:ichiriac/morpheus.git && cd morpheus
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## 2. Installer la pile FIGÉE

```bash
bash scripts/install_pinned.sh        # vllm 0.10.2 + torch 2.8.0+cu128 + transformers<5
                                      #  + morpheus[openai,anthropic,dev,jepa] + tau2-bench[gym]
source /root/.venv-morpheus/bin/activate
```

> **Où atterrit le venv, et pourquoi pas dans le dépôt.** `/workspace` est un volume réseau
> PERSISTANT mais **sous quota : 36119 Mio ≈ 35,3 Gio, mesuré au `dd` le 2026-07-16** (`df` y
> annonce 756 Tio — la taille du cluster, pas la vôtre). Le venv pèse **9,4 Go** et il reste
> ~5,4 Gio : il n'y rentre pas — même après le relèvement du quota à 38 Go — et
> `python3 -m venv .venv` échoue en `Errno 122` **à mi-install**.
> Défaut : `/root/.venv-morpheus` (overlay 50 Go, sans quota) — surchargeable par `VENV_DIR`,
> et le script REFUSE un `VENV_DIR` sous `/workspace`. Idem `PIP_CACHE_DIR`.
> **Contrepartie assumée** : `/root` est éphémère ⇒ venv à refaire après un restart de
> conteneur (~10 min, cache pip chaud). Les 19 Go de poids, eux, sont sur `/workspace` et ne
> se re-téléchargent jamais. Détails et méthode de mesure : `TODO.md` §7.

**Pourquoi figée** (pièges déjà payés — cf. `TODO.md` §Journal 1-2) :
- `vllm>=0.6.0` tire vLLM 0.25 → torch **cu130** → crash « NVIDIA driver too old ». On épingle
  `vllm==0.10.2` + `torch cu128` (rétro-compatible avec les drivers 12.8 **et** 13.0).
- `transformers 5.x` casse le tokenizer de vLLM 0.10.2. On épingle `>=4.55.2,<5`. Toute install
  qui tire `transformers` (`.[jepa]`, sentence-transformers) doit garder `<5`.

Vérif : la dernière ligne d'`install_pinned.sh` imprime les versions et `cuda available: True`.

## 3. Servir Qwen (terminal 1, tmux)

```bash
MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh     # HF_HOME=/workspace/.hf-cache + kernels awq_marlin gérés
```

Choix du modèle selon la VRAM (le script force les kernels **marlin**, ×9 vs legacy) :

| VRAM | Commande |
|---|---|
| **48 Go (A40)** — retenu | `bash scripts/serve_qwen_vllm.sh` (défaut AWQ 4-bit, MAX_LEN 32768) |
| 48 Go, +qualité | `MODEL=Qwen/Qwen3-32B-GPTQ-Int8 bash scripts/serve_qwen_vllm.sh` |
| 24 Go (4090) | `MAX_LEN=16384 bash scripts/serve_qwen_vllm.sh` |
| 80 Go (A100/H100) | `MODEL=Qwen/Qwen3-32B bash scripts/serve_qwen_vllm.sh` (bf16) |

> A40 = Ampere, **pas de FP8 natif** → rester en AWQ/GPTQ entier. Le 1er lancement re-télécharge
> les poids (~19 Go) dans `HF_HOME` (volume `/workspace` persistant) ; les suivants sont download-free.

## 4. Valider le LLM (terminal 2)

```bash
source /root/.venv-morpheus/bin/activate
morpheus check-llm --config configs/qwen_local.yaml           # doit finir par "OUI ✅"
```

## 5. Smoke tests

```bash
pytest -q                                                     # 48 tests verts (tests torch skippés hors GPU)
morpheus run --config configs/qwen_mock_fast.yaml --out runs/qwen_wm_fast   # boucle MPC + WM, ~1 min
```

## 6. (Optionnel) Réentraîner le JEPA validé à l'identique

Le checkpoint `checkpoints/jepa_tau2_align/jepa.pt` est déjà dans git (§0). Pour le **reconstruire**
depuis zéro (par ex. après modif de la perte) — **aucun serveur LLM requis** :

```bash
pip install -e ".[jepa]"                                      # torch déjà là ; ajoute sentence-transformers, datasets

# (a) Rejouer les trajectoires de référence τ² → data/tau2_replay/retail.jsonl (+ négatifs variés)
#     Nécessite tau2-bench installé/cloné (cf. specs/04). PAS de LLM.
python scripts/replay_reference_trajectories.py --domain retail --neg-fracs 0.4,0.65,0.9

# (b) Construire le split held-out par trajectoire (anti-leak)
python scripts/build_tau2_alignment_data.py

# (c) Entraîner l'alignement goal-relative en-domaine (~min sur GPU)
morpheus train-jepa --config configs/jepa_tau2_align.yaml

# (d) Gate officiel : doit afficher H1 PASS + H2 PASS
python scripts/validate_goal_signal.py \
    --trajectories data/tau2_replay/retail_align_val.jsonl \
    --checkpoint checkpoints/jepa_tau2_align/jepa.pt
```

## 7. Reprendre le travail en cours

Voir `TODO.md` → section **« Bilan session … / Reprise Phase 2 »**. Prochaine étape ouverte :
**mesure baseline vs JEPA-WM** sur un régime discriminant avec la politique Qwen —
`configs/qwen_tau2_jepawm.yaml` (lancer les deux variantes `--no-world-model` puis WM).

Retail scoring : le juge NL-assertions est câblable sur le vLLM local via `tau2_judge_*`
(`configs/qwen_tau2.yaml`) — Qwen-juge-Qwen = mesure indicative, pas une référence.
