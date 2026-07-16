# 04 — Étape 1 : brancher Qwen réel (RunPod + vLLM)

> But de l'étape : servir un vrai Qwen sur GPU et **valider que la politique produit un
> format d'action parsable sur le mock**, avant de passer à τ²-bench. Aucune modification
> du JEPA ici — on remplace juste le backend `stub` par `openai` (Qwen via vLLM).

## Vue d'ensemble

```
   RunPod pod (Linux + 1 GPU)
   ┌─────────────────────────────┐
   │  vLLM serve Qwen3-32B        │  ← scripts/serve_qwen_vllm.sh  (port 8000)
   │  endpoint OpenAI-compatible  │
   └──────────────┬──────────────┘
                  │ http://localhost:8000/v1
   ┌──────────────▼──────────────┐
   │  morpheus (policy+WM=openai) │  ← configs/qwen_local.yaml
   │  check-llm → run (mock)      │
   └─────────────────────────────┘
```

> **DÉCISION FIGÉE** : moteur **vLLM** + **GPU retenu : NVIDIA A40 48 Go**, modèle
> **Qwen3-32B-AWQ** avec `MAX_LEN=32768` (gros cache KV, batch large → bon pour les rafales MPC).

## Choix du GPU / modèle

| Modèle | VRAM (approx.) | GPU | Notes |
|---|---|---|---|
| `Qwen/Qwen3-32B-AWQ` (4-bit) | ~20-22 Go + KV | **A40 48G (retenu)** / RTX 4090 24G | **défaut**. Sur 48 Go : `MAX_LEN=32768`, débit élevé |
| `Qwen/Qwen3-32B-GPTQ-Int8` (8-bit) | ~34 Go + KV | A40 48G | montée en qualité si le jugement long est juste (plus lent) |
| `Qwen/Qwen3-Coder-30B-A3B` | ~18-20 Go (Q4) | 24G / 48G | MoE rapide (3B actifs), orienté coding/tool-use |
| `Qwen/Qwen3-32B` (bf16) | ~64 Go | A100 80G / H100 | pleine précision — **ne rentre pas** dans 48 Go |

> **A40 = architecture Ampere (GA102) → pas de FP8 natif** (FP8 = Ada/Hopper). Rester en
> **AWQ/GPTQ entier**. Le script auto-détecte AWQ/GPTQ. Démarrer en AWQ 4-bit (débit) ; basculer
> GPTQ 8-bit seulement si la qualité du raisonnement multi-tours le justifie.

## Procédure

> ### ⚠️ Layout disque — le piège n°1 du pod (mesuré le 2026-07-16)
>
> | | `/workspace` | `/root` |
> |---|---|---|
> | nature | volume réseau MooseFS | overlay conteneur |
> | survie au restart | **oui** | **non** |
> | quota | **31376 Mio ≈ 30,6 Gio** (mesuré au `dd`) | aucun (50 Go) |
> | contenu | dépôt, données, poids HF (19 Go), tau2-bench | **venv** (9,4 Go), cache pip |
>
> Le venv **ne rentre pas** sur `/workspace` : 9,4 Go demandés, ~0,8 Go libres. Un
> `python3 -m venv .venv` depuis le dépôt échoue en `Errno 122` **à mi-install**. `df` ment ici
> (il annonce 756 Tio, le cluster entier) — voir `TODO.md` §7 pour la méthode de mesure.
> Contrepartie : le venv est à refaire après chaque restart (~10 min) ; les 19 Go, jamais.

```bash
# sur le pod, après git clone du repo
bash scripts/install_pinned.sh               # LA recette. Pile figée + venv hors quota.
#   variables : VENV_DIR (défaut /root/.venv-morpheus), REPO, TAU2, PIP_CACHE_DIR
#   (`runpod_setup.sh` y délègue : il n'existe plus qu'une seule procédure d'install)

# terminal 1 : servir Qwen (laisser tourner ; tmux conseillé)
MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh
#   variables : MODEL, PORT, MAX_LEN, GPU_UTIL, TP (tensor-parallel = nb de GPU)

# terminal 2 : valider le branchement + le format de la politique
source /root/.venv-morpheus/bin/activate
morpheus check-llm --config configs/qwen_local.yaml
```

> **Pile FIGÉE — ne pas « mettre à jour ».** `install_pinned.sh` pose `vllm==0.10.2` +
> `torch 2.8.0+cu128` + `transformers 4.57.6` (<5). `pip install "vllm>=0.6.0"` installe vLLM
> 0.25 + torch cu130 + transformers 5.x : **ça s'installe, ça sert le modèle, et ça mesure sur
> une autre pile sans rien signaler** — c'est-à-dire le pire mode d'échec. Toutes les mesures du
> banc ont été prises sur la pile figée ; en changer invalide silencieusement les comparaisons.

### Ce que `check-llm` vérifie (les 3 étapes)
1. **Ping** : le serveur répond.
2. **Sortie BRUTE de la politique** sur une tâche mock — pour voir *exactement* ce que Qwen
   renvoie (format, blocs `<think>`, prose parasite).
3. **Parsing** : après strip du raisonnement + snap sur la whitelist d'outils, on vérifie
   que la 1re étape attendue (`authenticate_user`) est bien proposée.

Sortie attendue en fin : `→ la politique la propose : OUI ✅`.

## Si le format ne passe pas

Le parseur (`agents/policy.py::_parse_actions`, via `text.py`) est déjà tolérant :
- retire `<think>…</think>` et les fences ```` ``` ```` ;
- lit les lignes `ACTION: <tool> | ARGS: {…}` ;
- **filet** : à défaut de format, repère les noms d'outils autorisés cités dans le texte ;
- **snap** tout nom vers la whitelist (casse, recouvrement de tokens) ; écarte les hallucinations.

Leviers si Qwen dévie encore :
- **Mode thinking** : `configs/qwen_local.yaml` le désactive
  (`extra_body.chat_template_kwargs.enable_thinking: false`). Le réactiver peut améliorer la
  qualité au prix de la latence (le parseur strip de toute façon les blocs).
- **Température** : baisser `policy.temperature` (0.3-0.5) pour un format plus régulier.
- **Prompt** : durcir `_SYS` dans `agents/policy.py` (few-shot si besoin).

## Étape suivante (après validation sur le mock)

1. **Mesurer la baseline sur le mock** — utile comme sanity check du pipeline avec Qwen réel :
   ```bash
   morpheus run --config configs/qwen_local.yaml --no-world-model --out runs/qwen_baseline
   morpheus run --config configs/qwen_local.yaml --out runs/qwen_wm
   ```
2. **Câbler τ²-bench** (`envs/tau2_adapter.py`, `TODO(tau2)`) puis basculer `eval.env: tau2`.
3. Ajouter la **ligne de référence Sonnet 4.6** (config `kind: anthropic`).

C'est seulement après ça qu'on entame le **JEPA latent (Phase 2)**.
