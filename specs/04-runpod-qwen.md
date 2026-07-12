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

## Choix du GPU / modèle (budget < 5000 €, ~32 Go VRAM)

| Modèle | VRAM (approx.) | GPU RunPod typique | Notes |
|---|---|---|---|
| `Qwen/Qwen3-32B` (bf16) | ~64 Go | A100 80G / H100, ou 2×48G | pleine précision, hors carte 32 Go |
| `Qwen/Qwen3-32B-AWQ` (4-bit) | ~20-22 Go | **RTX 4090 24G / A6000 48G** | **cible single-GPU du projet** |
| `Qwen/Qwen3-Coder-30B-A3B` | ~18-20 Go (Q4) | RTX 4090 24G | MoE rapide, orienté coding/tool-use |

> Pour rester fidèle à la contrainte « une carte < 5000 € », privilégier **Qwen3-32B-AWQ**
> ou **Qwen3-Coder-30B-A3B** sur une **RTX 4090 24G**. Le script auto-détecte AWQ/GPTQ.
> Pour juste débloquer le pipeline, une A100 80G en bf16 est plus simple (pas de quant).

## Procédure

```bash
# sur le pod, après git clone du repo
bash scripts/runpod_setup.sh                 # venv + vllm + morpheus[openai,anthropic,dev]

# terminal 1 : servir Qwen (laisser tourner ; tmux conseillé)
MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh
#   variables : MODEL, PORT, MAX_LEN, GPU_UTIL, TP (tensor-parallel = nb de GPU)

# terminal 2 : valider le branchement + le format de la politique
source .venv/bin/activate
morpheus check-llm --config configs/qwen_local.yaml
```

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
