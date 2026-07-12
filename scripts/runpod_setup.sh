#!/usr/bin/env bash
# Prépare un pod RunPod (Linux + GPU) pour morpheus. À lancer une fois après le clone.
# Idéalement sur une image de base « vLLM » ou « PyTorch CUDA » RunPod.
set -euo pipefail

echo ">> Python : $(python3 --version)"
echo ">> GPU :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (nvidia-smi absent)"

# venv isolé (recommandé)
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
# vLLM tire torch + CUDA adaptés à l'image.
pip install "vllm>=0.6.0"
# morpheus + connecteurs
pip install -e ".[openai,anthropic,dev]"

echo ""
echo ">> Setup terminé. Étapes suivantes :"
echo "   1) Serveur Qwen :   bash scripts/serve_qwen_vllm.sh"
echo "   2) (autre shell)    source .venv/bin/activate"
echo "   3) Valider Qwen :   morpheus check-llm --config configs/qwen_local.yaml"
echo "   4) Baseline/WM :    morpheus run --config configs/qwen_local.yaml --no-world-model --out runs/qwen_baseline"
echo "                       morpheus run --config configs/qwen_local.yaml --out runs/qwen_wm"
