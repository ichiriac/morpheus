#!/usr/bin/env bash
# Sert Qwen via vLLM sur un endpoint OpenAI-compatible (http://localhost:8000/v1).
# Cible : pod Linux RunPod avec 1 GPU. Voir specs/04-runpod-qwen.md.
#
# Usage :
#   bash scripts/serve_qwen_vllm.sh                  # défaut : Qwen3-32B
#   MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh   # variante 4-bit (~24 Go)
#
# Laisser tourner dans un terminal (ou tmux), puis dans un autre :
#   morpheus check-llm --config configs/qwen_local.yaml
set -euo pipefail

# GPU retenu = NVIDIA A40 48 Go (Ampere). Variante débit : Qwen/Qwen3-Coder-30B-A3B ;
# montée en qualité : Qwen/Qwen3-32B-GPTQ-Int8.
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-8000}"
# 48 Go : 32768 confortable. Sur 24 Go, exporter MAX_LEN=16384 pour éviter l'OOM du cache KV.
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
# Nombre de GPU (tensor parallel). 1 par défaut ; 2 si le modèle ne tient pas sur une carte.
TP="${TP:-1}"

echo ">> Modèle : $MODEL | port : $PORT | max_len : $MAX_LEN | TP : $TP"

# Si `vllm` n'est pas sur le PATH, auto-activer le venv créé par runpod_setup.sh.
if ! command -v vllm >/dev/null 2>&1 && [ -f .venv/bin/activate ]; then
  echo ">> venv détecté, activation de .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Choisir le lanceur : CLI `vllm serve` (récent) ou module python (fallback).
if command -v vllm >/dev/null 2>&1; then
  RUNNER=(vllm serve "$MODEL")
elif python -c "import vllm" >/dev/null 2>&1; then
  RUNNER=(python -m vllm.entrypoints.openai.api_server --model "$MODEL")
else
  echo "!! vLLM introuvable. Lance d'abord :  bash scripts/runpod_setup.sh" >&2
  echo "   puis :  source .venv/bin/activate  &&  bash scripts/serve_qwen_vllm.sh" >&2
  echo "   (ou installe dans l'env courant :  pip install vllm)" >&2
  exit 127
fi

# Quantization auto-détectée par vLLM pour les repos *-AWQ / *-GPTQ.
QUANT_ARG=()
case "$MODEL" in
  *AWQ*)  QUANT_ARG=(--quantization awq) ;;
  *GPTQ*) QUANT_ARG=(--quantization gptq) ;;
esac

exec "${RUNNER[@]}" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP" \
  --enable-prefix-caching \
  "${QUANT_ARG[@]}"
