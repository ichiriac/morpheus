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

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
# Nombre de GPU (tensor parallel). 1 par défaut ; 2 si le modèle ne tient pas sur une carte.
TP="${TP:-1}"

echo ">> Modèle : $MODEL | port : $PORT | max_len : $MAX_LEN | TP : $TP"

# Quantization auto-détectée par vLLM pour les repos *-AWQ / *-GPTQ.
QUANT_ARG=()
case "$MODEL" in
  *AWQ*)  QUANT_ARG=(--quantization awq) ;;
  *GPTQ*) QUANT_ARG=(--quantization gptq) ;;
esac

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP" \
  --enable-prefix-caching \
  "${QUANT_ARG[@]}"
