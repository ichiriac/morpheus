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
# Cache HF sur le volume PERSISTANT /workspace (pas l'overlay / éphémère de RunPod) :
# garantit qu'un redémarrage ne re-télécharge PAS les ~19 Go de poids. Surchargeable.
export HF_HOME="${HF_HOME:-/workspace/.hf-cache}"
# 48 Go : 32768 confortable. Sur 24 Go, exporter MAX_LEN=16384 pour éviter l'OOM du cache KV.
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
# Nombre de GPU (tensor parallel). 1 par défaut ; 2 si le modèle ne tient pas sur une carte.
TP="${TP:-1}"

echo ">> Modèle : $MODEL | port : $PORT | max_len : $MAX_LEN | TP : $TP"

# Si `vllm` n'est pas sur le PATH, auto-activer le venv créé par install_pinned.sh.
# Défaut /root/.venv-morpheus : HORS QUOTA (le venv pèse 9,4 Go, /workspace n'a ~5,4 Gio libres
# sur un plafond mesuré à 36119 Mio ≈ 35,3 Gio le 2026-07-16). Le `.venv/` du dépôt est un vestige :
# il n'a jamais pu y tenir. Gardé en secours au cas où quelqu'un aurait un layout à lui.
VENV_DIR="${VENV_DIR:-/root/.venv-morpheus}"
if ! command -v vllm >/dev/null 2>&1; then
  for _v in "$VENV_DIR/bin/activate" ".venv/bin/activate"; do
    if [ -f "$_v" ]; then
      echo ">> venv détecté, activation de $_v"
      # shellcheck disable=SC1091
      source "$_v"
      break
    fi
  done
fi

# Choisir le lanceur : CLI `vllm serve` (récent) ou module python (fallback).
if command -v vllm >/dev/null 2>&1; then
  RUNNER=(vllm serve "$MODEL")
elif python -c "import vllm" >/dev/null 2>&1; then
  RUNNER=(python -m vllm.entrypoints.openai.api_server --model "$MODEL")
else
  echo "!! vLLM introuvable. Lance d'abord :  bash scripts/install_pinned.sh" >&2
  echo "   puis :  source $VENV_DIR/bin/activate  &&  bash scripts/serve_qwen_vllm.sh" >&2
  echo "   (n'installe PAS vllm à la main dans l'env courant : la pile est figée, cf. le" >&2
  echo "    journal TODO §1-2 — une version libre mesure sur une autre pile en silence.)" >&2
  exit 127
fi

# Quantization pour les repos *-AWQ / *-GPTQ. On force les kernels *Marlin* (Ampere+),
# bien plus rapides que awq/gptq « legacy » — sinon vLLM avertit « awq quantization is not
# fully optimized yet » et le débit s'effondre (~3 tok/s vs 20-40 tok/s en marlin sur A40).
QUANT_ARG=()
case "$MODEL" in
  *AWQ*)  QUANT_ARG=(--quantization awq_marlin) ;;
  *GPTQ*) QUANT_ARG=(--quantization gptq_marlin) ;;
esac

exec "${RUNNER[@]}" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP" \
  --enable-prefix-caching \
  "${QUANT_ARG[@]}"
