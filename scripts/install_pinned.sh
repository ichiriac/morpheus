#!/usr/bin/env bash
# Install PINNED (journal TODO §1-2) — pile connue-bonne, PAS runpod_setup.sh (vllm>=0.6.0 casse).
# Driver actuel = CUDA 13.0 (rétro-compatible avec les wheels cu128).
set -euo pipefail
cd /workspace/morpheus

echo ">> Python : $(python3 --version)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

# §1 : pile figée — vLLM 0.10.2 + torch 2.8.0+cu128 (évite vllm 0.25 → torch cu130).
pip install "vllm==0.10.2" --extra-index-url https://download.pytorch.org/whl/cu128

# §2 : transformers < 5 (5.x casse le tokenizer de vLLM 0.10.2).
pip install "transformers>=4.55.2,<5"

# morpheus + connecteurs.
pip install -e ".[openai,anthropic,dev]"

echo ">> INSTALL_OK"
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
