#!/usr/bin/env bash
# OBSOLÈTE — ce script installait `vllm>=0.6.0` et créait le venv dans le dépôt. Les deux sont
# des pièges documentés (journal TODO §1-2), et il RESTAIT recommandé par specs/04 pendant que
# install_pinned.sh disait « PAS runpod_setup.sh ». Cette contradiction a coûté la soirée du
# 2026-07-15 : le runbook envoyait droit dans le piège que le repo documentait deux fichiers
# plus loin. Il n'y a désormais qu'UNE recette d'install.
#
# Ce qu'il faisait de faux, pour mémoire :
#   1. `pip install "vllm>=0.6.0"` → vLLM 0.25 + torch cu130 + transformers 5.x. Ça s'installe,
#      ça sert le modèle — et ça mesure sur une AUTRE pile que celle du banc, sans rien signaler.
#   2. `python3 -m venv .venv` depuis le dépôt → venv de 9,4 Go sur /workspace, qui est sous
#      quota (~30,6 Gio, ~0,8 Go libres) → `Errno 122` à mi-install.
#   3. `[openai,anthropic,dev]` sans `[jepa]` → le banc canonique crashe (sentence-transformers).
set -euo pipefail

echo "!! scripts/runpod_setup.sh est OBSOLÈTE (pile non figée + venv sous quota)." >&2
echo "   → délégation à scripts/install_pinned.sh, seule recette supportée." >&2
echo "" >&2
exec bash "$(dirname "$0")/install_pinned.sh" "$@"
