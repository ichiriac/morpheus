#!/usr/bin/env bash
# Install PINNED — LA procédure d'install du projet. Pile connue-bonne (journal TODO §1-2).
# `runpod_setup.sh` délègue ici : il n'existe plus de seconde recette.
#
# Usage :
#   bash scripts/install_pinned.sh                      # venv dans /root/.venv-morpheus
#   VENV_DIR=/ailleurs/.venv bash scripts/install_pinned.sh
#
# ┌─ LAYOUT DISQUE — mesuré le 2026-07-16, ne pas rediscuter sans re-mesurer ─────────────┐
# │ /workspace = volume réseau MooseFS, PERSISTANT, mais SOUS QUOTA :                     │
# │     plafond MESURÉ (2026-07-16, après relèvement) = 36119 Mio ≈ 35,3 Gio.             │
# │     Méthode : chunks dd jusqu'à `Disk quota exceeded`. Deux méthodes indépendantes    │
# │     concordent à 3 Mio près (base+écrit = 36116 ; du au refus = 36119), sur deux      │
# │     granularités (128/8 Mio et 256/16 Mio).                                           │
# │     Le quota PROVISIONNÉ est de 38 Go décimaux (36240 Mio nominal ; mesuré −0,33 %,   │
# │     même signature que la mesure précédente : 33 Go → 31376 Mio, −0,3 %). Il avait    │
# │     été ANNONCÉ à 35 Go — ce qui aurait fait +8,2 % d'écart : l'annonce était fausse. │
# │     Historique : 31376 Mio (≈33 Go) avant le relèvement du 2026-07-16.                │
# │     `df` est INUTILISABLE ici : il annonce 756 Tio (la taille du cluster MooseFS, pas │
# │     la part du pod) ; `statvfs` ment pareil ; pas de binaire mfs*, pas de getfattr ;  │
# │     fallocate non supporté. La seule mesure fiable est empirique.                     │
# │   → y vivent : le dépôt, les données, les poids HF (19 Go), tau2-bench.               │
# │                                                                                       │
# │ /root (overlay 50 Go) = ÉPHÉMÈRE, meurt au restart du conteneur, mais SANS quota.     │
# │   → y vivent : le venv (9,4 Go) et le cache pip.                                      │
# │                                                                                       │
# │ POURQUOI le venv NE PEUT PAS aller sur /workspace : il pèse 9,4 Go et le volume est   │
# │ occupé à ~29,9/35,3 Gio (30612/36119 Mio) ⇒ ~5,4 Gio libres : insuffisant, et ça      │
# │ reste vrai APRÈS le relèvement à 38 Go.                                               │
# │ Ce n'est pas un réglage, c'est de l'arithmétique : `python3 -m                        │
# │ venv .venv` depuis le dépôt échoue en `Errno 122`, à mi-installation, APRÈS plusieurs │
# │ minutes de téléchargement. C'est le bug qui a coûté la soirée du 2026-07-15.          │
# │ Idem PIP_CACHE_DIR : le pointer sur /workspace remplit le quota et fait échouer       │
# │ l'install (le cache y a gonflé de 1,4 à 6,8 Go en une seule install ratée).           │
# │ Contrepartie ASSUMÉE : le venv est à refaire à chaque restart (~10 min). Les 19 Go de │
# │ poids, eux, ne se re-téléchargent JAMAIS (HF_HOME=/workspace/.hf-cache).              │
# │ NB : /workspace/.pip-cache (6,8 Go) est du POIDS MORT — il contient la pile NON figée │
# │ (vllm 0.25 / torch cu130) et rien ne le lit. Le purger rendrait ~12,2 Gio libres, et  │
# │ le venv y tiendrait alors. Tentant, mais /workspace est un FS RÉSEAU : les imports    │
# │ Python (vllm, torch) y seraient lents. Mesurer avant de déplacer.                     │
# └───────────────────────────────────────────────────────────────────────────────────────┘
set -euo pipefail
# pipefail est VITAL : sans lui, `bash install.sh | tee log` renvoie le code de tee, et un
# `Disk quota exceeded` en pleine install se présente comme un SUCCÈS (exit 0). C'est
# exactement comme ça que le bug de quota est passé inaperçu. Un script d'install qui peut
# mentir sur son propre code retour est un piège permanent.

REPO="${REPO:-/workspace/morpheus}"
TAU2="${TAU2:-/workspace/tau2-bench}"
VENV_DIR="${VENV_DIR:-/root/.venv-morpheus}"     # défaut HORS QUOTA (cf. layout ci-dessus)
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.pip-cache}"   # idem : hors quota
export HF_HOME="${HF_HOME:-/workspace/.hf-cache}"           # 19 Go, PERSISTANT : ne pas déplacer

cd "$REPO"
echo ">> Python  : $(python3 --version)"
echo ">> venv    : $VENV_DIR   (éphémère, hors quota)"
echo ">> pip cache : $PIP_CACHE_DIR"
echo ">> GPU     : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo '(nvidia-smi absent)')"

# Garde-fou : refuser un VENV_DIR sous /workspace — il n'y rentre pas (cf. layout).
# Échappatoire : ALLOW_WORKSPACE_VENV=1, si et seulement si vous avez RE-MESURÉ la place
# (par ex. après avoir purgé /workspace/.pip-cache). Ne vous fiez pas à `df`, il ment ici.
case "$(readlink -m "$VENV_DIR")" in
  /workspace/*)
    if [ "${ALLOW_WORKSPACE_VENV:-0}" != "1" ]; then
      echo "!! VENV_DIR pointe sous /workspace, qui est sous quota : plafond mesuré 36119 Mio" >&2
      echo "   (≈35,3 Gio) pour ~29,9 Gio occupés ⇒ ~5,4 Gio libres, or le venv pèse 9,4 Go." >&2
      echo "   L'install échouerait en Errno 122 à mi-parcours, après plusieurs minutes." >&2
      echo "   Utilise un chemin hors /workspace (défaut : /root/.venv-morpheus)." >&2
      echo "   Si vous avez fait de la place ET re-mesuré : ALLOW_WORKSPACE_VENV=1." >&2
      exit 2
    fi
    echo ">> ALLOW_WORKSPACE_VENV=1 : venv sous /workspace malgré le quota (vous avez re-mesuré)." >&2
    ;;
esac

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# §1 — pile figée : vLLM 0.10.2 + torch 2.8.0+cu128.
#   `vllm>=0.6.0` tire vLLM 0.25 → torch cu130. Historiquement : crash « driver too old ».
#   Le driver actuel (580.x / CUDA 13.0) l'accepterait, mais on reste figé : la pile figée est
#   celle sur laquelle TOUTES les mesures du banc ont été prises. En changer invalide les
#   comparaisons run-à-run sans que rien ne l'indique.
pip install "vllm==0.10.2" --extra-index-url https://download.pytorch.org/whl/cu128

# §2 — morpheus. [jepa] est OBLIGATOIRE, pas optionnel : il fournit sentence-transformers, et
#   `eval/runner.py` construit le JepaWorldModel dès que `jepa_wm.enabled` — donc MÊME sous
#   `--no-world-model`. Sans lui, le banc canonique crashe à la 45e seconde.
pip install -e ".[openai,anthropic,dev,jepa]"

# §3 — tau2-bench, en editable depuis le clone PERSISTANT. L'extra [gym] fournit gymnasium,
#   que `tau2.registry` importe sans condition (sinon ModuleNotFoundError au démarrage du run).
pip install -e "${TAU2}[gym]"

# §4 — transformers < 5, EN DERNIER : [jepa] (sentence-transformers) et tau2 tolèrent 5.x et le
#   remontent. Or 5.x casse le tokenizer de vLLM 0.10.2 (`Qwen2Tokenizer has no attribute
#   all_special_tokens_extended`). Ce pin doit donc être posé APRÈS eux, jamais avant.
pip install "transformers>=4.55.2,<5"

echo ""
echo ">> VÉRIFICATION (le code retour ne suffit pas — on exerce la pile) :"
python - <<'PY'
import torch, transformers, vllm
print(f"   vllm {vllm.__version__} | torch {torch.__version__} | transformers {transformers.__version__}")
print(f"   cuda available: {torch.cuda.is_available()} | "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
import sentence_transformers, gymnasium, tau2  # noqa: F401
print(f"   sentence-transformers {sentence_transformers.__version__} | gymnasium {gymnasium.__version__} | tau2 OK")
assert vllm.__version__ == "0.10.2", f"vllm {vllm.__version__} != 0.10.2 (pile figée violée)"
assert transformers.__version__.startswith("4."), f"transformers {transformers.__version__} >= 5 (casse le tokenizer vLLM)"
assert torch.cuda.is_available(), "CUDA indisponible"
print("   INSTALL_OK — pile figée conforme")
PY

echo ""
echo ">> Ensuite :"
echo "   source $VENV_DIR/bin/activate"
echo "   MODEL=Qwen/Qwen3-32B-AWQ bash scripts/serve_qwen_vllm.sh        # terminal 1 (tmux)"
echo "   morpheus check-llm --config configs/qwen_local.yaml             # terminal 2"
