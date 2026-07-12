"""JEPA — world-model latent (Phase 2).

Sépare volontairement :
- `data.py` / `encoders.py` : SANS torch (normalisation des trajectoires, embeddings numpy)
  → validables sur n'importe quelle machine, y compris sans GPU.
- `model.py` / `losses.py` / `train.py` : torch (entraînement sur GPU RunPod).

Idée : `E_state` = encodeur pré-entraîné GELÉ (option A des specs), on entraîne le prédicteur
`P` (+ `E_action` + une projection) à prédire l'embedding de l'état résultant. Cf. specs/05.
"""

from .data import Transition, load_transitions, synthetic_transitions

__all__ = ["Transition", "load_transitions", "synthetic_transitions"]
