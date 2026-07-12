"""Encodeurs texte → vecteur. SANS torch (numpy).

- `HashingEncoder` : déterministe, hors-ligne, aucune dépendance — pour smoke tests / CI et
  pour dérisquer tout le pipeline sans télécharger de modèle.
- `SentenceTransformerEncoder` : vrai encodeur pré-entraîné (import paresseux).
- (RunPod) option future : hidden states de Qwen comme `E_state` (cf. specs/01, option A).

Les encodeurs renvoient des `np.ndarray` float32 (n, dim), L2-normalisés. On les traite comme
un `E_state` GELÉ : l'entraînement JEPA ne rétropropage pas dedans (Phase 2 v0).
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_TOK = re.compile(r"[a-z0-9_]+")


def _stable_hash(token: str, seed: int) -> int:
    h = hashlib.md5(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


class HashingEncoder:
    """Hashing trick + signe : bag-of-tokens projeté en `dim` dimensions, déterministe.

    Ce n'est PAS sémantique (pas d'embeddings appris) — juste un encodeur stable et gratuit
    pour valider la mécanique d'entraînement. En prod, remplacer par SentenceTransformer/Qwen.
    """

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOK.findall((t or "").lower()):
                h = _stable_hash(tok, self.seed)
                bucket = h % self.dim
                sign = 1.0 if (h >> 1) & 1 else -1.0
                out[i, bucket] += sign
        # L2-normalisation (évite les normes qui explosent avec la longueur du texte)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerEncoder requiert `pip install sentence-transformers`"
            ) from e
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


def build_encoder(kind: str = "hashing", **kwargs):
    if kind == "hashing":
        return HashingEncoder(**{k: v for k, v in kwargs.items() if k in ("dim", "seed")})
    if kind == "sentence_transformer":
        return SentenceTransformerEncoder(
            **{k: v for k, v in kwargs.items() if k in ("model_name", "device")}
        )
    raise ValueError(f"encodeur inconnu : {kind!r}")
