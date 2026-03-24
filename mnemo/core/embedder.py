from __future__ import annotations

from typing import Protocol

import numpy as np

from mnemo.exceptions import EmbeddingError


class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True): ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._backend: EmbeddingBackend | None = None

    def _load_backend(self) -> EmbeddingBackend:
        if self._backend is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover
                raise EmbeddingError(
                    "sentence-transformers is required for embedding operations"
                ) from exc
            self._backend = SentenceTransformer(self.model_name)
        return self._backend

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._load_backend()
        vectors = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
