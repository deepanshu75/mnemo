from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from mnemo.core.store import StoredChunk
from mnemo.utils.scoring import reciprocal_rank_fusion


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


@dataclass(frozen=True)
class RetrievalHit:
    chunk: StoredChunk
    score: float


class HybridRetriever:
    def search(
        self,
        *,
        query: str,
        chunks: list[StoredChunk],
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not chunks:
            return []
        # Lexical ranking: exact/keyword overlap via BM25.
        corpus_tokens = [_tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(_tokenize(query))

        # Semantic ranking: cosine-like score on normalized vectors.
        vectors = np.stack([chunk.embedding for chunk in chunks], axis=0)
        vector_scores = np.dot(vectors, query_embedding)

        bm25_ranking = [idx for idx, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)]
        vector_ranking = [idx for idx, _ in sorted(enumerate(vector_scores), key=lambda x: x[1], reverse=True)]
        # Fuse both rankings so neither signal dominates alone.
        fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievalHit(chunk=chunks[idx], score=score) for idx, score in ordered]
