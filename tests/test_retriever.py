import numpy as np

from mnemo.core.retriever import HybridRetriever
from mnemo.core.store import StoredChunk


def _chunk(cid: int, text: str, emb: list[float]) -> StoredChunk:
    return StoredChunk(
        id=cid,
        text=text,
        source=None,
        namespace="default",
        created_at="2026-01-01T00:00:00+00:00",
        embedding=np.array(emb, dtype=np.float32),
    )


def test_hybrid_retriever_uses_bm25_and_vector():
    retriever = HybridRetriever()
    chunks = [
        _chunk(1, "auth jwt token refresh", [1.0, 0.0]),
        _chunk(2, "docker compose deployment", [0.0, 1.0]),
        _chunk(3, "jwt middleware auth flow", [0.8, 0.2]),
    ]
    query_embedding = np.array([0.9, 0.1], dtype=np.float32)
    hits = retriever.search(
        query="how auth token works",
        chunks=chunks,
        query_embedding=query_embedding,
        top_k=3,
    )
    assert hits
    texts = [hit.chunk.text for hit in hits]
    assert "docker compose deployment" not in texts[:1]
    assert any("auth" in text for text in texts[:2])
