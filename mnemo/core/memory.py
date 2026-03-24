from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mnemo.core.chunker import Chunker, ChunkerConfig
from mnemo.core.embedder import SentenceTransformerEmbedder
from mnemo.core.retriever import HybridRetriever
from mnemo.core.store import SQLiteStore


@dataclass(frozen=True)
class MemoryResult:
    text: str
    score: float
    source: str | None
    namespace: str
    created_at: str


class MemoryStore:
    def __init__(
        self,
        *,
        path: str = "./memory.db",
        namespace: str = "default",
        chunk_strategy: str = "sentence",
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        semantic_threshold: float = 0.72,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        self.path = path
        self.namespace = namespace
        self.store = SQLiteStore(path)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.chunker = Chunker(
            ChunkerConfig(
                strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                semantic_threshold=semantic_threshold,
            )
        )
        self.retriever = HybridRetriever()

    def add(self, text: str, *, source: str | None = None) -> int:
        # Semantic chunking needs sentence embeddings to find boundaries.
        if self.chunker.config.strategy == "semantic":
            chunks = self.chunker.chunk(text, embed_fn=self.embedder.embed)
        else:
            chunks = self.chunker.chunk(text)
        if not chunks:
            return 0
        # One embedding per chunk, then persist text + vector together.
        embeddings = self.embedder.embed(chunks)
        return self.store.insert_many(
            namespace=self.namespace,
            texts=chunks,
            embeddings=embeddings,
            source=source,
        )

    def search(self, query: str, *, top_k: int = 5) -> list[MemoryResult]:
        chunks = self.store.fetch_namespace(self.namespace)
        if not chunks:
            return []
        # Embed query once and score it against all stored chunk vectors.
        query_embedding = self.embedder.embed([query])[0]
        hits = self.retriever.search(
            query=query,
            chunks=chunks,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return [
            MemoryResult(
                text=hit.chunk.text,
                score=hit.score,
                source=hit.chunk.source,
                namespace=hit.chunk.namespace,
                created_at=hit.chunk.created_at,
            )
            for hit in hits
        ]

    def context(self, *, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "No relevant memory found."
        lines = ["Relevant memory context:"]
        for idx, row in enumerate(results, start=1):
            lines.append(f"{idx}. {row.text}")
        return "\n".join(lines)

    def clear(self, namespace: str | None = None) -> int:
        target = namespace or self.namespace
        return self.store.clear_namespace(target)

    def stats(self) -> dict[str, object]:
        ns = self.store.list_namespaces()
        return {
            "path": str(Path(self.path).resolve()),
            "db_size_bytes": Path(self.path).stat().st_size if Path(self.path).exists() else 0,
            "namespaces": ns,
            "active_namespace": self.namespace,
            "active_namespace_count": self.store.count_namespace(self.namespace),
        }
