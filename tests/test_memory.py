from pathlib import Path

import numpy as np

from mnemo import MemoryStore


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            low = text.lower()
            if "auth" in low or "jwt" in low:
                vectors.append([1.0, 0.0, 0.0])
            elif "docker" in low or "deploy" in low:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)


def test_add_and_search_semantic_similarity(tmp_path: Path):
    db = tmp_path / "memory.db"
    store = MemoryStore(path=str(db), embedder=FakeEmbedder())
    store.add("Auth uses JWT and short-lived access tokens")
    store.add("Deployment uses Docker and listens on port 8080")
    hits = store.search("how does auth work?", top_k=2)
    assert hits
    assert "Auth uses JWT" in hits[0].text


def test_namespace_isolation(tmp_path: Path):
    db = tmp_path / "memory.db"
    a = MemoryStore(path=str(db), namespace="ns-a", embedder=FakeEmbedder())
    b = MemoryStore(path=str(db), namespace="ns-b", embedder=FakeEmbedder())
    a.add("Only namespace A has this auth note")
    b.add("Only namespace B has deployment note")
    hits_b = b.search("auth note", top_k=3)
    assert all("namespace A" not in row.text for row in hits_b)


def test_persistence_after_restart(tmp_path: Path):
    db = tmp_path / "memory.db"
    first = MemoryStore(path=str(db), embedder=FakeEmbedder())
    first.add("Auth note persists across process restart")
    assert db.exists()
    second = MemoryStore(path=str(db), embedder=FakeEmbedder())
    hits = second.search("auth", top_k=1)
    assert hits
    assert "persists" in hits[0].text
