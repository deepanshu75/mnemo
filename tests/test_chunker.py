import numpy as np

from mnemo.core.chunker import Chunker, ChunkerConfig


def test_fixed_chunker_non_empty():
    chunker = Chunker(ChunkerConfig(strategy="fixed", chunk_size=4, overlap=1))
    chunks = chunker.chunk("one two three four five six seven eight")
    assert chunks
    assert all(chunks)


def test_sentence_chunker_groups_sentences():
    chunker = Chunker(ChunkerConfig(strategy="sentence", chunk_size=6))
    chunks = chunker.chunk("One short sentence. Two tiny sentence. Three short sentence.")
    assert len(chunks) >= 2
    assert all(chunks)


def test_semantic_chunker_no_empty_chunks():
    chunker = Chunker(ChunkerConfig(strategy="semantic", chunk_size=10, semantic_threshold=0.8))

    def embed_fn(parts):
        return np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ][: len(parts)],
            dtype=np.float32,
        )

    text = "Auth uses JWT. Tokens expire quickly. Deployment uses Docker."
    chunks = chunker.chunk(text, embed_fn=embed_fn)
    assert chunks
    assert all(chunks)
