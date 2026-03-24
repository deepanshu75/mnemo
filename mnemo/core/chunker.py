from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkerConfig:
    strategy: str = "sentence"
    chunk_size: int = 220
    overlap: int = 40
    semantic_threshold: float = 0.72


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences if sentences else [text.strip()]


def fixed_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for idx in range(0, len(words), step):
        part = words[idx : idx + chunk_size]
        if part:
            chunks.append(" ".join(part).strip())
    return chunks


def sentence_chunks(text: str, chunk_size: int) -> list[str]:
    pieces = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in pieces:
        sentence_len = max(1, len(sentence.split()))
        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def semantic_chunks(
    text: str,
    *,
    embed_fn,
    chunk_size: int,
    threshold: float,
) -> list[str]:
    pieces = split_sentences(text)
    if len(pieces) <= 1:
        return [text.strip()] if text.strip() else []
    vectors = np.asarray(embed_fn(pieces), dtype=np.float32)
    chunks: list[str] = []
    current = [pieces[0]]
    current_len = len(pieces[0].split())
    for idx in range(1, len(pieces)):
        similarity = float(np.dot(vectors[idx - 1], vectors[idx]))
        piece = pieces[idx]
        piece_len = len(piece.split())
        boundary = similarity < threshold or current_len + piece_len > chunk_size
        if boundary:
            chunks.append(" ".join(current).strip())
            current = [piece]
            current_len = piece_len
        else:
            current.append(piece)
            current_len += piece_len
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


class Chunker:
    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config

    def chunk(self, text: str, *, embed_fn=None) -> list[str]:
        clean = text.strip()
        if not clean:
            return []
        if self.config.strategy == "fixed":
            return fixed_chunks(clean, self.config.chunk_size, self.config.overlap)
        if self.config.strategy == "sentence":
            return sentence_chunks(clean, self.config.chunk_size)
        if self.config.strategy == "semantic":
            if embed_fn is None:
                raise ValueError("semantic strategy requires embed_fn")
            return semantic_chunks(
                clean,
                embed_fn=embed_fn,
                chunk_size=self.config.chunk_size,
                threshold=self.config.semantic_threshold,
            )
        raise ValueError(f"unknown chunker strategy: {self.config.strategy}")
