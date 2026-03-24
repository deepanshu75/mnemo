from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from mnemo.exceptions import StoreError


@dataclass(frozen=True)
class StoredChunk:
    id: int
    text: str
    source: str | None
    namespace: str
    created_at: str
    embedding: np.ndarray


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        # WAL improves concurrent read/write behavior for local workloads.
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
            )
            conn.commit()

    @staticmethod
    def _to_blob(vector: np.ndarray) -> bytes:
        # Store embeddings compactly as float32 bytes in SQLite.
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def insert_many(
        self,
        *,
        namespace: str,
        texts: Iterable[str],
        embeddings: np.ndarray,
        source: str | None,
    ) -> int:
        rows = list(texts)
        if not rows:
            return 0
        if len(rows) != len(embeddings):
            raise StoreError("text and embedding count mismatch")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memories(namespace, text, source, created_at, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (namespace, text, source, now, self._to_blob(embeddings[idx]))
                    for idx, text in enumerate(rows)
                ],
            )
            conn.commit()
        return len(rows)

    def fetch_namespace(self, namespace: str) -> list[StoredChunk]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, text, source, namespace, created_at, embedding
                FROM memories
                WHERE namespace = ?
                ORDER BY id ASC
                """,
                (namespace,),
            )
            rows = cursor.fetchall()
        return [
            StoredChunk(
                id=row[0],
                text=row[1],
                source=row[2],
                namespace=row[3],
                created_at=row[4],
                embedding=self._from_blob(row[5]),
            )
            for row in rows
        ]

    def clear_namespace(self, namespace: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
            conn.commit()
            return int(cursor.rowcount)

    def list_namespaces(self) -> list[str]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT DISTINCT namespace FROM memories ORDER BY namespace")
            return [row[0] for row in cursor.fetchall()]

    def count_namespace(self, namespace: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace = ?",
                (namespace,),
            )
            return int(cursor.fetchone()[0])
