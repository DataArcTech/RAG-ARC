"""
Owner-scoped sqlite KV cache for embeddings.

Why sqlite (not Redis)
- Redis IO spikes previously caused instability in local/docker environments.
- sqlite keeps IO local and predictable; WAL supports concurrent reads + serialized writes.

This cache is intentionally "dumb":
- No cross-owner sharing (caller must pass an owner-scoped path).
- Keys are pre-hashed (caller controls the scheme and includes model fingerprint).
"""
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass(frozen=True)
class SqliteEmbeddingCacheStats:
    hits: int
    misses: int
    sets: int


class SqliteEmbeddingCache:
    """
    A tiny sqlite cache mapping `key -> float32 embedding bytes`.

    Table schema:
        k TEXT PRIMARY KEY
        v BLOB NOT NULL  (float32 bytes)
        dim INTEGER NOT NULL
    """

    def __init__(self, *, db_path: str, max_in_keys_per_query: int = 500):
        self.db_path = str(db_path)
        self.max_in_keys_per_query = int(max(1, max_in_keys_per_query))
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        _ensure_parent_dir(self.db_path)
        conn = sqlite3.connect(self.db_path, timeout=60.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "k TEXT PRIMARY KEY,"
            "v BLOB NOT NULL,"
            "dim INTEGER NOT NULL"
            ");"
        )
        conn.commit()
        self._conn = conn
        return conn

    def stats(self) -> SqliteEmbeddingCacheStats:
        with self._lock:
            return SqliteEmbeddingCacheStats(hits=int(self._hits), misses=int(self._misses), sets=int(self._sets))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def get_many(self, keys: List[str]) -> Dict[str, Tuple[bytes, int]]:
        """
        Return mapping: key -> (embedding_bytes, dim).
        """
        clean = [str(k) for k in (keys or []) if str(k or "").strip()]
        if not clean:
            return {}

        out: Dict[str, Tuple[bytes, int]] = {}
        with self._lock:
            conn = self._connect()
            for i in range(0, len(clean), self.max_in_keys_per_query):
                batch = clean[i : i + self.max_in_keys_per_query]
                placeholders = ",".join(["?"] * len(batch))
                rows = conn.execute(
                    f"SELECT k, v, dim FROM embedding_cache WHERE k IN ({placeholders})",
                    batch,
                ).fetchall()
                for k, v, dim in rows:
                    if k is None or v is None or dim is None:
                        continue
                    out[str(k)] = (bytes(v), int(dim))

            # Track hit/miss stats by keys requested.
            hits = len(out)
            misses = len(clean) - hits
            self._hits += hits
            self._misses += max(0, misses)

        return out

    def set_many(self, items: Iterable[Tuple[str, bytes, int]]) -> None:
        rows = [(str(k), sqlite3.Binary(v), int(dim)) for (k, v, dim) in items if str(k or "").strip() and v]
        if not rows:
            return
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (k, v, dim) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
            self._sets += len(rows)

