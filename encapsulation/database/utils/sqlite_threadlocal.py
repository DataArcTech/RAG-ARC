"""Thread-local SQLite connection wrapper.

Rationale:
- SQLite connections are not inherently safe to share across threads.
- DeepSearch may execute retrieval in a threadpool for concurrency.
- This wrapper gives each thread its own sqlite3.Connection while preserving the
  `conn.cursor()/commit()/rollback()` call sites used throughout the codebase.
"""
import sqlite3
import threading
from typing import Any, Mapping


class ThreadLocalSQLiteConnection:
    """A thin proxy that creates one sqlite3.Connection per thread."""

    def __init__(
        self,
        db_path: str,
        *,
        timeout: float = 30.0,
        pragmas: Mapping[str, Any] | None = None,
    ) -> None:
        self._db_path = db_path
        self._timeout = float(timeout)
        self._pragmas = dict(pragmas or {})
        self._tls = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=self._timeout)
            conn.execute("PRAGMA foreign_keys=ON")
            for key, value in self._pragmas.items():
                conn.execute(f"PRAGMA {key}={value}")
            setattr(self._tls, "conn", conn)
        return conn

    def cursor(self, *args: Any, **kwargs: Any):
        return self._get_conn().cursor(*args, **kwargs)

    def commit(self) -> None:
        self._get_conn().commit()

    def rollback(self) -> None:
        self._get_conn().rollback()

    def close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            return
        conn.close()
        delattr(self._tls, "conn")

    def __getattr__(self, name: str):
        return getattr(self._get_conn(), name)

