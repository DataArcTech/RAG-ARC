from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

from .base import FileDB


def _normalize_prefix(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    text = text.strip("/").replace("..", "")
    return text


def _normalize_key(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    text = text.replace("..", "")
    return text.lstrip("/")


@dataclass
class PrefixedFileDB(FileDB):
    """A transparent key-prefix wrapper around another FileDB.

    Keeps caller-facing keys stable while grouping objects physically under a
    prefix in the backend (e.g., S3/MinIO object keys).
    """

    inner: FileDB
    key_prefix: str

    def __post_init__(self) -> None:
        self.key_prefix = _normalize_prefix(self.key_prefix)

    def _full_key(self, key: str) -> str:
        normalized = _normalize_key(key)
        if not self.key_prefix:
            return normalized
        if not normalized:
            return self.key_prefix
        return f"{self.key_prefix}/{normalized}"

    def _strip_prefix(self, storage_key: str) -> str:
        normalized = _normalize_key(storage_key)
        if not self.key_prefix:
            return normalized
        prefix = f"{self.key_prefix}/"
        if normalized == self.key_prefix:
            return ""
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
        return normalized

    def store(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, bool]:
        storage_key, overwritten = self.inner.store(self._full_key(key), data, content_type=content_type, **kwargs)
        return self._strip_prefix(storage_key), bool(overwritten)

    def retrieve(self, key: str, **kwargs: Any) -> bytes:
        return self.inner.retrieve(self._full_key(key), **kwargs)

    def delete(self, key: str, **kwargs: Any) -> bool:
        return bool(self.inner.delete(self._full_key(key), **kwargs))

    def exists(self, key: str, **kwargs: Any) -> bool:
        return bool(self.inner.exists(self._full_key(key), **kwargs))

    def list_keys(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        full_prefix = self._full_key(prefix or "") if prefix else (self.key_prefix or None)
        keys = self.inner.list_keys(prefix=full_prefix, limit=limit, **kwargs)
        stripped = [self._strip_prefix(k) for k in keys]
        return [k for k in stripped if k]

    def generate_presigned_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
        **kwargs: Any,
    ) -> str:
        return self.inner.generate_presigned_url(
            self._full_key(key),
            expiration_seconds=expiration_seconds,
            method=method,
            **kwargs,
        )

