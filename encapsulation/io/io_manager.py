import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from encapsulation.database.file_db.base import FileDB
from encapsulation.io.io_refs import from_io_ref, to_io_ref
from framework.module import AbstractModule
from framework.virtual_paths import is_io_path, resolve_io_to_local_path

logger = logging.getLogger(__name__)


def _normalize_key(token: str) -> str:
    """Normalize a storage key (no traversal, POSIX separators, no leading slash)."""

    text = str(token or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    text = text.replace("..", "")
    return text.lstrip("/").strip()


def _join_key(parts: Sequence[str]) -> str:
    items = [_normalize_key(p) for p in parts if str(p or "").strip()]
    items = [p for p in items if p]
    return "/".join(items)


@dataclass(frozen=True)
class IOManagerPutResult:
    ref: str
    key: str
    overwritten: bool


class IOManager(AbstractModule):
    """Infrastructure I/O manager backed by a FileDB (LocalDB first; other backends later)."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self._blob_store: FileDB = config.file_db_config.build()
        self._default_namespace = str(getattr(config, "default_namespace", "io") or "io").strip() or "io"

    @property
    def blob_store(self) -> FileDB:
        return self._blob_store

    def resolve_local_path(self, io_path: str, *, ensure_parent: bool = False) -> Path:
        """Resolve an `io://...` virtual path into a local filesystem path.

        Phase 1: maps to LocalDB root (`IO_STORE_BASE_PATH`, default `./data/localdb`).
        """

        if not is_io_path(io_path):
            raise ValueError(f"IOManager.resolve_local_path expects io://..., got: {io_path!r}")
        path = resolve_io_to_local_path(io_path)
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_local_dir(self, io_dir: str, *, ensure: bool = True) -> Path:
        """Resolve an `io://...` virtual directory into a local filesystem directory path."""

        path = self.resolve_local_path(io_dir, ensure_parent=False)
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _full_key(self, *, namespace: Optional[str], key: str) -> str:
        ns = str(namespace or self._default_namespace).strip() or self._default_namespace
        return _join_key([ns, key])

    def put_bytes(
        self,
        *,
        key: str,
        payload: bytes,
        namespace: Optional[str] = None,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> IOManagerPutResult:
        full_key = self._full_key(namespace=namespace, key=key)
        storage_key, was_overwritten = self._blob_store.store(full_key, payload, content_type=content_type, **kwargs)
        return IOManagerPutResult(ref=to_io_ref(storage_key), key=str(storage_key), overwritten=bool(was_overwritten))

    def get_bytes(self, ref_or_key: str, **kwargs: Any) -> Optional[bytes]:
        key = _normalize_key(from_io_ref(ref_or_key))
        if not key:
            return None
        try:
            return self._blob_store.retrieve(key, **kwargs)
        except KeyError:
            return None

    def put_text(
        self,
        *,
        key: str,
        text: str,
        namespace: Optional[str] = None,
        encoding: str = "utf-8",
        content_type: str = "text/plain; charset=utf-8",
        **kwargs: Any,
    ) -> IOManagerPutResult:
        payload = (text or "").encode(encoding)
        return self.put_bytes(key=key, payload=payload, namespace=namespace, content_type=content_type, **kwargs)

    def get_text(
        self,
        ref_or_key: str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        **kwargs: Any,
    ) -> Optional[str]:
        data = self.get_bytes(ref_or_key, **kwargs)
        if data is None:
            return None
        try:
            return data.decode(encoding, errors=errors)
        except Exception:
            return data.decode("utf-8", errors="replace")

    def put_json(
        self,
        *,
        key: str,
        payload: Dict[str, Any],
        namespace: Optional[str] = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> IOManagerPutResult:
        raw = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return self.put_text(
            key=key,
            text=raw,
            namespace=namespace,
            encoding=encoding,
            content_type="application/json; charset=utf-8",
            **kwargs,
        )

    def get_json(self, ref_or_key: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        text = self.get_text(ref_or_key, **kwargs)
        if text is None:
            return None
        try:
            parsed = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IOManager.get_json: invalid JSON for ref=%r: %s", ref_or_key, exc)
            return None
        return parsed if isinstance(parsed, dict) else None

    def exists(self, ref_or_key: str, **kwargs: Any) -> bool:
        key = _normalize_key(from_io_ref(ref_or_key))
        if not key:
            return False
        try:
            return bool(self._blob_store.exists(key, **kwargs))
        except Exception:
            return False

    def delete(self, ref_or_key: str, **kwargs: Any) -> bool:
        key = _normalize_key(from_io_ref(ref_or_key))
        if not key:
            return False
        try:
            return bool(self._blob_store.delete(key, **kwargs))
        except Exception:
            return False

    def list_keys(self, *, namespace: Optional[str] = None, prefix: Optional[str] = None, limit: Optional[int] = None) -> list[str]:
        ns = str(namespace or self._default_namespace).strip() or self._default_namespace
        full_prefix = _join_key([ns, str(prefix or "").strip()]) if prefix else _normalize_key(ns)
        keys = self._blob_store.list_keys(prefix=full_prefix, limit=limit)
        return [to_io_ref(k) for k in keys]

    def iter_bytes(
        self,
        refs_or_keys: Iterable[str],
        **kwargs: Any,
    ) -> list[tuple[str, Optional[bytes]]]:
        out: list[tuple[str, Optional[bytes]]] = []
        for item in refs_or_keys:
            out.append((str(item), self.get_bytes(str(item), **kwargs)))
        return out
