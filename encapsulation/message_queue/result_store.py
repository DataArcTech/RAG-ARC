import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from encapsulation.io.io_manager import IOManager
from framework.virtual_paths import IO_PATH_PREFIX, io_key, is_io_path


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_name(value: str, *, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    cleaned = _SAFE_NAME_RE.sub("_", text)
    return cleaned or fallback


class ResultStoreError(RuntimeError):
    pass


class ResultStore:
    def put_bytes(self, *, namespace: str, run_id: str, payload: bytes, ttl_seconds: int) -> str:
        raise NotImplementedError

    def get_bytes(self, ref: str) -> Optional[bytes]:
        raise NotImplementedError

    def delete(self, ref: str) -> None:
        return

    def put_json(self, *, namespace: str, run_id: str, payload: Dict[str, Any], ttl_seconds: int) -> str:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        return self.put_bytes(namespace=namespace, run_id=run_id, payload=data, ttl_seconds=ttl_seconds)

    def get_json(self, ref: str) -> Optional[Dict[str, Any]]:
        data = self.get_bytes(ref)
        if not data:
            return None
        try:
            parsed = json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ResultStoreError(f"Invalid JSON payload in result store ref={ref!r}") from exc
        return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True)
class IOManagerResultStoreSettings:
    io_manager: IOManager
    base_dir: str


def _split_virtual_dir(base_dir: str) -> tuple[str, str]:
    token = str(base_dir or "").strip()
    if not is_io_path(token):
        raise ResultStoreError(f"ResultStore base_dir must be an io:// virtual dir, got: {base_dir!r}")
    key = io_key(token)
    parts = [p for p in key.split("/") if p]
    if not parts:
        raise ResultStoreError(f"ResultStore base_dir is empty: {base_dir!r}")
    namespace = parts[0]
    prefix = "/".join(parts[1:])
    return namespace, prefix


class IOManagerResultStore(ResultStore):
    def __init__(self, settings: IOManagerResultStoreSettings):
        self._settings = settings
        self._namespace, self._prefix = _split_virtual_dir(settings.base_dir)

    def put_bytes(self, *, namespace: str, run_id: str, payload: bytes, ttl_seconds: int) -> str:  # noqa: ARG002
        ns = _sanitize_name(namespace, fallback="default")
        rid = _sanitize_name(run_id, fallback="run")
        suffix = f"{ns}/{rid}.json"
        key = "/".join([p for p in [self._prefix, suffix] if p])
        result = self._settings.io_manager.put_bytes(
            namespace=self._namespace,
            key=key,
            payload=payload,
            content_type="application/json",
        )
        return result.ref

    def get_bytes(self, ref: str) -> Optional[bytes]:
        if not isinstance(ref, str) or not ref.strip():
            return None
        text = ref.strip()
        if not text.startswith(IO_PATH_PREFIX):
            raise ResultStoreError(f"Unsupported result ref (expected io://...): {ref!r}")
        return self._settings.io_manager.get_bytes(text)

    def delete(self, ref: str) -> None:
        try:
            text = str(ref or "").strip()
            if not text.startswith(IO_PATH_PREFIX):
                return
            self._settings.io_manager.delete(text)
        except Exception:
            return


class MinioResultStore(ResultStore):
    def __init__(self, *, endpoint: str | None, bucket: str | None):  # noqa: ARG002
        # TODO(minio): Implement MinIO-backed ResultStore using the `minio` SDK, including secure credentials handling,
        # bucket/key naming strategy, and TTL/lifecycle integration.
        pass

    def put_bytes(self, *, namespace: str, run_id: str, payload: bytes, ttl_seconds: int) -> str:  # noqa: ARG002
        raise NotImplementedError("TODO(minio): MinIO result store is not implemented")

    def get_bytes(self, ref: str) -> Optional[bytes]:
        raise NotImplementedError("TODO(minio): MinIO result store is not implemented")


def build_result_store(*, backend: str, local_dir: str, minio_endpoint: str | None, minio_bucket: str | None) -> ResultStore:
    normalized = (backend or "").strip().lower() or "local"
    if normalized in {"local", "io"}:
        raise ResultStoreError("IOManagerResultStore requires an explicit io_manager (call build_result_store_with_io_manager)")
    if normalized == "minio":
        return MinioResultStore(endpoint=minio_endpoint, bucket=minio_bucket)
    raise ResultStoreError(f"Unknown MQ result store backend: {backend!r}")


def build_result_store_with_io_manager(
    *,
    backend: str,
    local_dir: str,
    minio_endpoint: str | None,
    minio_bucket: str | None,
    io_manager: IOManager,
) -> ResultStore:
    normalized = (backend or "").strip().lower() or "io"
    if normalized in {"local", "io"}:
        base_dir = str(local_dir or "").strip() or "io://mq_results"
        if not base_dir.startswith(IO_PATH_PREFIX):
            raise ResultStoreError("MQ result store directory must be an io:// virtual path")
        return IOManagerResultStore(IOManagerResultStoreSettings(io_manager=io_manager, base_dir=base_dir))
    if normalized == "minio":
        return MinioResultStore(endpoint=minio_endpoint, bucket=minio_bucket)
    raise ResultStoreError(f"Unknown MQ result store backend: {backend!r}")
