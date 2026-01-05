import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


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
class LocalFileResultStoreSettings:
    base_dir: Path


class LocalFileResultStore(ResultStore):
    def __init__(self, settings: LocalFileResultStoreSettings):
        self._settings = settings

    def put_bytes(self, *, namespace: str, run_id: str, payload: bytes, ttl_seconds: int) -> str:  # noqa: ARG002
        ns = _sanitize_name(namespace, fallback="default")
        rid = _sanitize_name(run_id, fallback="run")
        base_dir = self._settings.base_dir
        target_dir = base_dir / ns
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{rid}.json"
        tmp_path = target_dir / f".{rid}.{os.getpid()}.tmp"
        tmp_path.write_bytes(payload)
        tmp_path.replace(target_path)
        rel = f"{ns}/{rid}.json"
        return f"local://{rel}"

    def get_bytes(self, ref: str) -> Optional[bytes]:
        if not isinstance(ref, str) or not ref.strip():
            return None
        text = ref.strip()
        if not text.startswith("local://"):
            raise ResultStoreError(f"Unsupported local result ref: {ref!r}")
        rel = text[len("local://") :].lstrip("/")
        base_dir = self._settings.base_dir.resolve()
        target_path = (base_dir / rel).resolve()
        if base_dir not in target_path.parents and target_path != base_dir:
            raise ResultStoreError(f"Unsafe local result ref path traversal: {ref!r}")
        if not target_path.exists():
            return None
        return target_path.read_bytes()

    def delete(self, ref: str) -> None:
        try:
            text = (ref or "").strip()
            if not text.startswith("local://"):
                return
            rel = text[len("local://") :].lstrip("/")
            base_dir = self._settings.base_dir.resolve()
            target_path = (base_dir / rel).resolve()
            if base_dir not in target_path.parents and target_path != base_dir:
                return
            if target_path.exists():
                target_path.unlink(missing_ok=True)
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
    if normalized == "local":
        base_dir = Path(local_dir or "local/mq_results").expanduser()
        return LocalFileResultStore(LocalFileResultStoreSettings(base_dir=base_dir))
    if normalized == "minio":
        return MinioResultStore(endpoint=minio_endpoint, bucket=minio_bucket)
    raise ResultStoreError(f"Unknown MQ result store backend: {backend!r}")

