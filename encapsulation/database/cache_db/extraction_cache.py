"""Batch-friendly Redis cache for extraction results (GraphData etc.).

Design constraints:
- Redis has caused instability before; this cache is deliberately IO-light:
  - bulk_get uses MGET in batches
  - bulk_set uses pipelined SETEX in batches (no per-item roundtrips)
  - values are small (graph-only) and optionally compressed
- Cache keys are tenant+owner scoped to prevent cross-tenant/user leakage.
  - Cache keys are tenant-scoped by default (configurable to owner-scoped).

This module lives in encapsulation/ because it depends on Redis wiring.
"""
import base64
import hashlib
import json
import logging
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.database.cache_db.redis_db import RedisDB
from framework.tenant_context import get_tenant_id

logger = logging.getLogger(__name__)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _encode_value(value: dict, *, compress: bool) -> str:
    raw = _json_dumps(value).encode("utf-8")
    if not compress:
        return raw.decode("utf-8", errors="ignore")
    payload = zlib.compress(raw, level=1)
    return base64.b64encode(payload).decode("ascii")


def _decode_value(payload: str, *, compress: bool) -> Optional[dict]:
    if payload is None:
        return None
    try:
        if not compress:
            obj = json.loads(payload)
            return obj if isinstance(obj, dict) else None
        raw = base64.b64decode(payload.encode("ascii"))
        text = zlib.decompress(raw).decode("utf-8", errors="ignore")
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class ExtractionCacheConfig:
    enabled: bool
    ttl_seconds: int
    namespace: str
    compress: bool
    mget_batch_size: int
    mset_batch_size: int
    key_scope: str  # "tenant" | "owner"


class RedisExtractionCache:
    """Redis-backed extraction cache (MGET + pipelined SETEX)."""

    def __init__(self, cfg: ExtractionCacheConfig):
        self.cfg = cfg
        try:
            self._client = RedisDB(RedisConfig()).client  # singleton; config from env
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction cache disabled (Redis init failed): %s", exc)
            self._client = None

    def is_available(self) -> bool:
        return bool(self.cfg.enabled) and self._client is not None

    def make_key(
        self,
        *,
        owner_id: str,
        extractor_fingerprint: str,
        content_hash: str,
    ) -> str:
        tenant_id = get_tenant_id()
        scope = str(self.cfg.key_scope or "").strip().lower() or "tenant"
        if scope == "owner":
            owner = (str(owner_id or "").strip()) or "unknown_owner"
            return f"{self.cfg.namespace}:{tenant_id}:{owner}:{extractor_fingerprint}:{content_hash}"
        # Default: tenant-scoped (share across users within the same tenant; isolates across tenants).
        return f"{self.cfg.namespace}:{tenant_id}:{extractor_fingerprint}:{content_hash}"

    def bulk_get(self, keys: Iterable[str]) -> Dict[str, dict]:
        if not self.is_available():
            return {}
        keys_list = [str(k) for k in keys if str(k or "").strip()]
        if not keys_list:
            return {}

        out: Dict[str, dict] = {}
        batch = max(1, int(self.cfg.mget_batch_size or 256))
        try:
            for start in range(0, len(keys_list), batch):
                part = keys_list[start : start + batch]
                values = self._client.mget(part)
                for k, v in zip(part, values):
                    if v is None:
                        continue
                    obj = _decode_value(v, compress=self.cfg.compress)
                    if obj is not None:
                        out[k] = obj
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction cache bulk_get failed (treat as miss): %s", exc)
            return {}

        return out

    def bulk_set(self, mapping: Dict[str, dict]) -> None:
        if not self.is_available():
            return
        if not mapping:
            return

        ttl = max(1, int(self.cfg.ttl_seconds or 0))
        batch = max(1, int(self.cfg.mset_batch_size or 128))
        items = [(k, v) for k, v in mapping.items() if str(k or "").strip() and isinstance(v, dict)]
        if not items:
            return

        try:
            pipe = self._client.pipeline(transaction=False)
            buffered = 0
            for k, v in items:
                payload = _encode_value(v, compress=self.cfg.compress)
                pipe.setex(str(k), ttl, payload)
                buffered += 1
                if buffered >= batch:
                    pipe.execute()
                    pipe = self._client.pipeline(transaction=False)
                    buffered = 0
            if buffered:
                pipe.execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction cache bulk_set failed (ignored): %s", exc)


def content_hash_for_chunk_text(text: str) -> str:
    return _sha256_hex(text or "")
