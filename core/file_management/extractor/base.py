"""
Base extractor with simple concurrent extraction functionality.
"""

from abc import abstractmethod
from typing import List
import asyncio
import logging
import hashlib
import copy
import json

from encapsulation.data_model.schema import Chunk, GraphData
from framework.module import AbstractModule
from core.file_management.extractor.metadata_keys import EXTRACTION_ERROR_KEY, EXTRACTION_SKIPPED_KEY
from encapsulation.database.cache_db.extraction_cache import (
    ExtractionCacheConfig,
    RedisExtractionCache,
    content_hash_for_chunk_text,
)
from framework.tenant_context import get_tenant_id

logger = logging.getLogger(__name__)

_DEFAULT_ERROR_POLICY = "attach"  # attach | raise | empty


def _coerce_error_policy(value: object) -> str:
    policy = str(value or "").strip().lower()
    if policy in {"attach", "raise", "empty"}:
        return policy
    return _DEFAULT_ERROR_POLICY


def _build_error_payload(*, exc: BaseException, chunk_id: str | None, extractor_name: str, llm_model: str | None) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "extractor": extractor_name,
        "llm_model": llm_model,
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
    }
    return {k: v for k, v in payload.items() if v is not None}


class ExtractorBase(AbstractModule):
    """BaseExtractor: only responsible for single-round extraction and concurrency control"""

    def __init__(self, config):
        super().__init__(config)
        llm_config = getattr(config, "llm_config", None)
        self.llm = llm_config.build() if llm_config is not None else None

    @abstractmethod
    async def extract(self, chunk: Chunk) -> GraphData:
        """extract from a single chunk

        Args:
            chunk: Chunk to extract

        Returns:
            GraphData: Extracted graph data
        """
        pass

    async def process_chunk(self, chunk: Chunk, *, semaphore: asyncio.Semaphore) -> Chunk:
        """process a single chunk"""
        async with semaphore:
            error_policy = _coerce_error_policy(getattr(self.config, "error_policy", _DEFAULT_ERROR_POLICY))
            try:
                graph_data = await self.extract(chunk)
                chunk.graph = graph_data
                return chunk
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}", exc_info=True)
                if error_policy == "raise":
                    raise

                chunk_id = getattr(chunk, "id", None)
                llm_config = getattr(self.config, "llm_config", None)
                llm_model = getattr(llm_config, "model_name", None) if llm_config is not None else None
                payload = _build_error_payload(
                    exc=e,
                    chunk_id=str(chunk_id) if chunk_id else None,
                    extractor_name=self.__class__.__name__,
                    llm_model=str(llm_model) if llm_model else None,
                )

                graph = GraphData()
                if error_policy == "attach":
                    graph.metadata[EXTRACTION_ERROR_KEY] = payload
                    chunk.metadata[EXTRACTION_ERROR_KEY] = payload
                chunk.graph = graph  # empty graph data (with attached error if configured)
                return chunk

    async def _extract_concurrent_impl(self, chunks: List[Chunk]) -> List[Chunk]:
        """Internal concurrent extraction implementation (no dedup logic)."""
        if not chunks:
            return []

        logger.info(f"Starting concurrent extraction with max_concurrent={self.config.max_concurrent}")

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        batch_size = getattr(self.config, "batch_size", None)
        try:
            batch_size_int = int(batch_size) if batch_size is not None else 0
        except (TypeError, ValueError):
            batch_size_int = 0

        # Avoid spawning an unbounded number of tasks for large corpora:
        # when batch_size is unset, default to a window derived from max_concurrent.
        if batch_size is None or batch_size_int <= 0:
            batch_size_int = max(1, int(self.config.max_concurrent) * 4)
        else:
            # Guard: a batch window smaller than max_concurrent will never saturate the semaphore,
            # effectively reducing throughput while still advertising a higher concurrency.
            batch_size_int = max(int(self.config.max_concurrent), batch_size_int)

        # Use a rolling in-flight window to avoid head-of-line blocking: `gather()` on a whole batch
        # waits for the slowest chunk in the batch, which can dramatically reduce throughput for
        # long-tail latency distributions (common with remote LLM APIs).
        results: list[Chunk | None] = [None] * len(chunks)
        inflight: set[asyncio.Task] = set()

        async def _run_one(idx: int, chunk: Chunk) -> tuple[int, Chunk]:
            out_chunk = await self.process_chunk(chunk, semaphore=semaphore)
            return idx, out_chunk

        async def _drain(*, final: bool) -> None:
            nonlocal inflight
            if not inflight:
                return
            # Defensive: pass a stable snapshot to asyncio.wait in case the underlying set is
            # mutated by outer scheduling logic between event loop ticks.
            snapshot = tuple(inflight)
            done, pending = await asyncio.wait(
                snapshot,
                return_when=asyncio.ALL_COMPLETED if final else asyncio.FIRST_COMPLETED,
            )
            inflight = set(pending)
            for task in done:
                idx, out_chunk = task.result()
                results[idx] = out_chunk

        for idx, chunk in enumerate(chunks):
            inflight.add(asyncio.create_task(_run_one(idx, chunk)))
            if len(inflight) >= batch_size_int:
                await _drain(final=False)

        await _drain(final=True)

        # Preserve original order.
        return [item if item is not None else chunks[i] for i, item in enumerate(results)]

    async def _extract_concurrent_dedup_cache(self, chunks: List[Chunk]) -> List[Chunk]:
        """Extraction core: optional dedup + optional Redis cache + concurrent extraction."""
        if not chunks:
            return []

        cache = self._maybe_build_extraction_cache()
        extractor_fp = self._extractor_cache_fingerprint()

        # Optional: de-duplicate identical content within a single batch call.
        # This can dramatically reduce LLM calls for PDFs with repeated headers/footers.
        # Kept opt-in via config to avoid unexpected interactions for custom extractors.
        dedup_by_content = bool(getattr(self.config, "dedup_by_content", False))
        if not dedup_by_content and (cache is None or not cache.is_available()):
            return await self._extract_concurrent_impl(chunks)

        unique: list[Chunk] = []
        order: list[str] = []
        seen: set[str] = set()
        base_chunk_by_hash: dict[str, Chunk] = {}
        owner_id_by_hash: dict[str, str] = {}
        for chunk in chunks:
            content = getattr(chunk, "content", "") or ""
            h = "" if not content else content_hash_for_chunk_text(content)
            if h not in seen:
                seen.add(h)
                order.append(h)
                unique.append(chunk)
                base_chunk_by_hash[h] = chunk
                owner_id_by_hash[h] = str(
                    getattr(chunk, "owner_id", "")
                    or getattr(chunk, "metadata", {}).get("owner_id", "")
                    or ""
                ).strip()

        if len(unique) >= len(chunks) and (cache is None or not cache.is_available()):
            return await self._extract_concurrent_impl(chunks)

        logger.info(
            "Extractor dedup_by_content enabled: %d -> %d unique chunks (saved=%d)",
            len(chunks),
            len(unique),
            len(chunks) - len(unique),
        )

        # ==================== Redis-backed cache prefetch (batch IO) ====================
        key_by_hash: dict[str, str] = {}
        cached_by_hash: dict[str, Chunk] = {}
        if cache is not None and cache.is_available():
            max_unique = None
            try:
                max_unique = int(getattr(self.config, "extraction_cache_max_unique_keys_per_call", 0) or 0)
            except Exception:
                max_unique = None

            if max_unique is not None and max_unique > 0 and len(order) > max_unique:
                # Safety: skip Redis cache IO for extremely large batches.
                cache = None
            else:
                for h in order:
                    owner_id = owner_id_by_hash.get(h) or "unknown_owner"
                    key_by_hash[h] = cache.make_key(
                        owner_id=owner_id,
                        extractor_fingerprint=extractor_fp,
                        content_hash=h,
                    )

                hits = cache.bulk_get(key_by_hash.values())
                if hits:
                    for h in order:
                        ck = key_by_hash.get(h)
                        payload = hits.get(ck) if ck else None
                        if payload is None:
                            continue
                        base = base_chunk_by_hash.get(h)
                        if base is None:
                            continue
                        try:
                            graph_dict = payload.get("graph")
                            if isinstance(graph_dict, dict):
                                base.graph = GraphData.from_dict(graph_dict)
                            base.metadata.setdefault("extraction_cache", {})
                            base.metadata["extraction_cache"].update(
                                {"hit": True, "tenant_id": get_tenant_id(), "extractor_fp": extractor_fp}
                            )
                            cached_by_hash[h] = base
                        except Exception:
                            continue

        # Only extract cache misses.
        to_extract: list[Chunk] = []
        for h in order:
            if h in cached_by_hash:
                continue
            base = base_chunk_by_hash.get(h)
            if base is not None:
                to_extract.append(base)

        extracted_unique: list[Chunk] = []
        if to_extract:
            extracted_unique = await self._extract_concurrent_impl(to_extract)

        # Write misses back to cache (single pipelined burst, chunked).
        if cache is not None and cache.is_available() and extracted_unique:
            to_set: dict[str, dict] = {}
            max_value_bytes = None
            try:
                max_value_bytes = int(getattr(self.config, "extraction_cache_max_value_bytes", 0) or 0)
            except Exception:
                max_value_bytes = None
            for chunk in extracted_unique:
                content = getattr(chunk, "content", "") or ""
                h = "" if not content else content_hash_for_chunk_text(content)
                if not h:
                    continue
                owner_id = str(
                    getattr(chunk, "owner_id", "")
                    or getattr(chunk, "metadata", {}).get("owner_id", "")
                    or ""
                ).strip() or "unknown_owner"
                ck = cache.make_key(owner_id=owner_id, extractor_fingerprint=extractor_fp, content_hash=h)
                graph = getattr(chunk, "graph", None)
                try:
                    graph_dict = graph.to_dict() if graph is not None else GraphData().to_dict()
                except Exception:
                    graph_dict = GraphData().to_dict()
                payload = {"graph": graph_dict}
                if max_value_bytes is not None and max_value_bytes > 0:
                    try:
                        # Limit based on uncompressed JSON size (cheap & conservative).
                        if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")) > max_value_bytes:
                            continue
                    except Exception:
                        pass
                to_set[ck] = payload
            cache.bulk_set(to_set)

        by_key: dict[str, Chunk] = {}
        for h, base_chunk in cached_by_hash.items():
            by_key[h] = base_chunk
        for base_chunk in extracted_unique:
            content = getattr(base_chunk, "content", "") or ""
            h = "" if not content else content_hash_for_chunk_text(content)
            if h:
                by_key[h] = base_chunk

        out: list[Chunk] = []
        for chunk in chunks:
            content = getattr(chunk, "content", "") or ""
            h = "" if not content else content_hash_for_chunk_text(content)
            base = by_key.get(h)
            if base is None or base is chunk:
                out.append(chunk if base is None else base)
                continue
            # Copy graph (and extraction error metadata) without clobbering chunk-level metadata like page numbers.
            try:
                chunk.graph = copy.deepcopy(getattr(base, "graph", GraphData()))
                if EXTRACTION_ERROR_KEY in (getattr(base, "metadata", None) or {}):
                    chunk.metadata[EXTRACTION_ERROR_KEY] = copy.deepcopy(base.metadata.get(EXTRACTION_ERROR_KEY))
                if "extraction_cache" in (getattr(base, "metadata", None) or {}):
                    chunk.metadata["extraction_cache"] = copy.deepcopy(base.metadata.get("extraction_cache"))
            except Exception:
                chunk.graph = getattr(base, "graph", GraphData())
            out.append(chunk)
        return out

    async def extract_concurrent(self, chunks: List[Chunk]) -> List[Chunk]:
        """Extract from multiple chunks concurrently (supports optional chunk-role filtering)."""
        if not chunks:
            return []

        allowed_roles_raw = getattr(self.config, "extract_chunk_roles", None)
        allowed_roles: list[str] = []
        if isinstance(allowed_roles_raw, list):
            allowed_roles = [str(x).strip() for x in allowed_roles_raw if str(x or "").strip()]

        if not allowed_roles:
            return await self._extract_concurrent_dedup_cache(chunks)

        to_extract: list[Chunk] = []
        for chunk in chunks:
            meta = getattr(chunk, "metadata", None) or {}
            chunk.metadata = meta
            role = str(meta.get("chunk_role") or "").strip()
            if role in allowed_roles:
                to_extract.append(chunk)
                continue
            # Keep skipped chunks observable, but do not treat it as an "error".
            chunk.graph = GraphData()
            meta[EXTRACTION_SKIPPED_KEY] = {
                "reason": "chunk_role_filtered",
                "chunk_role": role,
                "allowed_chunk_roles": allowed_roles,
            }

        if not to_extract:
            return chunks

        extracted = await self._extract_concurrent_dedup_cache(to_extract)
        by_id = {str(getattr(c, "id", "") or ""): c for c in extracted if str(getattr(c, "id", "") or "").strip()}
        out: list[Chunk] = []
        for chunk in chunks:
            key = str(getattr(chunk, "id", "") or "")
            out.append(by_id.get(key, chunk))
        return out

    async def __call__(self, chunks: List[Chunk]) -> List[Chunk]:
        return await self.extract_concurrent(chunks)

    def _extractor_cache_fingerprint(self) -> str:
        """Stable fingerprint for cache key derivation (override in subclasses if needed)."""
        fp = None
        try:
            getter = getattr(self, "cache_fingerprint", None)
            if callable(getter):
                fp = str(getter()).strip() or None
        except Exception:
            fp = None
        if fp:
            return fp
        llm_cfg = getattr(self.config, "llm_config", None)
        model_name = str(getattr(llm_cfg, "model_name", "") or "").strip() if llm_cfg is not None else ""
        base = f"{self.__class__.__name__}:{model_name}"
        return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _maybe_build_extraction_cache(self) -> RedisExtractionCache | None:
        enabled = bool(getattr(self.config, "extraction_cache_enabled", False))
        if not enabled:
            return None
        try:
            cfg = ExtractionCacheConfig(
                enabled=enabled,
                ttl_seconds=int(getattr(self.config, "extraction_cache_ttl_seconds", 30 * 24 * 3600) or 0),
                namespace=str(getattr(self.config, "extraction_cache_namespace", "ragarc:extract") or "ragarc:extract").strip(),
                compress=bool(getattr(self.config, "extraction_cache_compress", True)),
                mget_batch_size=int(getattr(self.config, "extraction_cache_mget_batch_size", 256) or 256),
                mset_batch_size=int(getattr(self.config, "extraction_cache_mset_batch_size", 128) or 128),
                key_scope=str(getattr(self.config, "extraction_cache_key_scope", "tenant") or "tenant").strip().lower(),
            )
        except Exception:
            return None
        return RedisExtractionCache(cfg)
