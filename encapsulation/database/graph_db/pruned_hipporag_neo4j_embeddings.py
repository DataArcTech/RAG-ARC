import json
import os
import logging
import re
import random
import threading
import time
import copy
from typing import List, Dict, Any, Optional, Sequence, Set, Tuple

import numpy as np

import faiss

from encapsulation.data_model.schema import Chunk
from core.utils.path_guard import safe_leaf_name

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jEmbeddingsMixin:
    @staticmethod
    def _faiss_distance_to_similarity(*, distance: float, metric: str, index_type: str | None) -> float:
        """
        Convert a FAISS `search()` distance into a comparable similarity score.

        Notes:
        - For `Index*IP` (used by metric='cosine' with normalized vectors, and metric='ip'), FAISS returns
          the inner product where larger is better (already a similarity).
        - For `IndexHNSWFlat` with `METRIC_INNER_PRODUCT` (our cosine HNSW indices), FAISS returns
          `-||u-v||^2` where vectors are L2-normalized. For normalized vectors, `||u-v||^2 = 2 - 2*cos`,
          so we recover cosine similarity via: `cos = 1 + distance/2`.
        - For `Index*L2`, FAISS returns an L2 distance where smaller is better; we convert to a score by negating.
        """
        m = str(metric or "").strip().lower()
        it = str(index_type or "").strip().lower()
        if m in {"cosine", "ip"}:
            if it == "hnsw":
                # distance ~= -||u-v||^2 for normalized vectors (see docstring).
                return 1.0 + (float(distance) / 2.0)
            return float(distance)  # flat/ivf inner-product indices return IP directly
        return -float(distance)

    def _owner_scoped_faiss_lock(self) -> threading.Lock:
        lock = getattr(self, "_owner_scoped_faiss_lock_obj", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_owner_scoped_faiss_lock_obj", lock)
        return lock

    def _owner_scoped_faiss_missing_warned(self) -> set[str]:
        warned = getattr(self, "_owner_scoped_faiss_missing_warned_keys", None)
        if warned is None:
            warned = set()
            setattr(self, "_owner_scoped_faiss_missing_warned_keys", warned)
        return warned

    @staticmethod
    def _faiss_index_artifacts_ready(index_dir: str) -> bool:
        return os.path.exists(os.path.join(index_dir, "index.pkl")) and os.path.exists(os.path.join(index_dir, "index.faiss"))

    def _faiss_owner_scoped_dir(self, kind: str, *, owner_id: Optional[str]) -> str:
        """
        Resolve an owner-scoped on-disk directory for graph FAISS artifacts.

        Layout:
          <storage_path>/{fact_index,entity_index}/<owner_leaf>/

        Notes:
        - This avoids cross-run pollution when benchmark runs use distinct owners.
        - `owner_id=None` is stored under `__GLOBAL__` (not a multi-owner aggregate).
        """
        base_dir = os.path.join(str(self.storage_path), f"{kind}_index")
        owner_leaf = safe_leaf_name(owner_id or "__GLOBAL__", default="__GLOBAL__")
        return os.path.join(base_dir, owner_leaf)

    @staticmethod
    def _clone_faiss_config_for_path(config: Any, *, index_path: str) -> Any:
        """
        Create a per-path clone of a FAISS config so @shared_module does not reuse instances.
        """
        if hasattr(config, "model_copy"):
            try:
                return config.model_copy(update={"index_path": index_path})
            except Exception:  # noqa: BLE001
                pass
        cloned = copy.copy(config)
        try:
            setattr(cloned, "index_path", str(index_path))
        except Exception:  # noqa: BLE001
            # As a last resort, mutate in place (still isolates if caller passes unique config objects).
            setattr(config, "index_path", str(index_path))
            return config
        return cloned

    def _owner_scoped_faiss_db(self, kind: str, *, owner_id: Optional[str]):
        """
        Return a FaissVectorDB bound to the owner-scoped directory (lazy-load from disk).
        """
        # Backward compatibility: legacy deployments stored a single shared FAISS index directly under
        # `<storage_path>/{fact_index,entity_index}/index.{faiss,pkl}` (without owner subdirectories).
        # Newer retrieval paths may request owner-scoped FAISS DBs; when owner-scoped artifacts are
        # missing but legacy artifacts exist, fall back to the shared index to avoid silently disabling
        # fact/entity retrieval for normal owner scopes.
        template_db = self.fact_faiss_db if kind == "fact" else self.entity_faiss_db
        base_dir = os.path.join(str(self.storage_path), f"{kind}_index")
        owner_dir = self._faiss_owner_scoped_dir(kind, owner_id=owner_id)
        if not self._faiss_index_artifacts_ready(owner_dir) and self._faiss_index_artifacts_ready(base_dir):
            return template_db

        cache_attr = "_owner_scoped_fact_faiss" if kind == "fact" else "_owner_scoped_entity_faiss"
        lock = self._owner_scoped_faiss_lock()
        with lock:
            cache: Dict[str, Any] = getattr(self, cache_attr, None) or {}
            setattr(self, cache_attr, cache)

        key = str(owner_id) if owner_id is not None else "__GLOBAL__"
        with lock:
            cached = cache.get(key)
        if cached is not None:
            db = cached
        else:
            index_dir = self._faiss_owner_scoped_dir(kind, owner_id=owner_id)
            cfg = self._clone_faiss_config_for_path(getattr(template_db, "config", template_db), index_path=index_dir)

            from encapsulation.database.vector_db.faiss import FaissVectorDB

            db = FaissVectorDB(cfg)

            with lock:
                cache[key] = db

        index_dir = getattr(getattr(db, "config", None), "index_path", None) or self._faiss_owner_scoped_dir(kind, owner_id=owner_id)

        # Important: index artifacts can appear after process startup (e.g. ingestion writes index files).
        # If we cache an empty DB before both files are present, we must retry loading on later calls.
        with lock:
            should_attempt_load = getattr(db, "index", None) is None

            if should_attempt_load and self._faiss_index_artifacts_ready(index_dir):
                try:
                    db.load_index(index_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load %s FAISS index for owner=%s at %s: %s", kind, owner_id, index_dir, exc)
            elif should_attempt_load:
                warned_key = f"{kind}:{key}"
                warned = self._owner_scoped_faiss_missing_warned()
                should_warn = warned_key not in warned
                if should_warn:
                    warned.add(warned_key)
                    logger.warning(
                        "%s FAISS index artifacts not found for owner=%s under %s (expected index.faiss + index.pkl)",
                        kind,
                        owner_id,
                        index_dir,
                    )

        return db

    def get_fact_faiss_db(self, owner_id: Optional[object]):
        owner_str = None if owner_id is None else str(owner_id)
        owner_norm = self._restore_owner_id(owner_str)
        return self._owner_scoped_faiss_db("fact", owner_id=owner_norm)

    def get_entity_faiss_db(self, owner_id: Optional[object]):
        owner_str = None if owner_id is None else str(owner_id)
        owner_norm = self._restore_owner_id(owner_str)
        return self._owner_scoped_faiss_db("entity", owner_id=owner_norm)

    def iter_owner_scoped_faiss_dbs(self, kind: str):
        """
        Iterate over all owner-scoped FAISS DBs present on disk for this store.

        This is primarily used for admin/global retrieval where owner_id=None.
        """
        base_dir = os.path.join(str(self.storage_path), f"{kind}_index")
        if not os.path.isdir(base_dir):
            return []
        out = []
        for name in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full) or name.startswith("."):
                continue
            owner = None if name == "__GLOBAL__" else name
            out.append((owner, self._owner_scoped_faiss_db(kind, owner_id=owner)))
        return out

    def _graph_embedding_settings(self) -> Dict[str, float]:
        config_batch_size = getattr(getattr(self.embedding_model, "config", None), "request_batch_size", None)
        batch_size = int(os.getenv("GRAPH_INDEX_EMBED_BATCH_SIZE", str(config_batch_size or 64)))
        max_retries = int(os.getenv("GRAPH_INDEX_EMBED_MAX_RETRIES", "6"))
        backoff_base = float(os.getenv("GRAPH_INDEX_EMBED_BACKOFF_BASE_SECONDS", "1.0"))
        backoff_cap = float(os.getenv("GRAPH_INDEX_EMBED_BACKOFF_MAX_SECONDS", "30.0"))
        sleep_between_batches = float(os.getenv("GRAPH_INDEX_EMBED_SLEEP_BETWEEN_BATCHES_SECONDS", "0.0"))
        return {
            "batch_size": float(max(1, batch_size)),
            "max_retries": float(max(0, max_retries)),
            "backoff_base": float(max(0.0, backoff_base)),
            "backoff_cap": float(max(0.1, backoff_cap)),
            "sleep_between_batches": float(max(0.0, sleep_between_batches)),
        }

    @staticmethod
    def _looks_like_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "429" in msg
            or "rate limit" in msg
            or "too many requests" in msg
            or "ratelimit" in msg
        )

    def _embedding_dimension_for_placeholders(self, *, fallback_from: Sequence[Sequence[float] | None] | None = None) -> int:
        """
        Resolve embedding dimension for placeholder (zero) vectors.

        We avoid hardcoding a dimension. Resolution order:
        1) infer from any successful embeddings in `fallback_from` (when present)
        2) embedding_model.config.embedding_dimensions (if present)
        3) embedding_model._embedding_dimension_cache (if set by prior probes)
        4) embedding_model.get_embedding_dimension() (may issue a single probe call)

        If dimension cannot be resolved, we raise with guidance to set EMBEDDING_DIMENSIONS.
        """
        if fallback_from:
            for vec in fallback_from:
                if vec is None:
                    continue
                try:
                    dim = len(vec)  # type: ignore[arg-type]
                    if dim > 0:
                        return int(dim)
                except Exception:
                    continue

        cfg = getattr(getattr(self, "embedding_model", None), "config", None)
        candidate = getattr(cfg, "embedding_dimensions", None)
        if candidate is not None:
            try:
                dim = int(candidate)
                if dim > 0:
                    return dim
            except Exception:
                pass

        cached = getattr(getattr(self, "embedding_model", None), "_embedding_dimension_cache", None)
        if cached is not None:
            try:
                dim = int(cached)
                if dim > 0:
                    return dim
            except Exception:
                pass

        get_dim = getattr(getattr(self, "embedding_model", None), "get_embedding_dimension", None)
        if callable(get_dim):
            dim = int(get_dim())
            if dim > 0:
                return dim

        raise RuntimeError(
            "Failed to resolve embedding dimension for placeholder vectors. "
            "Set EMBEDDING_DIMENSIONS (or ensure embedding_model.get_embedding_dimension() works)."
        )

    def _embed_texts_resilient(self, texts: List[str], *, purpose: str) -> List[List[float]]:
        """
        Resilient embedding wrapper for graph indexing.

        - Batches inputs to avoid 429 amplification
        - Retries on rate limit errors with jittered backoff
        - If a batch fails, splits down to per-item to isolate bad inputs / backend quirks
        """
        from config.graph_index_embedding import (
            GRAPH_INDEX_EMBED_FAILURE_POLICY,
            GRAPH_INDEX_EMBED_MAX_FAILURE_COUNT,
            GRAPH_INDEX_EMBED_MAX_FAILURE_RATIO,
        )

        settings = self._graph_embedding_settings()
        batch_size = int(settings["batch_size"])
        max_retries = int(settings["max_retries"])
        backoff_base = float(settings["backoff_base"])
        backoff_cap = float(settings["backoff_cap"])
        sleep_between_batches = float(settings["sleep_between_batches"])

        if not texts:
            return []

        # Keep output aligned with input to avoid partial-index corruption when a single item fails.
        results: list[list[float] | None] = [None] * len(texts)
        failures: List[Dict[str, Any]] = []

        # Normalize and pre-filter empties (they are treated as per-item failures in tolerant modes).
        normalized_items: list[tuple[int, str]] = []
        for idx, raw in enumerate(texts):
            if not isinstance(raw, str):
                raise TypeError(f"{purpose} embedding input must be str (index={idx}, type={type(raw).__name__})")
            stripped = raw.strip()
            if not stripped:
                failures.append({"purpose": purpose, "error": f"empty text (index={idx})"})
                continue
            normalized_items.append((idx, stripped.replace("\n", " ")))

        def _embed_batch(batch: List[str]) -> List[List[float]]:
            out = self.embedding_model.embed(batch)
            if isinstance(out, list) and out and isinstance(out[0], (int, float)):
                return [out]  # type: ignore[list-item]
            return out  # type: ignore[return-value]

        # Nothing to embed (all empty) -> handled by policy below.
        if not normalized_items:
            dim = self._embedding_dimension_for_placeholders(fallback_from=results)
            zero = [0.0] * dim
            if GRAPH_INDEX_EMBED_FAILURE_POLICY == "raise":
                raise ValueError(f"Graph index embedding failed ({purpose}): all inputs were empty")
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = zero
            logger.warning(
                "Graph index embedding filled %s empty inputs with zeros (purpose=%s).",
                len(failures),
                purpose,
            )
            return [vec for vec in results if vec is not None]  # type: ignore[return-value]

        for start in range(0, len(normalized_items), batch_size):
            batch_items = normalized_items[start:start + batch_size]
            batch = [text for _idx, text in batch_items]
            attempt = 0
            while True:
                try:
                    batch_embeddings = _embed_batch(batch)
                    if len(batch_embeddings) != len(batch):
                        raise RuntimeError(
                            f"{purpose} embeddings size mismatch: got {len(batch_embeddings)} for {len(batch)} inputs"
                        )
                    for (orig_idx, _text), emb in zip(batch_items, batch_embeddings):
                        results[orig_idx] = emb
                    break
                except Exception as exc:  # noqa: BLE001
                    if self._looks_like_rate_limit(exc) and attempt < max_retries:
                        delay = min(backoff_cap, backoff_base * (2 ** attempt)) * (0.8 + 0.4 * random.random())
                        logger.warning(
                            "Graph index embedding rate-limited (%s). Retrying in %.2fs (attempt %s/%s, batch=%s)",
                            purpose,
                            delay,
                            attempt + 1,
                            max_retries,
                            len(batch),
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue

                    # If a batch fails, split to isolate bad inputs / incompatible endpoints.
                    if len(batch) > 1:
                        logger.warning(
                            "Graph index embedding batch failed (%s, batch=%s). Splitting to per-item. error=%s",
                            purpose,
                            len(batch),
                            str(exc),
                        )
                        for (orig_idx, item) in batch_items:
                            try:
                                item_embedding = _embed_batch([item])
                                if len(item_embedding) != 1:
                                    raise RuntimeError(
                                        f"{purpose} embeddings size mismatch: got {len(item_embedding)} for 1 input"
                                    )
                                results[orig_idx] = item_embedding[0]
                            except Exception as item_exc:  # noqa: BLE001
                                failures.append({"purpose": purpose, "error": str(item_exc), "index": orig_idx})
                        break

                    # Single-item batch failure.
                    orig_idx = batch_items[0][0] if batch_items else None
                    failures.append({"purpose": purpose, "error": str(exc), "index": orig_idx})
                    break

            if sleep_between_batches > 0:
                time.sleep(sleep_between_batches)

        if failures:
            failure_count = len(failures)
            failure_ratio = float(failure_count) / float(max(1, len(texts)))
            too_many = (
                (GRAPH_INDEX_EMBED_MAX_FAILURE_COUNT > 0 and failure_count >= GRAPH_INDEX_EMBED_MAX_FAILURE_COUNT)
                or (GRAPH_INDEX_EMBED_MAX_FAILURE_RATIO > 0 and failure_ratio >= GRAPH_INDEX_EMBED_MAX_FAILURE_RATIO)
            )

            if GRAPH_INDEX_EMBED_FAILURE_POLICY == "raise" or too_many:
                raise RuntimeError(
                    f"Graph index embedding failed ({purpose}): failures={failure_count}/{len(texts)} "
                    f"ratio={failure_ratio:.3f} first_error={failures[0].get('error')}"
                )

            dim = self._embedding_dimension_for_placeholders(fallback_from=results)
            zero = [0.0] * dim
            filled = 0
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = zero
                    filled += 1
            logger.warning(
                "Graph index embedding had %s/%s failures (ratio=%.3f) for purpose=%s; policy=%s filled_with_zeros=%s first_error=%s",
                failure_count,
                len(texts),
                failure_ratio,
                purpose,
                GRAPH_INDEX_EMBED_FAILURE_POLICY,
                filled,
                failures[0].get("error"),
            )

        # `results` is fully filled (either real embeddings or placeholder zeros).
        return [vec for vec in results if vec is not None]  # type: ignore[return-value]

    def batch_generate_embeddings(
        self,
        *,
        chunk_ids: Optional[Sequence[str]] = None,
        entity_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch generate embeddings for all facts, entities, and chunks.

        This method processes embeddings in the following order:
        1. Chunk embeddings: Generated and stored in memory (not in FAISS)
        2. Entity embeddings: Generated and added to FAISS HNSW index
        3. Fact embeddings: Generated and added to FAISS Flat index

        Only new items (not already in FAISS/memory) are processed.
        """
        logger.info("Batch generating embeddings...")
        summary: Dict[str, Any] = {"chunks": {}, "entities": {}, "facts": {}}

        # 1. Generate chunk embeddings
        chunk_params: Dict[str, Any] = {}
        if chunk_ids:
            chunk_query = (
                "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids "
                "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata, c.source_file_id AS source_file_id"
            )
            chunk_params = {"chunk_ids": list(chunk_ids)}
        else:
            chunk_query = (
                "MATCH (c:Chunk) "
                "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata, c.source_file_id AS source_file_id"
            )
        chunks_data = self._execute_query(chunk_query, chunk_params or None)

        new_chunks = []
        new_chunk_ids = []
        for record in chunks_data:
            chunk_id = record['chunk_id']
            content = record.get('content')
            raw_meta = record.get("metadata")
            source_file_id = record.get("source_file_id")
            if chunk_id not in self.chunk_embeddings:
                # Prefer index_text/prompt_text for embedding (same as retrieval),
                # and optionally prefix file/path context for disambiguation.
                meta: Dict[str, Any] = {}
                if raw_meta:
                    try:
                        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
                    except Exception:
                        meta = {}
                if source_file_id and "source_file_id" not in meta:
                    meta["source_file_id"] = source_file_id

                base_text = meta.get("index_text") or meta.get("prompt_text") or content or ""
                try:
                    from encapsulation.database.utils.embedding_text import build_embedding_text, build_prefix_keys

                    cfg = getattr(self, "config", None)
                    prefix_keys = build_prefix_keys(getattr(cfg, "chunk_embedding_text_prefix_keys", None))
                    filename_root = getattr(cfg, "chunk_embedding_filename_root", None)
                    sep = getattr(cfg, "chunk_embedding_text_separator", "\n")
                    embed_text = build_embedding_text(
                        base_text=str(base_text),
                        metadata=meta,
                        prefix_keys=prefix_keys,
                        filename_root=filename_root,
                        separator=sep,
                    )
                except Exception:
                    embed_text = str(base_text)

                new_chunks.append(embed_text)
                new_chunk_ids.append(chunk_id)

        if new_chunks:
            logger.info(f"Batch generating embeddings for {len(new_chunks)} chunks...")
            chunk_embeddings = self._embed_texts_resilient(new_chunks, purpose="chunk")

            # Store embeddings
            for chunk_id, embedding in zip(new_chunk_ids, chunk_embeddings):
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                # Normalize for cosine similarity
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                self.chunk_embeddings[chunk_id] = embedding

            # Mark array needs rebuild
            self._chunk_embeddings_array = None
            logger.info(f"Chunk embeddings generated for {len(new_chunks)} chunks")
            summary["chunks"] = {"attempted": len(new_chunks), "embedded": len(new_chunk_ids)}
        else:
            summary["chunks"] = {"attempted": 0, "embedded": 0}

        # 2. Generate entity embeddings and add to FAISS HNSW
        entity_params: Dict[str, Any] = {}
        if entity_ids:
            entity_query = (
                "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids "
                "RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id"
            )
            entity_params = {"entity_ids": list(entity_ids)}
        else:
            entity_query = "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id"
        entities = self._execute_query(entity_query, entity_params or None)

        owners_to_entities: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for record in entities:
            owner = self._restore_owner_id(record.get("owner_id"))
            owners_to_entities.setdefault(owner, []).append(record)

        attempted_entities = 0
        embedded_entities = 0
        for owner, records in owners_to_entities.items():
            entity_db = self._owner_scoped_faiss_db("entity", owner_id=owner)
            new_entities: List[Chunk] = []
            for record in records:
                entity_id = record["entity_id"]
                entity_name = record["entity_name"]
                if entity_id in entity_db.docstore:
                    continue
                metadata = {"type": "entity"}
                if owner:
                    metadata["owner_id"] = owner
                new_entities.append(Chunk(id=entity_id, content=entity_name, owner_id=owner, metadata=metadata))

            attempted_entities += len(new_entities)
            if not new_entities:
                continue

            logger.info("Adding %s entities to FAISS HNSW (owner=%s)...", len(new_entities), owner)
            entity_texts = [chunk.content for chunk in new_entities]
            entity_embeddings = self._embed_texts_resilient(entity_texts, purpose="entity")

            for chunk, embedding in zip(new_entities, entity_embeddings):
                chunk.metadata["embedding"] = np.array(embedding, dtype=np.float32)

            entity_db.update_index(new_entities)
            entity_index_path = self._faiss_owner_scoped_dir("entity", owner_id=owner)
            entity_db.save_index(entity_index_path, "index")
            embedded_entities += len(new_entities)
            logger.info("Saved owner-scoped entity index to %s", entity_index_path)

        summary["entities"] = {"attempted": attempted_entities, "embedded": embedded_entities}

        # 3. Generate fact embeddings and add to FAISS Flat
        # Facts are stored as RELATES_TO relationships between entities
        fact_query = """
        MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
        RETURN r.fact_id AS fact_id,
               r.text AS text,
               r.predicate AS predicate,
               r.owner_id AS owner_id,
               r.source_chunk_ids AS source_chunk_ids,
               r.source_chunk_ids_truncated AS source_chunk_ids_truncated,
               h.entity_id AS head_id,
               t.entity_id AS tail_id,
               h.entity_name AS head_name,
               t.entity_name AS tail_name,
               h.entity_type AS head_type,
               t.entity_type AS tail_type
        """
        facts = self._execute_query(fact_query)

        owners_to_facts: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for record in facts:
            owner = self._restore_owner_id(record.get("owner_id"))
            owners_to_facts.setdefault(owner, []).append(record)

        attempted_facts = 0
        embedded_facts = 0
        updated_fact_meta_total = 0

        for owner, records in owners_to_facts.items():
            fact_db = self._owner_scoped_faiss_db("fact", owner_id=owner)
            new_facts: List[Chunk] = []
            updated_fact_meta = 0

            for record in records:
                fact_id = record["fact_id"]
                fact_text = record["text"]

                if fact_id not in fact_db.docstore:
                    metadata = {"type": "fact"}
                    if owner:
                        metadata["owner_id"] = owner
                    for key in (
                        "head_id",
                        "tail_id",
                        "head_name",
                        "tail_name",
                        "head_type",
                        "tail_type",
                        "predicate",
                    ):
                        if record.get(key) is not None:
                            metadata[key] = record.get(key)
                    from encapsulation.database.utils.fact_provenance import merge_provenance_into_fact_metadata

                    merge_provenance_into_fact_metadata(
                        metadata,
                        source_chunk_ids=record.get("source_chunk_ids"),
                        source_chunk_ids_truncated=record.get("source_chunk_ids_truncated"),
                    )
                    new_facts.append(Chunk(id=fact_id, content=fact_text, owner_id=owner, metadata=metadata))
                else:
                    chunk = fact_db.docstore.get(fact_id)
                    if not chunk:
                        continue
                    meta = getattr(chunk, "metadata", None)
                    if not isinstance(meta, dict):
                        continue
                    from encapsulation.database.utils.fact_provenance import merge_provenance_into_fact_metadata

                    if merge_provenance_into_fact_metadata(
                        meta,
                        source_chunk_ids=record.get("source_chunk_ids"),
                        source_chunk_ids_truncated=record.get("source_chunk_ids_truncated"),
                    ):
                        updated_fact_meta += 1

            attempted_facts += len(new_facts)

            if new_facts:
                logger.info("Adding %s facts to FAISS Flat (owner=%s)...", len(new_facts), owner)
                fact_texts = [chunk.content for chunk in new_facts]
                fact_embeddings = self._embed_texts_resilient(fact_texts, purpose="fact")
                for chunk, embedding in zip(new_facts, fact_embeddings):
                    chunk.metadata["embedding"] = np.array(embedding, dtype=np.float32)
                fact_db.update_index(new_facts)
                fact_index_path = self._faiss_owner_scoped_dir("fact", owner_id=owner)
                fact_db.save_index(fact_index_path, "index")
                embedded_facts += len(new_facts)
                logger.info("Saved owner-scoped fact index to %s", fact_index_path)

            if updated_fact_meta > 0:
                fact_index_path = self._faiss_owner_scoped_dir("fact", owner_id=owner)
                fact_db.save_index(fact_index_path, "index")
                logger.info(
                    "Backfilled fact provenance for %s existing facts (owner=%s); saved index to %s",
                    updated_fact_meta,
                    owner,
                    fact_index_path,
                )

            updated_fact_meta_total += updated_fact_meta

        summary["facts"] = {"attempted": attempted_facts, "embedded": embedded_facts, "updated_meta": updated_fact_meta_total}

        logger.info("Batch embedding generation completed!")
        return summary

    def _add_synonymy_edges(self, new_entity_ids: Optional[List[str]] = None, *, owner_id: Optional[object] = None):
        """
        Add synonymy edges between similar entities using FAISS HNSW.

        This method:
        1. Retrieves entities from Neo4j (all or only new ones for incremental update)
        2. Filters out short entities (<=2 alphanumeric characters)
        3. Performs batch FAISS search to find top-k similar entities
        4. Filters results by similarity threshold
        5. Excludes entity pairs already connected by facts
        6. Stores synonymy edges in Neo4j as SIMILAR_TO relationships

        Args:
            new_entity_ids: Optional list of new entity IDs for incremental update.
                          If None, processes all entities (full rebuild).
            owner_id: Optional owner scope. When set, only entities/edges for that owner are processed.
        """
        if not self.add_synonymy_edges:
            logger.info("Synonymy edges disabled")
            return

        from tqdm import tqdm

        owner_str = None if owner_id is None else str(owner_id)
        # In this store, node/edge owner_id is stored as the real owner UUID string (or __GLOBAL__).
        owner_filter = self._restore_owner_id(owner_str)

        # IMPORTANT: synonym edges must be computed against the correct owner-scoped entity FAISS index.
        # The store maintains per-owner entity indices on disk; using the template/global DB will silently
        # skip most entities. When owner is provided, force-load that owner-scoped DB.
        entity_db = self.entity_faiss_db
        if owner_filter is not None:
            entity_db = self.get_entity_faiss_db(owner_filter)

        # Determine if this is incremental or full rebuild
        if new_entity_ids:
            logger.info(f"Computing synonymy edges for {len(new_entity_ids)} new entities (incremental)...")
            # Get only new entities
            entity_query = """
            MATCH (e:Entity)
            WHERE e.entity_id IN $entity_ids
            """ + ("AND e.owner_id = $owner_id" if owner_filter is not None else "") + """
            RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id
            """
            params = {'entity_ids': new_entity_ids}
            if owner_filter is not None:
                params["owner_id"] = owner_filter
            entities = self._execute_query(entity_query, params)
        else:
            logger.info("Computing synonymy edges for all entities%s (full rebuild)...", f" owner={owner_filter}" if owner_filter else "")
            # Get all entities
            entity_query = (
                "MATCH (e:Entity) "
                + ("WHERE e.owner_id = $owner_id " if owner_filter is not None else "")
                + "RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id"
            )
            params = {"owner_id": owner_filter} if owner_filter is not None else None
            entities = self._execute_query(entity_query, params) if params is not None else self._execute_query(entity_query)

        if not entities:
            logger.warning("No entities found, skipping synonymy edge addition")
            return

        # Build entity metadata mapping for fast lookup
        entity_id_to_info = {
            record['entity_id']: {
                'name': record['entity_name'],
                'owner_id': self._restore_owner_id(record.get('owner_id'))
            }
            for record in entities
        }

        # Build a set to track existing entity-entity edges (fact edges only)
        existing_entity_entity_edges = set()

        # Get existing RELATES_TO edges (owner-scoped + optional incremental restriction).
        relation_query = """
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        WHERE 1=1
        """ + ("AND r.owner_id = $owner_id AND e1.owner_id = $owner_id AND e2.owner_id = $owner_id\n" if owner_filter is not None else "") + (
            "AND (e1.entity_id IN $entity_ids OR e2.entity_id IN $entity_ids)\n" if new_entity_ids else ""
        ) + """
        RETURN e1.entity_id AS head_id, e2.entity_id AS tail_id
        """
        rel_params: dict[str, Any] = {}
        if owner_filter is not None:
            rel_params["owner_id"] = owner_filter
        if new_entity_ids:
            rel_params["entity_ids"] = new_entity_ids
        relations = self._execute_query(relation_query, rel_params) if rel_params else self._execute_query(relation_query)

        for record in relations:
            head_id = record['head_id']
            tail_id = record['tail_id']
            existing_entity_entity_edges.add((head_id, tail_id))
            existing_entity_entity_edges.add((tail_id, head_id))

        logger.info(f"Built existing entity-entity edge set with {len(existing_entity_entity_edges)} directional edges")

        num_synonym_edges = 0
        edges_to_add = []  # Batch collect edges for Neo4j

        # Pre-extract and normalize all embeddings for batch search
        logger.info("Preparing embeddings for batch FAISS search...")
        valid_entities = []
        embeddings_list = []

        for record in entities:
            entity_id = record['entity_id']
            entity_name = record['entity_name']
            entity_owner = entity_id_to_info.get(entity_id, {}).get('owner_id')
            owner_key = self._owner_key(entity_owner)

            # Filter extremely short entities. Use the shared text normalization so this works for
            # non-Latin scripts (e.g., CJK) and avoids accidentally dropping all Chinese entities.
            from core.utils.text_processing import text_processing

            normalized = text_processing(entity_name)
            try:
                min_chars = int(getattr(self, "synonymy_edge_min_entity_chars", 3) or 3)
            except Exception:  # noqa: BLE001
                min_chars = 3
            if len(normalized.replace(" ", "")) < max(1, min_chars):
                continue

            # Get entity from FAISS docstore
            entity_chunk = entity_db.docstore.get(entity_id)
            if not entity_chunk:
                continue

            # Get embedding from docstore
            embedding = entity_chunk.metadata.get('embedding')
            if embedding is None:
                continue  # Skip if no embedding

            if isinstance(embedding, list):
                embedding = np.array(embedding).astype(np.float32)
            else:
                embedding = embedding.astype(np.float32)

            valid_entities.append((entity_id, entity_name, entity_owner, owner_key))
            embeddings_list.append(embedding)

        if not valid_entities:
            logger.warning("No valid entities for synonymy edge computation")
            return

        # Batch normalize embeddings
        embeddings_array = np.array(embeddings_list).astype(np.float32)
        if entity_db.config.normalize_L2 or entity_db.config.metric == "cosine":
            from core.utils.faiss_lock import FAISS_LOCK
            with FAISS_LOCK:
                faiss.normalize_L2(embeddings_array)

        logger.info(f"Prepared {len(valid_entities)} valid entities for synonymy edge computation")

        # Batch FAISS search
        logger.info("Performing batch FAISS search...")
        k = min(self.synonymy_edge_topk, entity_db.index.ntotal)
        from core.utils.faiss_lock import FAISS_LOCK
        with FAISS_LOCK:
            distances_batch, indices_batch = entity_db.index.search(embeddings_array, k)
        logger.info("Batch FAISS search completed")

        # Process results
        logger.info("Processing search results...")

        for i, ((entity_id, entity_name, entity_owner, owner_key), distances, indices) in enumerate(tqdm(
            zip(valid_entities, distances_batch, indices_batch),
            total=len(valid_entities),
            desc="Computing synonymy edges"
        )):
            # Log progress every 1000 entities
            if i > 0 and i % 1000 == 0:
                logger.info(f"Processed {i}/{len(valid_entities)} entities, found {num_synonym_edges} synonymy edges so far")

            num_added = 0
            for idx, distance in zip(indices, distances):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue

                # Get neighbor entity ID from index
                if idx not in entity_db.index_to_docstore_id:
                    continue

                neighbor_entity_id = entity_db.index_to_docstore_id[idx]

                # Skip deleted entities
                if neighbor_entity_id in entity_db.deleted_ids:
                    continue

                # Skip self
                if neighbor_entity_id == entity_id:
                    continue

                # Get neighbor name for validation (from cache, not Neo4j)
                neighbor_info = entity_id_to_info.get(neighbor_entity_id)
                if not neighbor_info:
                    continue

                neighbor_owner = neighbor_info.get('owner_id')
                if neighbor_owner != entity_owner:
                    continue

                neighbor_name = neighbor_info['name']

                similarity = self._faiss_distance_to_similarity(
                    distance=float(distance),
                    metric=str(entity_db.config.metric or ""),
                    index_type=getattr(entity_db, "saved_index_type", getattr(getattr(entity_db, "config", None), "index_type", None)),
                )

                # Check threshold
                if similarity < self.synonymy_edge_sim_threshold:
                    break  # Distances are sorted, can break early

                edge_key = (entity_id, neighbor_entity_id)
                reverse_edge_key = (neighbor_entity_id, entity_id)

                if edge_key not in existing_entity_entity_edges and reverse_edge_key not in existing_entity_entity_edges:
                    # Add UNIDIRECTIONAL edge (only one direction to avoid duplication)
                    edges_to_add.append((entity_id, neighbor_entity_id, similarity, owner_key))
                    num_synonym_edges += 1
                    num_added += 1

                    # Mark BOTH directions as added to avoid duplicates
                    existing_entity_entity_edges.add(edge_key)
                    existing_entity_entity_edges.add(reverse_edge_key)

                if num_added >= 100:
                    break

        # Batch insert all edges to Neo4j
        if edges_to_add:
            logger.info(f"Saving {len(edges_to_add)} directional synonymy edges to Neo4j...")

            # Use UNWIND for batch insertion
            batch_query = """
            UNWIND $edges AS edge
            MATCH (e1:Entity {entity_id: edge.entity_id_1})
            MATCH (e2:Entity {entity_id: edge.entity_id_2})
            MERGE (e1)-[r:SIMILAR_TO]-(e2)
            SET r.similarity = edge.similarity,
                r.owner_id = edge.owner_id,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """

            # Prepare batch data
            batch_data = [
                {
                    'entity_id_1': e1,
                    'entity_id_2': e2,
                    'similarity': sim,
                    'owner_id': owner_id
                }
                for e1, e2, sim, owner_id in edges_to_add
            ]

            self._execute_query(batch_query, {'edges': batch_data})

            logger.info(f"Added {num_synonym_edges} unique synonymy edges ({len(edges_to_add)} directional edges)")

            # Keep in-memory neighbor cache consistent for long-running services.
            # Without this, newly added SIMILAR_TO edges would only take effect after a restart.
            try:
                load_cache = getattr(self, "_load_graph_cache", None)
                if callable(load_cache):
                    load_cache()
                bump = getattr(self, "write_lock", None)
                if callable(bump):
                    with bump():
                        self._cache_version += 1  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to refresh graph cache after adding synonymy edges: %s", exc)
        else:
            logger.info("No synonymy edges to add")
