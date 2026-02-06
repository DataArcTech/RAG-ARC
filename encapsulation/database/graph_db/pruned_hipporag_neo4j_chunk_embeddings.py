import logging
import os
import pickle
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jChunkEmbeddingsMixin:
    @staticmethod
    def _chunk_owner_leaf(owner_id: Optional[str]) -> str:
        token = str(owner_id or "").strip() or "__GLOBAL__"
        return token.replace("/", "_").replace("\\", "_") or "__GLOBAL__"

    def _chunk_embeddings_sharded_enabled(self) -> bool:
        cfg = getattr(self, "config", None)
        return bool(getattr(cfg, "chunk_embeddings_owner_sharded", False))

    def _chunk_embeddings_dir(self, *, base_path: Optional[str] = None) -> str:
        cfg = getattr(self, "config", None)
        dirname = str(getattr(cfg, "chunk_embeddings_dirname", "chunk_embeddings") or "chunk_embeddings").strip()
        root = str(base_path or getattr(self, "storage_path", "") or "").strip()
        return os.path.join(root, dirname)

    def _chunk_embeddings_shard_path(self, *, owner_id: Optional[str], base_path: Optional[str] = None) -> str:
        base = self._chunk_embeddings_dir(base_path=base_path)
        leaf = self._chunk_owner_leaf(owner_id)
        name = str(getattr(self, "index_name", "index") or "index")
        return os.path.join(base, f"{name}_chunk_embeddings__{leaf}.pkl")

    def _load_chunk_embeddings(self):
        """
        Load chunk embeddings from disk if available.

        This is called during initialization to restore chunk embeddings
        that were saved during previous sessions.
        """
        # New layout: owner-scoped shards (when enabled).
        if self._chunk_embeddings_sharded_enabled():
            try:
                base = self._chunk_embeddings_dir()
                loaded_total: Dict[str, Any] = {}
                owner_by_chunk_id: Dict[str, Optional[str]] = {}
                chunk_ids_by_owner: Dict[Optional[str], set[str]] = {}
                if os.path.isdir(base):
                    prefix = f"{self.index_name}_chunk_embeddings__"
                    for fname in sorted(os.listdir(base)):
                        if not (fname.startswith(prefix) and fname.endswith(".pkl")):
                            continue
                        leaf = fname[len(prefix) : -len(".pkl")]
                        owner = None if leaf == "__GLOBAL__" else leaf
                        fpath = os.path.join(base, fname)
                        with open(fpath, "rb") as f:
                            shard = pickle.load(f)
                        if not isinstance(shard, dict):
                            continue
                        for cid, emb in shard.items():
                            cid_str = str(cid)
                            loaded_total[cid_str] = emb
                            owner_by_chunk_id[cid_str] = owner
                            chunk_ids_by_owner.setdefault(owner, set()).add(cid_str)

                if loaded_total:
                    with self.write_lock():
                        self.chunk_embeddings = loaded_total
                        self._chunk_embeddings_array = None
                        setattr(self, "_chunk_embedding_owner_by_chunk_id", owner_by_chunk_id)
                        setattr(self, "_chunk_ids_by_owner", chunk_ids_by_owner)
                        setattr(self, "_chunk_embeddings_dirty_owners", set())
                    logger.info("Loaded %s chunk embeddings from sharded dir %s", len(loaded_total), base)
                    return
            except Exception as e:
                logger.warning("Failed to load sharded chunk embeddings: %s", e)

        # Legacy layout: single shared pickle.
        embeddings_path = os.path.join(self.storage_path, f"{self.index_name}_chunk_embeddings.pkl")
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'rb') as f:
                    loaded = pickle.load(f)
                expected_dim = None
                cfg = getattr(getattr(self, "embedding_model", None), "config", None)
                candidate = getattr(cfg, "embedding_dimensions", None)
                if candidate is not None:
                    try:
                        expected_dim = int(candidate)
                    except Exception:
                        expected_dim = None
                if expected_dim is None and isinstance(loaded, dict) and loaded:
                    for value in loaded.values():
                        if value is None:
                            continue
                        try:
                            arr = np.array(value, dtype=np.float32)
                        except Exception:
                            continue
                        if arr.ndim == 1 and arr.shape[0] > 0:
                            expected_dim = int(arr.shape[0])
                            break

                filtered = {}
                dropped = 0
                if isinstance(loaded, dict) and expected_dim is not None and expected_dim > 0:
                    for chunk_id, emb in loaded.items():
                        if emb is None:
                            continue
                        try:
                            arr = np.array(emb, dtype=np.float32)
                            if arr.ndim != 1 or arr.shape[0] != expected_dim:
                                dropped += 1
                                continue
                            filtered[chunk_id] = arr
                        except Exception:
                            dropped += 1
                else:
                    filtered = loaded if isinstance(loaded, dict) else {}

                with self.write_lock():
                    self.chunk_embeddings = filtered
                    # Mark array for rebuild on first use
                    self._chunk_embeddings_array = None
                logger.info(f"Loaded {len(self.chunk_embeddings)} chunk embeddings from {embeddings_path}")
                if dropped:
                    logger.warning(
                        "Dropped %s chunk embeddings due to dimension/type mismatch (expected_dim=%s).",
                        dropped,
                        expected_dim,
                    )
            except Exception as e:
                logger.warning(f"Failed to load chunk embeddings: {e}")
        else:
            logger.info(f"No existing chunk embeddings found at {embeddings_path}")

    def _save_chunk_embeddings(self, *, base_path: str, name: str = "index") -> None:
        """Persist chunk embeddings (sharded by owner when enabled)."""
        if not self._chunk_embeddings_sharded_enabled():
            embeddings_path = os.path.join(base_path, f"{name}_chunk_embeddings.pkl")
            with open(embeddings_path, "wb") as f:
                pickle.dump(self.chunk_embeddings, f)
            logger.info("Saved chunk embeddings to %s", embeddings_path)
            return

        base = self._chunk_embeddings_dir(base_path=base_path)
        os.makedirs(base, exist_ok=True)

        dirty: set[Optional[str]] = getattr(self, "_chunk_embeddings_dirty_owners", None) or set()
        by_owner: Dict[Optional[str], set[str]] = getattr(self, "_chunk_ids_by_owner", None) or {}

        owners = dirty or set(by_owner.keys())
        if not owners:
            return

        for owner in owners:
            shard_path = self._chunk_embeddings_shard_path(owner_id=owner, base_path=base_path)
            ids = by_owner.get(owner) or set()
            payload = {}
            for cid in ids:
                if cid in self.chunk_embeddings:
                    payload[cid] = self.chunk_embeddings[cid]
            tmp_path = f"{shard_path}.tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, shard_path)

        try:
            dirty.clear()
        except Exception:
            pass

    def _append_chunk_embeddings(self, new_chunk_ids: List[str]):
        """
        Incrementally append new chunk embeddings to the array (OPTIMIZED).

        This method only processes new chunks instead of rebuilding the entire array,
        which is much faster for incremental updates.

        Args:
            new_chunk_ids: List of new chunk IDs to append
        """
        if not new_chunk_ids:
            logger.info("No new chunks to append")
            return

        import time

        start_time = time.time()

        logger.info(f"Appending {len(new_chunk_ids)} new chunk embeddings...")

        new_embeddings_list = []
        new_chunk_ids_ordered = []

        with self.read_lock():
            for cid in sorted(new_chunk_ids):  # Sort for consistency
                if cid not in self.chunk_embeddings:
                    logger.warning(f"Chunk {cid} not found in chunk_embeddings, skipping")
                    continue

                emb = self.chunk_embeddings[cid]

                # Ensure it's a numpy array
                if isinstance(emb, list):
                    emb = np.array(emb)
                elif not isinstance(emb, np.ndarray):
                    logger.warning(f"Chunk {cid} has invalid embedding type: {type(emb)}, skipping")
                    continue

                # Check shape consistency with existing array
                if self._chunk_embeddings_array is not None and len(self._chunk_embeddings_array) > 0:
                    expected_shape = (self._chunk_embeddings_array.shape[1],)
                    if emb.shape != expected_shape:
                        logger.warning(f"Chunk {cid} has shape {emb.shape}, expected {expected_shape}, skipping")
                        continue

                # Normalize if enabled (for cosine similarity)
                if self.normalize_chunk_embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                new_embeddings_list.append(emb)
                new_chunk_ids_ordered.append(cid)

        if new_embeddings_list:
            # Convert to array with optional float16
            if self.use_float16_embeddings:
                new_array = np.array(new_embeddings_list, dtype=np.float16)
            else:
                new_array = np.array(new_embeddings_list, dtype=np.float32)

            # Append to existing array or create new one
            with self.write_lock():
                if self._chunk_embeddings_array is not None and len(self._chunk_embeddings_array) > 0:
                    self._chunk_embeddings_array = np.vstack([self._chunk_embeddings_array, new_array])
                else:
                    self._chunk_embeddings_array = new_array

                # Update chunk IDs list
                if hasattr(self, '_chunk_ids_list') and self._chunk_ids_list:
                    self._chunk_ids_list.extend(new_chunk_ids_ordered)
                else:
                    self._chunk_ids_list = new_chunk_ids_ordered

            logger.info(
                f"Appended {len(new_chunk_ids_ordered)} embeddings in {time.time() - start_time:.2f}s"
            )
        else:
            logger.warning("No valid new embeddings to append")

    def _rebuild_chunk_embeddings_array(self):
        """
        Build the chunk embeddings array for fast dense retrieval.

        This method:
        - Converts the chunk_embeddings dict to a numpy array
        - Optionally uses float16 for memory efficiency
        - Normalizes embeddings for cosine similarity (if enabled)
        """
        with self.read_lock():
            if self._chunk_embeddings_array is not None:
                return  # Already built
            chunk_ids = list(self.chunk_embeddings.keys())

        logger.info("Rebuilding chunk embeddings array...")

        kept_chunk_ids: list[str] = []
        embeddings_list: list[np.ndarray] = []

        with self.read_lock():
            for i, cid in enumerate(chunk_ids):
                emb = self.chunk_embeddings.get(cid)
                if emb is None:
                    continue
                if isinstance(emb, list):
                    emb = np.array(emb)
                elif not isinstance(emb, np.ndarray):
                    logger.warning(f"Chunk {cid} has invalid embedding type: {type(emb)}")
                    continue

                if embeddings_list and emb.shape != embeddings_list[0].shape:
                    logger.error(f"Chunk {cid} (index {i}) has shape {emb.shape}, expected {embeddings_list[0].shape}")
                    logger.error(
                        f"  First chunk ID: {kept_chunk_ids[0] if kept_chunk_ids else 'N/A'}, "
                        f"shape: {embeddings_list[0].shape}"
                    )
                    logger.error(f"  Current chunk ID: {cid}, shape: {emb.shape}")
                    continue

                if self.normalize_chunk_embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                kept_chunk_ids.append(cid)
                embeddings_list.append(emb)

        if not embeddings_list:
            new_array = np.array([])
            logger.warning("No chunk embeddings found")
        else:
            try:
                if self.use_float16_embeddings:
                    new_array = np.array(embeddings_list, dtype=np.float16)
                else:
                    new_array = np.array(embeddings_list, dtype=np.float32)
            except ValueError as e:
                logger.error(f"Failed to build chunk embeddings array: {e}")
                logger.error(f"  Total chunks: {len(chunk_ids)}")
                logger.error(f"  Valid embeddings: {len(embeddings_list)}")
                if embeddings_list:
                    logger.error(f"  First embedding shape: {embeddings_list[0].shape}")
                    logger.error(f"  Last embedding shape: {embeddings_list[-1].shape}")
                raise

        with self.write_lock():
            if self._chunk_embeddings_array is not None:
                return
            self._chunk_ids_list = kept_chunk_ids
            self._chunk_embeddings_array = new_array

        if len(new_array) > 0 and isinstance(new_array, np.ndarray):
            dtype_str = "float16" if self.use_float16_embeddings else "float32"
            logger.info(
                f"Chunk embeddings array built ({dtype_str}): {len(kept_chunk_ids)} chunks, "
                f"memory: {new_array.nbytes / 1024 / 1024:.2f} MB"
            )
