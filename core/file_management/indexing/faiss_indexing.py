import asyncio
import logging
import os
import threading
from typing import Any, List, TYPE_CHECKING, Dict, Optional

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig

logger = logging.getLogger(__name__)


class FaissIndexer(BaseIndexer):
    """
    Concrete implementation for indexing chunks using FaissVectorDB.
    """

    def __init__(self, config: "FaissIndexerConfig"):
        """
        Initializes the Faiss indexer and its specific database instance.
        """
        super().__init__(config)
        self._template_cfg = config.index_config
        self._owner_scoped_enabled = bool(getattr(self._template_cfg, "owner_scoped_enabled", False))
        self._db_by_owner: Dict[str | None, object] = {}
        self._db_lock = threading.Lock()

        if not self._owner_scoped_enabled:
            self.faiss_db = config.index_config.build()
            # Load existing index if it exists
            try:
                if hasattr(self.faiss_db.config, 'index_path'):
                    self.faiss_db.load_index(self.faiss_db.config.index_path)
                    logger.info(f"Loaded existing FAISS index from {self.faiss_db.config.index_path}")
            except Exception as e:
                logger.info(f"No existing FAISS index found or failed to load: {e}. Will create new index when chunks are added.")
        else:
            self.faiss_db = None  # owner-scoped; loaded lazily per owner_id

    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Adds a batch of chunks to the FAISS index using an executor thread.

        FAISS update/save are synchronous and can be expensive; running them in the
        default executor prevents blocking the event loop and allows other indexers
        (BM25/graph) to run concurrently.
        """
        if not chunks:
            return []

        def _sync_update_and_save() -> List[str]:
            if not self._owner_scoped_enabled:
                chunk_ids = self.faiss_db.update_index(chunks) or []
                if hasattr(self.faiss_db.config, "index_path"):
                    self.faiss_db.save_index(self.faiss_db.config.index_path)
                return chunk_ids

            # Owner-scoped: group by owner_id (usually one owner per file).
            from core.utils.owner_guard import normalize_owner_id
            from encapsulation.database.index_scoping import owner_scoped_dir
            from encapsulation.database.vector_db.faiss import FaissVectorDB

            by_owner: Dict[str, List[Chunk]] = {}
            for c in chunks:
                owner_raw = getattr(c, "owner_id", None) or getattr(getattr(c, "metadata", None) or {}, "get", lambda *_: None)("owner_id")
                owner_norm = normalize_owner_id(owner_raw)
                if owner_norm is None:
                    continue
                by_owner.setdefault(owner_norm, []).append(c)

            out_ids: List[str] = []
            base_dir = str(getattr(self._template_cfg, "index_path", "") or "").strip()
            owner_dirname = str(getattr(self._template_cfg, "owner_scoped_dirname", "owners") or "owners")
            global_owner = str(getattr(self._template_cfg, "owner_scoped_global_owner_name", "__GLOBAL__") or "__GLOBAL__")

            for owner_id, owner_chunks in by_owner.items():
                index_dir = owner_scoped_dir(
                    base_dir,
                    owner_id=owner_id,
                    owner_dirname=owner_dirname,
                    global_owner_name=global_owner,
                )
                with self._db_lock:
                    db = self._db_by_owner.get(owner_id)
                    if db is None:
                        try:
                            cfg2 = self._template_cfg.model_copy(update={"index_path": index_dir})
                        except Exception:
                            cfg2 = self._template_cfg
                            try:
                                cfg2.index_path = index_dir
                            except Exception:
                                pass
                        db = FaissVectorDB(cfg2)
                        self._db_by_owner[owner_id] = db

                # Load if artifacts exist but db is not loaded yet.
                try:
                    # index_dir may be an `io://...` virtual path; FaissVectorDB.load_index resolves it via IOManager.
                    if getattr(db, "index", None) is None:
                        db.load_index(index_dir)
                except Exception:
                    pass

                chunk_ids = db.update_index(owner_chunks) or []
                out_ids.extend(chunk_ids)
                try:
                    db.save_index(index_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to save owner-scoped FAISS index for owner=%s: %s", owner_id, exc)

            return out_ids

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_update_and_save)

    def delete_chunks(self, chunk_ids: List[str], **kwargs: Any) -> bool:
        """
        Deletes a batch of chunks from the FAISS index using soft-delete (synchronous).
        """
        try:
            if not chunk_ids:
                return True

            if not self._owner_scoped_enabled:
                # Delete chunks from index (soft-delete)
                result = self.faiss_db.delete_index(chunk_ids)
                # Save the index to disk
                if hasattr(self.faiss_db.config, "index_path"):
                    self.faiss_db.save_index(self.faiss_db.config.index_path)
                return result if result is not None else False

            # Owner-scoped deletion requires owner_id context.
            from core.utils.owner_guard import normalize_owner_id
            from encapsulation.database.index_scoping import owner_scoped_dir
            from encapsulation.database.vector_db.faiss import FaissVectorDB

            owner_norm = normalize_owner_id(kwargs.get("owner_id"))
            if owner_norm is None:
                logger.error("Owner-scoped FAISS deletion requires owner_id; got None")
                return False

            base_dir = str(getattr(self._template_cfg, "index_path", "") or "").strip()
            owner_dirname = str(getattr(self._template_cfg, "owner_scoped_dirname", "owners") or "owners")
            global_owner = str(getattr(self._template_cfg, "owner_scoped_global_owner_name", "__GLOBAL__") or "__GLOBAL__")
            index_dir = owner_scoped_dir(
                base_dir,
                owner_id=owner_norm,
                owner_dirname=owner_dirname,
                global_owner_name=global_owner,
            )

            with self._db_lock:
                db = self._db_by_owner.get(owner_norm)
                if db is None:
                    try:
                        cfg2 = self._template_cfg.model_copy(update={"index_path": index_dir})
                    except Exception:
                        cfg2 = self._template_cfg
                        try:
                            cfg2.index_path = index_dir
                        except Exception:
                            pass
                    db = FaissVectorDB(cfg2)
                    self._db_by_owner[owner_norm] = db

            # Load existing artifacts if present and index not loaded.
            try:
                # index_dir may be an `io://...` virtual path; FaissVectorDB.load_index resolves it via IOManager.
                if getattr(db, "index", None) is None:
                    db.load_index(index_dir)
            except Exception:
                pass

            result = db.delete_index(chunk_ids)
            try:
                db.save_index(index_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save owner-scoped FAISS index for owner=%s: %s", owner_norm, exc)
            return bool(True if result is None else result)
        except Exception as e:
            logger.error(f"Failed to delete chunks from FAISS index: {e}")
            return False
