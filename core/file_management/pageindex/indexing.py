import asyncio
import logging
from typing import Any, Iterable, List, Optional

from config import pageindex as pageindex_cfg
from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from core.file_management.indexing.bm25_indexing import BM25Indexer
from core.file_management.indexing.faiss_indexing import FaissIndexer
from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


def _clone_faiss_config(base: Optional[FaissVectorDBConfig], *, index_path: str) -> FaissVectorDBConfig:
    section_index_type = pageindex_cfg.section_faiss_index_type()
    section_two_stage_enabled = pageindex_cfg.section_faiss_two_stage_enabled()
    section_two_stage_prefetch_k = pageindex_cfg.section_faiss_two_stage_prefetch_k()
    # Default to flat for section index unless explicitly overridden.
    effective_index_type = section_index_type or "flat"
    if base is not None:
        try:
            update = {"index_path": index_path}
            update["index_type"] = effective_index_type
            if section_two_stage_enabled is not None:
                update["two_stage_enabled"] = section_two_stage_enabled
            if section_two_stage_prefetch_k is not None:
                update["two_stage_prefetch_k"] = section_two_stage_prefetch_k
            return base.model_copy(update=update)
        except Exception:  # noqa: BLE001
            pass
    embedding_config = OpenAIEmbeddingConfig()
    kwargs: dict[str, Any] = {"index_path": index_path, "embedding_config": embedding_config}
    kwargs["index_type"] = effective_index_type
    if section_two_stage_enabled is not None:
        kwargs["two_stage_enabled"] = section_two_stage_enabled
    if section_two_stage_prefetch_k is not None:
        kwargs["two_stage_prefetch_k"] = section_two_stage_prefetch_k
    return FaissVectorDBConfig(**kwargs)


def _clone_bm25_config(base: Optional[BM25BuilderConfig], *, index_path: str) -> BM25BuilderConfig:
    if base is not None:
        try:
            return base.model_copy(update={"index_path": index_path})
        except Exception:  # noqa: BLE001
            pass
    return BM25BuilderConfig(index_path=index_path)


def resolve_base_faiss_config(indexers: Iterable[Any]) -> Optional[FaissVectorDBConfig]:
    for indexer in indexers or []:
        if isinstance(indexer, FaissIndexer):
            return getattr(indexer.faiss_db, "config", None)
    return None


def resolve_base_bm25_config(indexers: Iterable[Any]) -> Optional[BM25BuilderConfig]:
    for indexer in indexers or []:
        if isinstance(indexer, BM25Indexer):
            return getattr(indexer.bm25_builder, "config", None)
    return None


class PageIndexIndexers:
    def __init__(
        self,
        *,
        base_faiss_config: Optional[FaissVectorDBConfig],
        base_bm25_config: Optional[BM25BuilderConfig],
    ) -> None:
        self.section_faiss = None
        self.section_bm25 = None

        if pageindex_cfg.section_index_enabled():
            if base_faiss_config is not None:
                faiss_cfg = _clone_faiss_config(
                    base_faiss_config,
                    index_path=pageindex_cfg.section_faiss_index_path(),
                )
                self.section_faiss = FaissIndexer(FaissIndexerConfig(index_config=faiss_cfg))
            else:
                logger.warning("PageIndex section FAISS indexer skipped: base FAISS config missing")

            if base_bm25_config is not None:
                bm25_cfg = _clone_bm25_config(
                    base_bm25_config,
                    index_path=pageindex_cfg.section_bm25_index_path(),
                )
                self.section_bm25 = BM25Indexer(BM25IndexerConfig(index_config=bm25_cfg))
            else:
                logger.warning("PageIndex section BM25 indexer skipped: base BM25 config missing")

    async def _index_batch(self, indexer, chunks: List[Chunk]) -> Optional[List[str]]:
        if indexer is None or not chunks:
            return None
        try:
            return await indexer.update_index(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PageIndex indexing failed (%s): %s", type(indexer).__name__, exc)
            return None

    async def index_sections(self, chunks: List[Chunk]) -> dict:
        results: dict[str, Any] = {}
        if not chunks:
            return results
        tasks = []
        if self.section_faiss is not None:
            tasks.append(("section_faiss", self.section_faiss))
        if self.section_bm25 is not None:
            tasks.append(("section_bm25", self.section_bm25))

        coros = [self._index_batch(indexer, chunks) for _, indexer in tasks]
        indexed = await asyncio.gather(*coros, return_exceptions=False)
        for (name, _), ids in zip(tasks, indexed):
            results[name] = {"success": bool(ids), "indexed_count": len(ids or [])}
        return results
