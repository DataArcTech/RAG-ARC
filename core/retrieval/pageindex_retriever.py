import logging
from typing import Any, List, Optional

from config import pageindex as pageindex_cfg
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from core.retrieval.dense import DenseRetriever
from core.retrieval.tantivy_bm25 import TantivyBM25Retriever
from core.utils.fusion import RRFusion

logger = logging.getLogger(__name__)


def _coerce_file_id(meta: Any) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    for key in ("source_file_id", "file_id", "document_id", "doc_id"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    return None


def _build_dense_retriever(index_path: str) -> DenseRetriever:
    index_type_override = pageindex_cfg.section_faiss_index_type()
    two_stage_override = pageindex_cfg.section_faiss_two_stage_enabled()
    prefetch_k_override = pageindex_cfg.section_faiss_two_stage_prefetch_k()
    kwargs: dict[str, Any] = {
        "index_path": index_path,
        # Default: flat for section index (unless explicitly overridden).
        "index_type": index_type_override or "flat",
        "embedding_config": OpenAIEmbeddingConfig(),
    }
    # Optional overrides for section-only behavior. If unset, inherit from base FAISS config/env defaults.
    if two_stage_override is not None:
        kwargs["two_stage_enabled"] = two_stage_override
    if prefetch_k_override is not None:
        kwargs["two_stage_prefetch_k"] = prefetch_k_override
    faiss_cfg = FaissVectorDBConfig(**kwargs)
    retriever_cfg = DenseRetrieverConfig(index_config=faiss_cfg)
    return retriever_cfg.build()


def _build_bm25_retriever(index_path: str) -> TantivyBM25Retriever:
    bm25_cfg = BM25BuilderConfig(index_path=index_path)
    retriever_cfg = TantivyBM25RetrieverConfig(index_config=bm25_cfg)
    return retriever_cfg.build()


class PageIndexRetriever:
    def __init__(self) -> None:
        self.section_dense = None
        self.section_bm25 = None

        if pageindex_cfg.section_index_enabled():
            try:
                self.section_dense = _build_dense_retriever(pageindex_cfg.section_faiss_index_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section dense retriever unavailable: %s", exc)
            try:
                self.section_bm25 = _build_bm25_retriever(pageindex_cfg.section_bm25_index_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section BM25 retriever unavailable: %s", exc)

    def _filter_by_file_ids(self, chunks: List[Any], file_ids: Optional[List[str]]) -> List[Any]:
        if not file_ids:
            return chunks
        allowed = {str(fid) for fid in file_ids if str(fid)}
        if not allowed:
            return []
        out: List[Any] = []
        for chunk in chunks:
            meta = getattr(chunk, "metadata", None) or {}
            fid = _coerce_file_id(meta)
            if fid and fid in allowed:
                out.append(chunk)
        return out

    def retrieve_sections(
        self,
        query: str,
        *,
        owner_id: str,
        file_ids: Optional[List[str]] = None,
    ) -> List[Any]:
        if not pageindex_cfg.section_index_enabled():
            return []
        k_final = pageindex_cfg.section_top_k()
        k_candidates = max(pageindex_cfg.section_retrieval_candidates_k(), k_final)

        dense_hits: List[Any] = []
        bm25_hits: List[Any] = []
        if self.section_dense is not None:
            dense_hits = self.section_dense.invoke(query, k=k_candidates, owner_id=owner_id, with_score=True)
            dense_hits = self._filter_by_file_ids(dense_hits, file_ids)
        if self.section_bm25 is not None:
            filters = {"source_file_id": file_ids} if file_ids else None
            bm25_hits = self.section_bm25.invoke(
                query,
                k=k_candidates,
                owner_id=owner_id,
                with_score=True,
                filters=filters,
            )

        fusion = RRFusion(k=pageindex_cfg.section_rrf_k())
        return fusion.fuse([dense_hits, bm25_hits], k_final)
