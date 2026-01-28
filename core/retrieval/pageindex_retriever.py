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
    faiss_cfg = FaissVectorDBConfig(
        index_path=index_path,
        embedding_config=OpenAIEmbeddingConfig(),
    )
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
        self.doc_dense = None
        self.doc_bm25 = None

        if pageindex_cfg.section_index_enabled():
            try:
                self.section_dense = _build_dense_retriever(pageindex_cfg.section_faiss_index_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section dense retriever unavailable: %s", exc)
            try:
                self.section_bm25 = _build_bm25_retriever(pageindex_cfg.section_bm25_index_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section BM25 retriever unavailable: %s", exc)

        if pageindex_cfg.doc_routing_enabled():
            try:
                self.doc_dense = _build_dense_retriever(pageindex_cfg.doc_routing_faiss_index_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Doc routing dense retriever unavailable: %s", exc)
            try:
                self.doc_bm25 = _build_bm25_retriever(pageindex_cfg.doc_routing_bm25_index_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Doc routing BM25 retriever unavailable: %s", exc)

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

    def retrieve_docs(
        self,
        query: str,
        *,
        owner_id: str,
    ) -> List[str]:
        if not pageindex_cfg.doc_routing_enabled():
            return []
        k_final = pageindex_cfg.doc_top_k()
        k_candidates = max(pageindex_cfg.doc_retrieval_candidates_k(), k_final)

        dense_hits: List[Any] = []
        bm25_hits: List[Any] = []
        if self.doc_dense is not None:
            dense_hits = self.doc_dense.invoke(query, k=k_candidates, owner_id=owner_id, with_score=True)
        if self.doc_bm25 is not None:
            bm25_hits = self.doc_bm25.invoke(query, k=k_candidates, owner_id=owner_id, with_score=True)

        fusion = RRFusion(k=pageindex_cfg.doc_rrf_k())
        fused = fusion.fuse([dense_hits, bm25_hits], k_final)

        file_ids: List[str] = []
        for chunk in fused:
            meta = getattr(chunk, "metadata", None) or {}
            fid = _coerce_file_id(meta)
            if not fid:
                fid = str(getattr(chunk, "id", "") or "").strip()
            if fid and fid not in file_ids:
                file_ids.append(fid)
        return file_ids

    def retrieve_doc_chunks(
        self,
        query: str,
        *,
        owner_id: str,
        k_final: Optional[int] = None,
        k_candidates: Optional[int] = None,
    ) -> List[Any]:
        """Return fused doc-routing chunks (title + doc description) with scores in metadata.

        This is the same doc-routing signal as `retrieve_docs`, but exposes the full chunk payload
        so DeepSearch tools can surface doc_description/filename to the LLM.
        """

        if not pageindex_cfg.doc_routing_enabled():
            return []
        final_k = int(k_final) if isinstance(k_final, int) else pageindex_cfg.doc_top_k()
        cand_k = int(k_candidates) if isinstance(k_candidates, int) else pageindex_cfg.doc_retrieval_candidates_k()
        cand_k = max(int(final_k), int(cand_k))

        dense_hits: List[Any] = []
        bm25_hits: List[Any] = []
        if self.doc_dense is not None:
            dense_hits = self.doc_dense.invoke(query, k=cand_k, owner_id=owner_id, with_score=True)
        if self.doc_bm25 is not None:
            bm25_hits = self.doc_bm25.invoke(query, k=cand_k, owner_id=owner_id, with_score=True)

        fusion = RRFusion(k=pageindex_cfg.doc_rrf_k())
        return fusion.fuse([dense_hits, bm25_hits], final_k)
