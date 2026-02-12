from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from core.file_management.indexing.bm25_indexing import BM25Indexer
from core.file_management.indexing.faiss_indexing import FaissIndexer
from core.file_management.pageindex.indexing import (
    _clone_faiss_config,
    resolve_base_bm25_config,
    resolve_base_faiss_config,
)


def _base_cfg(*, index_type: str) -> FaissVectorDBConfig:
    # Embedding config is not built in these tests; keep it minimal and deterministic.
    emb = OpenAIEmbeddingConfig(openai_api_key="test", openai_base_url="http://test")
    return FaissVectorDBConfig(index_path="./local/tmp/test_faiss", index_type=index_type, embedding_config=emb)


def test_section_faiss_defaults_to_flat_even_if_base_is_hnsw(monkeypatch) -> None:
    monkeypatch.delenv("SECTION_FAISS_INDEX_TYPE", raising=False)
    monkeypatch.delenv("SECTION_FAISS_TWO_STAGE_ENABLED", raising=False)
    monkeypatch.delenv("SECTION_FAISS_TWO_STAGE_PREFETCH_K", raising=False)

    cloned = _clone_faiss_config(_base_cfg(index_type="hnsw"), index_path="./local/tmp/test_section_faiss")
    assert cloned.index_type == "flat"


def test_section_faiss_respects_explicit_index_type_override(monkeypatch) -> None:
    monkeypatch.setenv("SECTION_FAISS_INDEX_TYPE", "hnsw")
    cloned = _clone_faiss_config(_base_cfg(index_type="flat"), index_path="./local/tmp/test_section_faiss")
    assert cloned.index_type == "hnsw"


def test_section_faiss_two_stage_override_applies(monkeypatch) -> None:
    monkeypatch.setenv("SECTION_FAISS_INDEX_TYPE", "hnsw")
    monkeypatch.setenv("SECTION_FAISS_TWO_STAGE_ENABLED", "true")
    monkeypatch.setenv("SECTION_FAISS_TWO_STAGE_PREFETCH_K", "123")

    cloned = _clone_faiss_config(_base_cfg(index_type="flat"), index_path="./local/tmp/test_section_faiss")
    assert cloned.index_type == "hnsw"
    assert cloned.two_stage_enabled is True
    assert int(cloned.two_stage_prefetch_k) == 123


def test_resolve_base_configs_owner_scoped_indexers(tmp_path) -> None:
    emb = OpenAIEmbeddingConfig(openai_api_key="test", openai_base_url="http://test")
    faiss_cfg = FaissVectorDBConfig(
        index_path=str(tmp_path / "faiss"),
        owner_scoped_enabled=True,
        embedding_config=emb,
    )
    bm25_cfg = BM25BuilderConfig(
        index_path=str(tmp_path / "bm25"),
        owner_scoped_enabled=True,
    )

    faiss_indexer = FaissIndexer(FaissIndexerConfig(index_config=faiss_cfg))
    bm25_indexer = BM25Indexer(BM25IndexerConfig(index_config=bm25_cfg))

    assert faiss_indexer.faiss_db is None
    assert bm25_indexer.bm25_builder is None
    assert resolve_base_faiss_config([faiss_indexer]) is not None
    assert resolve_base_bm25_config([bm25_indexer]) is not None
