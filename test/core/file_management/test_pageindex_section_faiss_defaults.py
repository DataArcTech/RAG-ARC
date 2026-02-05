import os

from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from core.file_management.pageindex.indexing import _clone_faiss_config


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

