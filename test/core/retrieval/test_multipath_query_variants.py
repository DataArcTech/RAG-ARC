from dataclasses import dataclass
from unittest.mock import patch

from encapsulation.data_model.schema import Chunk
from core.retrieval.multipath import MultiPathRetriever
from core.utils import query_variants


class _DummyRetriever:
    def __init__(self, *, results: list[Chunk]):
        self.results = results
        self.invoked_queries: list[str] = []

    def invoke(self, query: str, **kwargs):  # noqa: ARG002
        self.invoked_queries.append(str(query))
        return list(self.results)


@dataclass
class _DummyConfig:
    fusion_method: str = "rrf"
    rrf_k: int = 60
    weights: list[float] | None = None
    retrievers: list[object] = None  # only used for weights defaults
    search_kwargs: dict = None
    built_retrievers: list[object] | None = None
    fusion_instance: object | None = None


@patch("config.benchmark_mode.benchmark_mode_enabled", return_value=False)
def test_multipath_calls_retriever_for_distinct_query_variants(_mock_bench, monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "generate_query_variants",
        lambda query: [str(query), "計劃特點"],
    )
    retriever = _DummyRetriever(results=[Chunk(id="c1", content="x", metadata={"score": 1.0})])
    config = _DummyConfig(retrievers=[object()], search_kwargs={"k": 5, "with_score": True}, built_retrievers=[retriever])
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    out = multipath._get_relevant_chunks("计划特点", k=5, owner_id="owner")
    assert out
    # Expect original + OpenCC s2t variant.
    assert retriever.invoked_queries[0] == "计划特点"
    assert "計劃特點" in retriever.invoked_queries


@patch("config.benchmark_mode.benchmark_mode_enabled", return_value=False)
def test_multipath_does_not_duplicate_calls_when_no_variants(_mock_bench, monkeypatch):
    monkeypatch.setattr(
        query_variants,
        "generate_query_variants",
        lambda query: [str(query)],
    )
    retriever = _DummyRetriever(results=[Chunk(id="c1", content="x", metadata={"score": 1.0})])
    config = _DummyConfig(retrievers=[object()], search_kwargs={"k": 5, "with_score": True}, built_retrievers=[retriever])
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    _ = multipath._get_relevant_chunks("hello world", k=5, owner_id="owner")
    assert retriever.invoked_queries == ["hello world"]
