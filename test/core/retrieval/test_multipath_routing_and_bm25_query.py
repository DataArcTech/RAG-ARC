from types import SimpleNamespace

from encapsulation.data_model.schema import Chunk


class _StubRetriever:
    def __init__(self, *, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    def invoke(self, query: str, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **{k: kwargs.get(k) for k in ("owner_id", "k")}})
        # Return 1 chunk per retriever so fusion sees something.
        return [
            Chunk(
                id=f"{self.name}-1",
                content=f"{self.name}:{query}",
                metadata={"source_file_id": f"file-{self.name}", "score": 1.0},
            )
        ]


def _make_cfg(retrievers, built_retrievers):
    # Minimal MultiPath config surface used by MultiPathRetriever.
    return SimpleNamespace(
        type="multipath",
        fusion_method="rrf",
        rrf_k=60,
        weights=[1.0 for _ in retrievers],
        retrievers=retrievers,
        built_retrievers=built_retrievers,
        search_kwargs={"k": 10, "with_score": True},
    )


def test_multipath_uses_bm25_query_and_can_disable_graph_via_routing_ratios() -> None:
    from core.retrieval.multipath import MultiPathRetriever

    dense_cfg = SimpleNamespace(type="dense")
    bm25_cfg = SimpleNamespace(type="tantivy_bm25")
    graph_cfg = SimpleNamespace(type="pruned_hipporag_neo4j_retrieval")

    dense = _StubRetriever(name="dense")
    bm25 = _StubRetriever(name="bm25")
    graph = _StubRetriever(name="graph")

    cfg = _make_cfg([dense_cfg, bm25_cfg, graph_cfg], [dense, bm25, graph])
    mp = MultiPathRetriever(cfg)

    out = mp.invoke(
        "original query",
        owner_id="owner",
        return_subgraph_info=False,
        k=10,
        bm25_query="bm25 keywords",
        retrieval_ratios={"dense": 1.0, "bm25": 1.0, "graph": 0.0},
    )
    assert out, "expected some fused chunks"

    assert dense.calls and dense.calls[0]["query"] == "original query"
    assert bm25.calls and bm25.calls[0]["query"] == "bm25 keywords"
    assert graph.calls == [], "graph retriever should be disabled when routing ratio graph=0"


def test_multipath_can_disable_graph_via_weights() -> None:
    from core.retrieval.multipath import MultiPathRetriever

    dense_cfg = SimpleNamespace(type="dense")
    bm25_cfg = SimpleNamespace(type="tantivy_bm25")
    graph_cfg = SimpleNamespace(type="pruned_hipporag_neo4j_retrieval")

    dense = _StubRetriever(name="dense")
    bm25 = _StubRetriever(name="bm25")
    graph = _StubRetriever(name="graph")

    cfg = _make_cfg([dense_cfg, bm25_cfg, graph_cfg], [dense, bm25, graph])
    cfg.weights = [1.0, 1.0, 0.0]
    mp = MultiPathRetriever(cfg)

    out = mp.invoke(
        "original query",
        owner_id="owner",
        return_subgraph_info=False,
        k=10,
        bm25_query="bm25 keywords",
        retrieval_ratios=None,
    )
    assert out, "expected some fused chunks"

    assert dense.calls and dense.calls[0]["query"] == "original query"
    assert bm25.calls and bm25.calls[0]["query"] == "bm25 keywords"
    assert graph.calls == [], "graph retriever should be disabled when weight<=0"
