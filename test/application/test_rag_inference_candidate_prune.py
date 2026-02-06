import uuid
from types import SimpleNamespace

from encapsulation.data_model.schema import Chunk

from application.rag_inference.module import RAGInference


class _StubKnowledge:
    def is_file_active(self, file_id: str) -> bool:  # noqa: ARG002
        return True


class _StubRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)

    def invoke(self, query: str, **kwargs):  # noqa: ANN001, ARG002
        return list(self._chunks)


class _SpyReranker:
    def __init__(self) -> None:
        self.last_chunks_in: int | None = None

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5):  # noqa: ANN001, ARG002
        self.last_chunks_in = len(chunks)
        return list(chunks)[:top_k]

    def get_reranker_info(self) -> dict:
        return {}


def test_rag_inference_prunes_candidates_by_file_before_rerank(monkeypatch):
    # Avoid any optional PageIndex behavior during this unit test.
    monkeypatch.setenv("PAGEINDEX_ENABLED", "0")

    chunks: list[Chunk] = []
    # 10 files, 10 chunks each => 100 chunks retrieved.
    for f in range(10):
        for i in range(10):
            # Make file-0 most relevant.
            score = 10.0 - i if f == 0 else 1.0
            chunks.append(
                Chunk(
                    id=f"c{f}_{i}",
                    content="x",
                    metadata={"source_file_id": f"file-{f}", "filename": f"file-{f}.pdf", "score": score},
                )
            )

    rag = object.__new__(RAGInference)
    rag._knowledge_module = _StubKnowledge()
    rag.query_rewriter = SimpleNamespace(rewrite_query=lambda q, **_: q)
    rag.retriever = _StubRetriever(chunks)
    spy = _SpyReranker()
    rag.reranker = spy
    rag.llm = SimpleNamespace()
    rag.graph_retriever = None
    rag.pageindex_retriever = None
    rag._intent_routing = None
    rag._tavily_client = None
    rag.config = SimpleNamespace(
        candidate_selection=SimpleNamespace(
            graph_candidates_k=30,
            web_candidates_k=0,
            rerank_keep_k=5,
            file_prune_enabled=True,
            file_prune_max_files=4,
            file_prune_max_chunks_per_file=6,
        ),
        web_search=SimpleNamespace(enabled=False, timeout_seconds=0.1, timeout_grace_seconds=0.0),
    )

    rag._build_messages_and_context(
        query="hello",
        owner_id=uuid.uuid4(),
        return_subgraph=False,
        include_share=False,
    )

    # max_files=4, max_chunks_per_file=6 => <= 24 chunks fed into rerank.
    assert spy.last_chunks_in is not None
    assert spy.last_chunks_in <= 24

