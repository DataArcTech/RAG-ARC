import uuid
from types import SimpleNamespace

from application.rag_inference.module_bench import RAGInferenceBench
from encapsulation.data_model.schema import Chunk


class _StubRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def invoke(self, query, *, owner_id, return_subgraph_info, k):  # noqa: ANN001
        self.calls.append(
            {
                "query": query,
                "owner_id": owner_id,
                "return_subgraph_info": return_subgraph_info,
                "k": k,
            }
        )
        return list(self._chunks)


class _StubReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, chunks, *, top_k):  # noqa: ANN001
        self.calls.append({"query": query, "top_k": top_k, "chunks_in": len(chunks)})
        return list(chunks)[:top_k]


class _StubLLM:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.messages = messages
        return "bench-answer"


def test_rag_inference_bench_runs_without_rewrite_web_history(monkeypatch):
    monkeypatch.setenv("bench_mode", "1")

    chunks = [
        Chunk(id="c1", content="Paris is the capital of France.", metadata={"filename": "doc1", "prompt_text": "Paris is the capital of France."}),
        Chunk(id="c2", content="Berlin is the capital of Germany.", metadata={"filename": "doc2", "prompt_text": "Berlin is the capital of Germany."}),
    ]
    retriever = _StubRetriever(chunks)
    reranker = _StubReranker()
    llm = _StubLLM()

    rag = SimpleNamespace(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        config=SimpleNamespace(candidate_selection=SimpleNamespace(graph_candidates_k=10, rerank_keep_k=1)),
    )
    bench = RAGInferenceBench.from_rag_inference(rag)
    owner_id = uuid.uuid4()
    result = bench.chat("What is the capital of France?", owner_id=owner_id)

    assert result.answer == "bench-answer"
    assert len(result.chunks) == 1
    assert retriever.calls and retriever.calls[0]["query"] == "What is the capital of France?"
    assert llm.messages and llm.messages[0]["role"] == "system"
    assert "Return only the answer" in llm.messages[0]["content"]
