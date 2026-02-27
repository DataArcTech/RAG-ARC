import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from encapsulation.data_model.schema import Chunk
from encapsulation.web_search.tavily_client import TavilySearchClient, TavilySearchResult

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


@pytest.fixture(autouse=True)
def _seed_registry() -> None:
    from framework.register import Register

    reg = Register()
    reg.registrations.setdefault("account", object())
    reg.registrations.setdefault("chat_session", object())
    reg.registrations.setdefault("chat_message", object())
    reg.registrations.setdefault("rag_inference", object())
    reg.registrations.setdefault("knowledge", object())


class DummyQueryRewriter:
    def rewrite_query(self, q):  # noqa: ANN001
        return q


class DummyRetriever:
    def invoke(self, q, **kwargs):  # noqa: ANN001
        k = int(kwargs.get("k") or 30)
        out = []
        for i in range(k):
            out.append(
                Chunk(
                    id=f"chunk-{i}",
                    content=f"graph chunk {i} for {q}",
                    metadata={"score": float(k - i), "filename": "graph"},
                )
            )
        return out


class DummyReranker:
    def rerank(self, _query, chunks, **kwargs):  # noqa: ANN001
        top_k = kwargs.get("top_k")

        def _rank_key(ch):  # noqa: ANN001
            src = (getattr(ch, "metadata", None) or {}).get("source")
            return (0 if src == "web.tavily" else 1)

        ranked = sorted(chunks, key=_rank_key)
        if top_k is not None:
            return ranked[: int(top_k)]
        return ranked

    def get_reranker_info(self):
        return {"type": "dummy"}


class DummyLLM:
    def stream_chat(self, _messages):  # noqa: ANN001
        def _gen():
            yield "assistant ok"

        return _gen()


class StubTavily:
    def search(self, *, query, max_results):  # noqa: ANN001
        results = []
        for i in range(int(max_results)):
            results.append(
                TavilySearchResult(
                    title=f"Web {i}",
                    url=f"https://example.com/{i}",
                    content=f"snippet {i} for {query}",
                    score=1.0 - i * 0.1,
                )
            )
        return results

    @staticmethod
    def to_evidence_chunks(*, results, step_id, query):  # noqa: ANN001
        return TavilySearchClient.to_evidence_chunks(results=results, step_id=step_id, query=query)


def _build_stub_rag():
    from application.rag_inference.module import RAGInference

    rag = object.__new__(RAGInference)
    rag.config = SimpleNamespace(
        web_search=SimpleNamespace(timeout_seconds=1.0, timeout_grace_seconds=0.0, enabled=True),
        candidate_selection=SimpleNamespace(graph_candidates_k=30, web_candidates_k=5, rerank_keep_k=5),
    )
    rag.query_rewriter = DummyQueryRewriter()
    rag.retriever = DummyRetriever()
    rag.reranker = DummyReranker()
    rag.llm = DummyLLM()
    rag.graph_retriever = None
    rag._knowledge_module = None
    rag._tavily_client = StubTavily()
    return rag


@patch("application.rag_inference.module.benchmark_mode_enabled", return_value=False)
def test_stream_chat_web_search_includes_chunks_and_progress(_mock_bench):
    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4(), type=0)
    rag = _build_stub_rag()

    progress_events: list[dict] = []

    def _collect(ev: dict) -> None:
        progress_events.append(ev)

    _messages, chunks, _subgraph_data, _subgraph_info = rag._build_messages_and_context(
        query="hello",
        owner_id=user.id,
        return_subgraph=False,
        progress_callback=_collect,
        history_text=None,
        enable_web_search=True,
        session_id=session_id,
        user_type=user.type,
        include_share=False,
        share_owner_id=None,
        debug_state=None,
        disable_intent_routing=False,
    )

    web_progress = [ev for ev in progress_events if ev.get("stage") == "web_search"]
    assert any((ev.get("status") == "start") for ev in web_progress)
    end_events = [ev for ev in web_progress if ev.get("status") == "end"]
    assert end_events
    web_end = end_events[-1]
    aggregation = web_end.get("aggregation") or {}
    assert aggregation.get("total_in") == 5
    assert aggregation.get("total_out") == 2

    assert len(chunks) == 5
    web_chunks = [c for c in chunks if (getattr(c, "metadata", None) or {}).get("source") == "web.tavily"]
    assert len(web_chunks) == 2


def test_stream_chat_defaults_to_no_web_search():
    session_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4(), type=0)
    rag = _build_stub_rag()

    progress_events: list[dict] = []

    def _collect(ev: dict) -> None:
        progress_events.append(ev)

    _messages, chunks, _subgraph_data, _subgraph_info = rag._build_messages_and_context(
        query="hello",
        owner_id=user.id,
        return_subgraph=False,
        progress_callback=_collect,
        history_text=None,
        enable_web_search=False,
        session_id=session_id,
        user_type=user.type,
        include_share=False,
        share_owner_id=None,
        debug_state=None,
        disable_intent_routing=False,
    )

    assert not any((ev.get("stage") == "web_search") for ev in progress_events)
    assert len(chunks) == 5
    assert not any(((getattr(c, "metadata", None) or {}).get("source") == "web.tavily") for c in chunks)

