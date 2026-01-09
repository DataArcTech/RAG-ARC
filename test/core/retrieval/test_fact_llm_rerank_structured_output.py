from types import SimpleNamespace

import numpy as np

from core.retrieval.graph_retrieveal.pruned_hipporag_facts import _PrunedHippoRAGFactsMixin


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._responses:
            return ""
        return self._responses.pop(0)


class _Runner(_PrunedHippoRAGFactsMixin):
    def __init__(self, llm: _FakeLLM):
        self.llm_client = llm
        self.config = SimpleNamespace(
            fact_retrieval_top_k=5,
            max_facts_after_reranking=3,
            enable_llm_reranking=True,
        )

    def _get_query_embedding(self, query: str) -> np.ndarray:  # noqa: ARG002
        return np.zeros(3, dtype=np.float32)


def test_llm_rerank_accepts_json_object_indices() -> None:
    llm = _FakeLLM(['{"indices":[2,1]}'])
    runner = _Runner(llm)

    candidate_facts = [
        ("h1", "r", "t1", "owner"),
        ("h2", "r", "t2", "owner"),
        ("h3", "r", "t3", "owner"),
    ]
    candidate_indices = [10, 11, 12]

    facts, idxs = runner._llm_rerank_filter("q", candidate_facts, candidate_indices, len_after_rerank=2)
    assert facts == [candidate_facts[1], candidate_facts[0]]
    assert idxs == [candidate_indices[1], candidate_indices[0]]
    assert len(llm.calls) == 1
    assert llm.calls[0]["kwargs"].get("response_format") == {"type": "json_object"}


def test_llm_rerank_retries_on_parse_failure_before_fallback() -> None:
    llm = _FakeLLM(["I think 1 and 2 are best.", '{"indices":[1,3]}'])
    runner = _Runner(llm)

    candidate_facts = [
        ("h1", "r", "t1", "owner"),
        ("h2", "r", "t2", "owner"),
        ("h3", "r", "t3", "owner"),
    ]
    candidate_indices = [10, 11, 12]

    facts, idxs = runner._llm_rerank_filter("q", candidate_facts, candidate_indices, len_after_rerank=2)
    assert facts == [candidate_facts[0], candidate_facts[2]]
    assert idxs == [candidate_indices[0], candidate_indices[2]]
    assert len(llm.calls) == 2

