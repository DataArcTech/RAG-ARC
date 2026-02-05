from dataclasses import dataclass

import httpx
import openai
import pytest

from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.llm.embedding.openai import OpenAIEmbeddingLLM


@dataclass
class _Item:
    embedding: list[float]


@dataclass
class _Resp:
    data: list[_Item]


class _StubEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *args, **kwargs):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://test"))
        return _Resp(data=[_Item(embedding=[0.1, 0.2, 0.3])])


class _StubClient:
    def __init__(self) -> None:
        self.embeddings = _StubEmbeddings()


def test_embedding_transient_retry_then_success(monkeypatch) -> None:
    # Avoid any real sleeping in unit tests.
    monkeypatch.setattr("encapsulation.llm.embedding.openai.time.sleep", lambda _s: None)
    # Avoid creating a real OpenAI client.
    monkeypatch.setattr("encapsulation.llm.embedding.openai.create_openai_sync_client", lambda _cfg: _StubClient())
    monkeypatch.setattr("encapsulation.llm.embedding.openai.create_openai_async_client", lambda _cfg: None)

    cfg = OpenAIEmbeddingConfig(
        openai_api_key="test",
        openai_base_url="http://test",
        supports_batch_input=True,
        request_batch_size=64,
        # Keep rate-limit retries off in this unit test; focus on transient retry.
        rate_limit_max_retries=0,
        transient_max_retries=2,
        transient_backoff_initial_seconds=0.0,
        transient_backoff_max_seconds=0.0,
        transient_backoff_jitter_seconds=0.0,
    )
    llm = OpenAIEmbeddingLLM(cfg)
    out = llm.embed("hello")
    assert out == [0.1, 0.2, 0.3]
    assert llm.client.embeddings.calls == 2

