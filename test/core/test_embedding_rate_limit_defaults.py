import os

from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig


def test_openai_embedding_rate_limit_defaults(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MAX_RETRIES", raising=False)
    monkeypatch.delenv("EMBEDDING_RATE_LIMIT_MAX_RETRIES", raising=False)
    monkeypatch.delenv("EMBEDDING_RATE_LIMIT_DEFAULT_SLEEP_SECONDS", raising=False)
    monkeypatch.delenv("EMBEDDING_RATE_LIMIT_MAX_SLEEP_SECONDS", raising=False)

    cfg = OpenAIEmbeddingConfig(
        loading_method="openai",
        openai_api_key="test-key",
        openai_base_url="https://example.com/v1",
    )

    assert cfg.max_retries == 0
    assert cfg.rate_limit_default_sleep_seconds == 60.0
    assert cfg.rate_limit_max_sleep_seconds == 60.0
    assert cfg.rate_limit_max_retries == 6

