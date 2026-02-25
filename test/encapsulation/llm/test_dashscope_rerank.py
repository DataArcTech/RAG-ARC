"""Tests for DashScope rerank client (encapsulation/llm/rerank/dashscope_rerank.py)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from encapsulation.llm.rerank.dashscope_rerank import DashScopeRerankLLM


class _StubConfig:
    type = "dashscope_rerank"
    api_key = "test-key-123"
    base_url = "https://dashscope.example.com/compatible-api/v1"
    model_name = "qwen3-rerank"
    instruct = "Given a web search query, retrieve relevant passages that answer the query."
    timeout_seconds = 10.0


class _UnconfiguredConfig:
    type = "dashscope_rerank"
    api_key = ""
    base_url = "https://dashscope.example.com/compatible-api/v1"
    model_name = "qwen3-rerank"
    instruct = ""
    timeout_seconds = 10.0


def test_is_configured_with_key() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    assert client.is_configured() is True


def test_is_not_configured_without_key() -> None:
    client = DashScopeRerankLLM(_UnconfiguredConfig())
    assert client.is_configured() is False


def test_get_model_info() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    info = client.get_model_info()
    assert info["model"] == "qwen3-rerank"
    assert info["provider"] == "dashscope"
    assert info["base_url"] == "https://dashscope.example.com/compatible-api/v1"


def test_parse_response_standard() -> None:
    data = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.82},
            {"index": 1, "relevance_score": 0.50},
        ]
    }
    parsed = DashScopeRerankLLM._parse_response(data, num_documents=3)
    assert parsed == [(0, 0.95), (2, 0.82), (1, 0.50)]


def test_parse_response_empty_results() -> None:
    data = {"results": []}
    parsed = DashScopeRerankLLM._parse_response(data, num_documents=3)
    assert parsed == []


def test_parse_response_missing_results_key() -> None:
    data = {"something_else": []}
    parsed = DashScopeRerankLLM._parse_response(data, num_documents=3)
    assert parsed == []


def test_parse_response_filters_out_of_range_indices() -> None:
    data = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 99, "relevance_score": 0.8},  # out of range
            {"index": -1, "relevance_score": 0.7},  # negative
        ]
    }
    parsed = DashScopeRerankLLM._parse_response(data, num_documents=3)
    assert len(parsed) == 1
    assert parsed[0] == (0, 0.9)


def test_parse_response_handles_invalid_items() -> None:
    data = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": None, "relevance_score": 0.5},
            {"index": 1},  # missing score
            "not a dict",
        ]
    }
    parsed = DashScopeRerankLLM._parse_response(data, num_documents=3)
    assert len(parsed) == 1
    assert parsed[0] == (0, 0.9)


def test_build_payload() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    payload = client._build_payload("test query", ["doc1", "doc2"], top_k=2)
    assert payload["model"] == "qwen3-rerank"
    assert payload["query"] == "test query"
    assert payload["documents"] == ["doc1", "doc2"]
    assert payload["top_n"] == 2
    assert "instruct" in payload


def test_build_payload_no_top_k() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    payload = client._build_payload("test query", ["doc1", "doc2"], top_k=None)
    assert "top_n" not in payload


def test_build_headers() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer test-key-123"
    assert headers["Content-Type"] == "application/json"


def test_rerank_empty_chunks() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    result = client.rerank(query="test", chunks=[], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_arerank_empty_documents() -> None:
    client = DashScopeRerankLLM(_StubConfig())
    result = await client.arerank(query="test", documents=[], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_arerank_calls_api_and_parses_response() -> None:
    """Mock httpx.AsyncClient to verify the API call and response parsing."""
    client = DashScopeRerankLLM(_StubConfig())

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.70},
            {"index": 2, "relevance_score": 0.50},
        ]
    }

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    import httpx as _httpx_mod
    with patch.object(_httpx_mod, "AsyncClient", return_value=mock_async_client):
        result = await client.arerank(
            query="what is the revenue?",
            documents=["doc about costs", "doc about revenue", "doc about hiring"],
            top_k=3,
        )

    assert result == [(1, 0.95), (0, 0.70), (2, 0.50)]
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    assert "/reranks" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["model"] == "qwen3-rerank"
    assert payload["query"] == "what is the revenue?"
    assert len(payload["documents"]) == 3


@pytest.mark.asyncio
async def test_arerank_handles_api_error() -> None:
    """Verify that API errors are caught and re-raised as RuntimeError."""
    client = DashScopeRerankLLM(_StubConfig())

    mock_async_client = AsyncMock()
    mock_async_client.post.side_effect = Exception("connection timeout")
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    import httpx as _httpx_mod
    with patch.object(_httpx_mod, "AsyncClient", return_value=mock_async_client):
        with pytest.raises(RuntimeError, match="DashScope rerank failed"):
            await client.arerank(
                query="test query",
                documents=["doc1", "doc2"],
            )
