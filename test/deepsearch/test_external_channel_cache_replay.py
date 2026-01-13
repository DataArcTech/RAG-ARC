import json

import pytest

from encapsulation.deepsearch.external.channel import ExternalSearchChannel


@pytest.mark.asyncio
async def test_external_channel_replay_returns_tavily_chunks(tmp_path):
    task = {
        "step_id": "web_01",
        "channel": "web",
        "tool": "web.search",
        "tool_args": {"query": "singapore american school brochure"},
        "metadata": {"provider": "tavily", "source": "test"},
        "requires_external": True,
    }

    cache_key = ExternalSearchChannel._cache_key(  # noqa: SLF001
        provider="tavily",
        tool_name="web.search",
        query=task["tool_args"]["query"],
        task=task,
    )

    record = {
        "schema_version": 1,
        "provider": "tavily",
        "tool_name": "web.search",
        "query": task["tool_args"]["query"],
        "task": task,
        "response": {
            "event": "replay",
            "diagnostics": {"cached": True},
            "chunks": [
                {
                    "chunk_id": "tavily-quality_web_r1_00-0",
                    "source": "web.tavily",
                    "content": "Web Title Line\nWeb snippet body",
                    "score": 1.0,
                    "provenance": {"provider": "tavily", "url": "https://example.com/doc.pdf"},
                }
            ],
        },
    }
    (tmp_path / f"{cache_key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    channel = ExternalSearchChannel(
        tool_manager=object(),
        config={
            "enabled": True,
            "default_provider": "tavily",
            "max_rounds": 1,
            "context_window_limit": 1,
            "http_timeout": 1.0,
            "endpoint_url": "https://api.tavily.com/search",
            "max_results": 1,
            "tool_timeout_seconds": 1.0,
            "cache_mode": "replay",
            "cache_dir": str(tmp_path),
            "tavily_api_key": None,
        },
    )

    result = await channel.run([task], reasoning_trace={"question": "test", "evidences": []})

    assert result["logs"] and result["logs"][0]["status"] == "replay"
    assert result["logs"][0]["cache_key"] == cache_key
    assert result["evidences"] and result["evidences"][0]["chunk_id"].startswith("tavily-")
    assert result["evidences"][0]["provenance"]["url"] == "https://example.com/doc.pdf"

