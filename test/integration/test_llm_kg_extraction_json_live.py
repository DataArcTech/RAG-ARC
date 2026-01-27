import os
import re

import pytest

from encapsulation.data_model.schema import Chunk
from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1",
    reason="integration test opt-in: set RUN_RAGARC_INTEGRATION_TESTS=1",
)


def _has_chat_creds() -> bool:
    return bool(os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")) and bool(
        os.getenv("CHAT_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE_URL")
    )


def _build_llm_config() -> OpenAIChatConfig:
    return OpenAIChatConfig(
        model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        max_tokens=800,
        openai_api_key=os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("CHAT_API_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


@pytest.mark.asyncio
async def test_live_llm_hipporag2_extractor_json_smoke() -> None:
    if not _has_chat_creds():
        pytest.skip("Missing chat credentials (CHAT_API_KEY/OPENAI_API_KEY and CHAT_API_BASE_URL/OPENAI_BASE_URL)")

    cfg = HippoRAG2ExtractorConfig(
        type="hipporag2_extractor",
        llm_config=_build_llm_config(),
        max_concurrent=1,
        enable_temporal_extraction=False,
        enable_mindmap_extraction=False,
        enable_sdf_extraction=False,
    )
    extractor = cfg.build()

    chunk = Chunk(
        id="live-1",
        content="Alice works at Acme Corp. Bob manages Alice at Acme Corp.",
        metadata={"source": "integration_smoke"},
    )

    graph = await extractor.extract(chunk)
    assert len(graph.entities) >= 2
    assert len(graph.relations) >= 1
    rel_re = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for rel in graph.relations:
        assert rel_re.match(str(rel[1] or ""))


@pytest.mark.asyncio
async def test_live_llm_graphextractor_json_smoke() -> None:
    if not _has_chat_creds():
        pytest.skip("Missing chat credentials (CHAT_API_KEY/OPENAI_API_KEY and CHAT_API_BASE_URL/OPENAI_BASE_URL)")

    cfg = GraphExtractorConfig(
        type="graph_extractor",
        llm_config=_build_llm_config(),
        max_concurrent=1,
    )
    extractor = cfg.build()

    chunk = Chunk(
        id="live-2",
        content="Alice works at Acme Corp. Acme Corp is located in Cupertino.",
        metadata={"source": "integration_smoke"},
    )

    graph = await extractor.extract(chunk)
    assert len(graph.entities) >= 2
    assert len(graph.relations) >= 1

