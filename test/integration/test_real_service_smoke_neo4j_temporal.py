import os
import uuid

import pytest

from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig
from config.core.file_management.indexing.graph_indexing.pruned_hipporag_indexing_config import PrunedHippoRAGIndexerConfig
from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from core.deepsearch.tools.base import ToolRunRequest
from core.deepsearch.tools import GraphOpsTool
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.schema import Chunk


def _env_ready() -> bool:
    required = ("NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE", "OPENAI_API_KEY", "OPENAI_BASE_URL")
    return all(str(os.getenv(name, "")).strip() for name in required)


class _Neo4jCypherAdapter:
    def __init__(self, graph_store):
        self._graph_store = graph_store

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001
        owner_scope = getattr(access_scope, "scope_id", None) if access_scope is not None else None
        if owner_scope is None:
            raise RuntimeError("access_scope is required")
        owner_key = self._graph_store._owner_key(owner_scope)
        merged = dict(params or {})
        merged["global_owner"] = getattr(self._graph_store, "OWNER_GLOBAL_KEY", "__GLOBAL__")
        merged["owner_id"] = owner_key
        return self._graph_store._execute_query(str(cypher), merged)

    def metadata(self):
        return {}


@pytest.mark.asyncio
async def test_real_service_smoke_latest_truth_neo4j():  # pragma: no cover - opt-in integration
    if os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test opt-in: set RUN_RAGARC_INTEGRATION_TESTS=1")
    if not _env_ready():
        pytest.skip("integration test requires Neo4j + OpenAI-compatible env")

    owner_id = str(uuid.uuid4())

    extractor_cfg = HippoRAG2ExtractorConfig(
        llm_config=OpenAIChatConfig(model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0.0),
        max_concurrent=2,
        enable_temporal_extraction=True,
    )
    graph_store_cfg = PrunedHippoRAGNeo4jConfig(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        embedding=OpenAIEmbeddingConfig(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
        shared_instance=False,
    )
    indexer = PrunedHippoRAGIndexerConfig(extractor_config=extractor_cfg, graph_store_config=graph_store_cfg).build()

    chunks = [
        Chunk(
            id="policy-v1",
            owner_id=owner_id,
            content="概念：远程办公政策。自2024年1月1日起，远程办公政策规定：每周可远程办公2天。",
            metadata={"source": "integration", "version": "v1"},
        ),
        Chunk(
            id="policy-v2",
            owner_id=owner_id,
            content="概念：远程办公政策。自2024年6月1日起，远程办公政策调整为：每周可远程办公0天。",
            metadata={"source": "integration", "version": "v2"},
        ),
        Chunk(
            id="empty-chunk",
            owner_id=owner_id,
            content="（空图 chunk，用于验证不会被 indexer 丢弃。）",
            metadata={"source": "integration", "note": "expected_empty_graph"},
        ),
    ]

    await indexer.update_index(chunks)
    adapter = _Neo4jCypherAdapter(indexer.graph_store)
    tool = GraphOpsTool()

    result = await tool.run(
        ToolRunRequest(
            question="每周可远程办公几天？",
            plan_step="integration",
            context_evidences=[],
            adapter=adapter,
            access_scope=GraphAccessScope(scope_id=owner_id),
            extra={
                "mode": "template",
                "template": "latest_truth",
                "template_args": {"topic": "远程办公政策", "predicates": ["HAS_POLICY"]},
            },
        )
    )
    assert "latest_truth" in result.summary
    assert (result.diagnostics or {}).get("value")
