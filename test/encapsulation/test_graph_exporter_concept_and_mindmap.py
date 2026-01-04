from typing import Any, Dict, List

from core.knowledge_graph.schema import schema_from_dict
from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j


class _StubStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        self.config = type("_Cfg", (), {"kg_schema_path": "./kg_schema.yml"})()
        self.kg_schema = schema_from_dict({"version": "v1", "default_domain": "default", "domains": {"default": {}}})

    def _owner_key(self, owner_id: str) -> str:  # noqa: D401
        return str(owner_id)

    def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        text = str(query or "")
        if "EntityAlias" in text and "collect(DISTINCT a.alias_text)" in text:
            return [{"entity_id": "entity-a", "aliases": ["中国平安", "Ping An"]}]
        if "MATCH (e:Entity)-[r:CANONICAL_OF]->(c:EntityCanonical)" in text:
            return [{"entity_id": "entity-a", "canonical_name": "平安保险", "canonical_key": "ping an|company"}]
        if "MATCH (c:Chunk)" in text and "RETURN c.chunk_id AS node_id" in text:
            return [
                {
                    "node_id": "chunk-1",
                    "node_type": "chunk",
                    "content": "条款：自2024年6月1日起生效。",
                    "metadata": '{"mindmap":{"nodes":[{"level":"1","content":"概念层: 保险条款"},{"level":"1.1","content":"过程层: 批单生效"}]},"business_time":{"effective_date":"2024-06-01T00:00:00+00:00"}}',
                    "entity_name": None,
                    "entity_type": None,
                }
            ]
        if "MATCH (e:Entity)" in text and "WHERE e.entity_id IN $entity_ids" in text:
            return [
                {"node_id": "entity-a", "node_type": "entity", "content": None, "metadata": None, "entity_name": "平安保险", "entity_type": "Company"}
            ]
        if "MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)" in text:
            return []
        if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
            return []
        if "MATCH (m:KGIngestMeta" in text:
            return []
        return []


def test_export_subgraph_includes_mindmap_preview_and_concept_hints() -> None:
    store = _StubStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"chunk-1", "entity-a"},
        owner_id="owner-1",
        max_edges=0,
    )

    assert payload["chunks"]
    chunk = payload["chunks"][0]
    assert "mindmap" in chunk
    assert "business_time" in chunk

    assert payload["nodes"]
    node = payload["nodes"][0]
    assert node.get("canonical_name") == "平安保险"
    assert node.get("aliases") == ["中国平安", "Ping An"]
