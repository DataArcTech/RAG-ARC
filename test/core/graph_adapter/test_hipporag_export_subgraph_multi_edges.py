from contextlib import nullcontext
from typing import Any, Dict, List

from core.graph_adapter.hipporag import HippoRAGGraphAdapter
from core.knowledge_graph.schema import schema_from_dict


class PrunedHippoRAGNeo4jStore:  # noqa: N801 - intentional name match for adapter dispatch
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        self.config = type("_Cfg", (), {"kg_schema_path": "./kg_schema.yml"})()
        self.kg_schema = schema_from_dict(
            {
                "version": "v1",
                "default_domain": "default",
                "domains": {"default": {"direction_sensitive_relations": ["OWNS"]}},
            }
        )

    def read_lock(self):
        return nullcontext()

    def _owner_key(self, owner_id: str) -> str:
        return str(owner_id)

    def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        text = str(query or "")
        if "MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)" in text:
            return [
                {
                    "source_id": "chunk-1",
                    "target_id": "entity-a",
                    "rel_type": "MENTIONS",
                    "weight": 1.0,
                    "predicate": None,
                    "fact_id": None,
                    "source_chunk_ids": None,
                    "schema_version": None,
                    "domain": None,
                    "source_type": "chunk",
                    "target_type": "entity",
                    "source_name": None,
                    "target_name": "A",
                }
            ]
        if "MATCH (c:Chunk)" in text:
            return [{"node_id": "chunk-1", "node_type": "chunk", "content": "c", "entity_name": None, "entity_type": None}]
        if "MATCH (e:Entity)" in text and "WHERE e.entity_id IN $entity_ids" in text:
            return [
                {"node_id": "entity-a", "node_type": "entity", "content": None, "entity_name": "A", "entity_type": "Entity"},
                {"node_id": "entity-b", "node_type": "entity", "content": None, "entity_name": "B", "entity_type": "Entity"},
            ]
        if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
            return [
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "rel_type": "RELATES_TO",
                    "weight": 1.0,
                    "predicate": "OWNS",
                    "fact_id": "fact-1",
                    "source_chunk_ids": [["chunk-1"]],
                    "schema_version": "v1",
                    "domain": "default",
                    "source_type": "entity",
                    "target_type": "entity",
                    "source_name": "A",
                    "target_name": "B",
                },
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "rel_type": "RELATES_TO",
                    "weight": 1.0,
                    "predicate": "HAS_POLICY",
                    "fact_id": "fact-2",
                    "source_chunk_ids": [["chunk-1"]],
                    "schema_version": "v1",
                    "domain": "default",
                    "source_type": "entity",
                    "target_type": "entity",
                    "source_name": "A",
                    "target_name": "B",
                },
            ]
        return []


class _StubRetriever:
    def __init__(self, graph_store):
        self.graph_store = graph_store


def test_hipporag_adapter_export_subgraph_preserves_multi_predicate_edges() -> None:
    graph_store = PrunedHippoRAGNeo4jStore()
    adapter = HippoRAGGraphAdapter(_StubRetriever(graph_store))
    payload = adapter._export_subgraph(  # noqa: SLF001
        {
            "subgraph_nodes": ["chunk-1", "entity-a", "entity-b"],
            "seed_entity_ids": ["entity-a"],
            "retrieved_chunk_ids": ["chunk-1"],
            "node_ppr_scores": {},
            "owner_id": "owner-1",
            "owner_scope": "owner-1",
        }
    )
    fact_edges = [e for e in payload.get("edges", []) if e.get("relation") in {"OWNS", "HAS_POLICY"}]
    assert {e.get("relation") for e in fact_edges} == {"OWNS", "HAS_POLICY"}
    owns = next(e for e in fact_edges if e.get("relation") == "OWNS")
    policy = next(e for e in fact_edges if e.get("relation") == "HAS_POLICY")
    assert owns.get("directed") is True
    assert policy.get("directed") is False
