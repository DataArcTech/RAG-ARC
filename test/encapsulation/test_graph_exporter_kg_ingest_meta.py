from typing import Any, Dict, List

from core.knowledge_graph.schema import schema_from_dict
from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j


class _MetaStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        self.config = type("_Cfg", (), {"kg_schema_path": "./kg_schema.yml"})()
        self.kg_schema = schema_from_dict(
            {"version": "v1", "default_domain": "default", "domains": {"default": {"direction_sensitive_relations": ["OWNS"]}}}
        )

    def _owner_key(self, owner_id: str) -> str:
        return str(owner_id)

    def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        text = str(query or "")
        if "MATCH (m:KGIngestMeta" in text:
            return [
                {
                    "chunks_total": 4,
                    "chunks_graph_empty": 1,
                    "chunks_extraction_failed": 1,
                    "triples_total": 10,
                    "triples_kept": 7,
                    "triples_dropped_endpoints": 2,
                    "triples_dropped_schema": 1,
                    "predicates_aliased": 2,
                    "predicates_kept": 4,
                    "predicates_collapsed": 1,
                    "predicates_rejected": 0,
                    "predicates_allowlist_rejected": 0,
                    "triples_kept_direction_insensitive": 3,
                    "endpoint_drop_ratio": 0.2,
                    "fact_provenance_max_source_chunks": 50,
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ]

        if "RETURN count(n) AS total_nodes" in text:
            return [{"total_nodes": 1}]

        if "RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id" in text:
            return [{"node_id": "entity-a", "node_type": "entity", "content": None, "entity_name": "A", "entity_type": "Entity"}]

        if "WITH startNode(r) AS s, endNode(r) AS t" in text:
            return []

        if "RETURN count(r)/2 AS total_edges" in text:
            return [{"total_edges": 0}]

        if "MATCH (e:Entity)" in text and "WHERE e.entity_id IN $entity_ids" in text:
            return [{"node_id": "entity-a", "node_type": "entity", "content": None, "entity_name": "A", "entity_type": "Entity"}]

        if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
            return []

        if "MATCH (c:Chunk)" in text:
            return []

        if "MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)" in text:
            return []

        return []


def test_exporter_includes_persisted_kg_ingest_stats_in_metadata() -> None:
    store = _MetaStore()

    full = GraphExporterNeo4j.export_full_graph(
        graph_store=store,
        owner_id="owner-1",
        max_nodes=10,
        max_edges=10,
        include_node_types=["entity"],
    )
    assert full["metadata"]["kg_ingest_stats"]["triples_total"] == 10
    assert full["metadata"]["kg_ingest_stats"]["chunks_total"] == 4
    assert full["metadata"]["kg_ingest_stats"]["predicates_kept"] == 4

    sub = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-a"},
        owner_id="owner-1",
        max_edges=10,
    )
    assert sub["metadata"]["kg_ingest_stats"]["triples_kept"] == 7
