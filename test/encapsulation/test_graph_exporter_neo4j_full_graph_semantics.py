from typing import Any, Dict, List

from core.knowledge_graph.schema import schema_from_dict
from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j


class _RecorderStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self, *, direction_sensitive: list[str]):
        self.queries: list[str] = []
        self.kg_schema = schema_from_dict(
            {
                "version": "v1",
                "default_domain": "default",
                "domains": {"default": {"direction_sensitive_relations": direction_sensitive}},
            }
        )
        self.config = type("_Cfg", (), {"kg_schema_path": "./kg_schema.yml"})()

    def _owner_key(self, owner_id: str) -> str:
        return str(owner_id)

    def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        text = str(query or "")
        self.queries.append(text)

        if "RETURN count(n) AS total_nodes" in text:
            return [{"total_nodes": 3}]

        if "RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id" in text:
            return [
                {"node_id": "chunk-1", "node_type": "chunk", "content": "c", "entity_name": None, "entity_type": None},
                {"node_id": "entity-a", "node_type": "entity", "content": None, "entity_name": "A", "entity_type": "Entity"},
                {"node_id": "entity-b", "node_type": "entity", "content": None, "entity_name": "B", "entity_type": "Entity"},
            ]

        if "WITH startNode(r) AS s, endNode(r) AS t" in text:
            return self._edge_rows()

        if "RETURN count(r)/2 AS total_edges" in text:
            return [{"total_edges": 2}]

        return []

    def _edge_rows(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


def test_export_full_graph_uses_startnode_endnode_for_orientation() -> None:
    class _Store(_RecorderStore):
        def _edge_rows(self):
            return []

    store = _Store(direction_sensitive=[])
    GraphExporterNeo4j.export_full_graph(
        graph_store=store,
        owner_id="owner-1",
        max_nodes=10,
        max_edges=10,
        include_node_types=["chunk", "entity"],
        directed_edges=False,
    )
    joined = "\n".join(store.queries)
    assert "WITH startNode(r) AS s, endNode(r) AS t" in joined


def test_export_full_graph_preserve_multi_edges_for_undirected_predicates() -> None:
    class _Store(_RecorderStore):
        def _edge_rows(self):
            return [
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "rel_type": "RELATES_TO",
                    "weight": 1.0,
                    "predicate": "HAS_FEATURE",
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

    store = _Store(direction_sensitive=[])

    collapsed = GraphExporterNeo4j.export_full_graph(
        graph_store=store,
        owner_id="owner-1",
        max_nodes=10,
        max_edges=50,
        include_node_types=["entity"],
        directed_edges=False,
        preserve_multi_edges=False,
    )
    facts_collapsed = [e for e in collapsed["edges"] if e.get("relation") in {"HAS_FEATURE", "HAS_POLICY"}]
    assert len(facts_collapsed) == 1
    assert facts_collapsed[0].get("directed") is False

    kept = GraphExporterNeo4j.export_full_graph(
        graph_store=store,
        owner_id="owner-1",
        max_nodes=10,
        max_edges=50,
        include_node_types=["entity"],
        directed_edges=False,
        preserve_multi_edges=True,
    )
    facts_kept = [e for e in kept["edges"] if e.get("relation") in {"HAS_FEATURE", "HAS_POLICY"}]
    assert len(facts_kept) == 2
    assert all(e.get("directed") is False for e in facts_kept)


def test_export_full_graph_direction_sensitive_predicates_are_not_collapsed() -> None:
    class _Store(_RecorderStore):
        def _edge_rows(self):
            return [
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "rel_type": "RELATES_TO",
                    "weight": 1.0,
                    "predicate": "OWNS",
                    "fact_id": "fact-owns-1",
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
                    "predicate": "OWNS",
                    "fact_id": "fact-owns-2",
                    "source_chunk_ids": [["chunk-2"]],
                    "schema_version": "v1",
                    "domain": "default",
                    "source_type": "entity",
                    "target_type": "entity",
                    "source_name": "A",
                    "target_name": "B",
                },
            ]

    store = _Store(direction_sensitive=["OWNS"])
    payload = GraphExporterNeo4j.export_full_graph(
        graph_store=store,
        owner_id="owner-1",
        max_nodes=10,
        max_edges=50,
        include_node_types=["entity"],
        directed_edges=False,
        preserve_multi_edges=False,
    )
    owns_edges = [e for e in payload["edges"] if e.get("relation") == "OWNS"]
    assert len(owns_edges) == 2
    assert all(e.get("directed") is True for e in owns_edges)

