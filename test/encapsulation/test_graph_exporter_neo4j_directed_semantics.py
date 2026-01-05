from typing import Any, Dict, List

from core.knowledge_graph.schema import schema_from_dict
from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j


class _StubNeo4jStore:
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
                    "target_name": "A公司",
                }
            ]
        if "MATCH (c:Chunk)" in text:
            return [{"node_id": "chunk-1", "node_type": "chunk", "content": "c", "entity_name": None, "entity_type": None}]
        if "MATCH (e:Entity)" in text and "WHERE e.entity_id IN $entity_ids" in text:
            return [
                {"node_id": "entity-a", "node_type": "entity", "content": None, "entity_name": "A公司", "entity_type": "Company"},
                {"node_id": "entity-b", "node_type": "entity", "content": None, "entity_name": "B公司", "entity_type": "Company"},
            ]
        if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
            return [
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "rel_type": "RELATES_TO",
                    "weight": 1.0,
                    "predicate": "OWNS",
                    "fact_id": "fact-owns",
                    "source_chunk_ids": [["chunk-1"]],
                    "schema_version": "v1",
                    "domain": "default",
                    "source_type": "entity",
                    "target_type": "entity",
                    "source_name": "A公司",
                    "target_name": "B公司",
                },
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "rel_type": "RELATES_TO",
                    "weight": 1.0,
                    "predicate": "HAS_POLICY",
                    "fact_id": "fact-policy",
                    "source_chunk_ids": [["chunk-1"]],
                    "schema_version": "v1",
                    "domain": "default",
                    "source_type": "entity",
                    "target_type": "entity",
                    "source_name": "A公司",
                    "target_name": "B公司",
                },
            ]
        return []


def test_export_subgraph_marks_directed_edges_only_for_direction_sensitive_predicates() -> None:
    store = _StubNeo4jStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"chunk-1", "entity-a", "entity-b"},
        seed_entity_ids={"entity-a"},
        retrieved_chunk_ids=["chunk-1"],
        node_ppr_scores={},
        owner_id="owner-1",
        directed_edges=False,
        preserve_multi_edges=True,
        max_edges=50,
    )

    facts = [e for e in payload["edges"] if e.get("relation") in {"OWNS", "HAS_POLICY"}]
    assert {e.get("relation") for e in facts} == {"OWNS", "HAS_POLICY"}
    owns = next(e for e in facts if e.get("relation") == "OWNS")
    policy = next(e for e in facts if e.get("relation") == "HAS_POLICY")
    assert owns.get("directed") is True
    assert policy.get("directed") is False


def test_export_subgraph_preserve_multi_edges_controls_deduping() -> None:
    class _DedupStore(_StubNeo4jStore):
        def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
            text = str(query or "")
            if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
                return [
                    {
                        "source_id": "entity-a",
                        "target_id": "entity-b",
                        "rel_type": "RELATES_TO",
                        "weight": 1.0,
                        "predicate": "HAS_FEATURE",
                        "fact_id": "fact-feature",
                        "source_chunk_ids": [["chunk-1"]],
                        "schema_version": "v1",
                        "domain": "default",
                        "source_type": "entity",
                        "target_type": "entity",
                        "source_name": "A公司",
                        "target_name": "B公司",
                    },
                    {
                        "source_id": "entity-a",
                        "target_id": "entity-b",
                        "rel_type": "RELATES_TO",
                        "weight": 1.0,
                        "predicate": "HAS_POLICY",
                        "fact_id": "fact-policy",
                        "source_chunk_ids": [["chunk-1"]],
                        "schema_version": "v1",
                        "domain": "default",
                        "source_type": "entity",
                        "target_type": "entity",
                        "source_name": "A公司",
                        "target_name": "B公司",
                    },
                ]
            return super()._execute_query(query, params)

    store = _DedupStore()
    payload_collapsed = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-a", "entity-b"},
        owner_id="owner-1",
        directed_edges=False,
        preserve_multi_edges=False,
        max_edges=50,
    )
    facts_collapsed = [e for e in payload_collapsed["edges"] if e.get("relation") in {"HAS_FEATURE", "HAS_POLICY"}]
    # When preserve_multi_edges is disabled, non-direction-sensitive predicates between the same node pair collapse.
    assert len(facts_collapsed) == 1

    payload_kept = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-a", "entity-b"},
        owner_id="owner-1",
        directed_edges=False,
        preserve_multi_edges=True,
        max_edges=50,
    )
    facts_kept = [e for e in payload_kept["edges"] if e.get("relation") in {"HAS_FEATURE", "HAS_POLICY"}]
    assert len(facts_kept) == 2


def test_export_subgraph_directed_edges_flag_forces_all_fact_edges_directed() -> None:
    store = _StubNeo4jStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-a", "entity-b"},
        owner_id="owner-1",
        directed_edges=True,
        preserve_multi_edges=True,
        max_edges=50,
    )
    facts = [e for e in payload["edges"] if e.get("relation") in {"OWNS", "HAS_POLICY"}]
    assert facts
    assert all(e.get("directed") is True for e in facts)


def test_export_subgraph_opposite_directions_non_sensitive_collapse_when_undirected() -> None:
    class _OppositeDirStore(_StubNeo4jStore):
        def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
            text = str(query or "")
            if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
                return [
                    {
                        "source_id": "entity-a",
                        "target_id": "entity-b",
                        "rel_type": "RELATES_TO",
                        "weight": 1.0,
                        "predicate": "HAS_FEATURE",
                        "fact_id": "fact-ab",
                        "source_chunk_ids": [["chunk-1"]],
                        "schema_version": "v1",
                        "domain": "default",
                        "source_type": "entity",
                        "target_type": "entity",
                        "source_name": "A公司",
                        "target_name": "B公司",
                    },
                    {
                        "source_id": "entity-b",
                        "target_id": "entity-a",
                        "rel_type": "RELATES_TO",
                        "weight": 1.0,
                        "predicate": "HAS_FEATURE",
                        "fact_id": "fact-ba",
                        "source_chunk_ids": [["chunk-2"]],
                        "schema_version": "v1",
                        "domain": "default",
                        "source_type": "entity",
                        "target_type": "entity",
                        "source_name": "B公司",
                        "target_name": "A公司",
                    },
                ]
            return super()._execute_query(query, params)

    store = _OppositeDirStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-a", "entity-b"},
        owner_id="owner-1",
        directed_edges=False,
        preserve_multi_edges=True,
        max_edges=50,
    )
    feature_edges = [e for e in payload["edges"] if e.get("relation") == "HAS_FEATURE"]
    assert len(feature_edges) == 1


def test_export_subgraph_opposite_directions_keep_when_schema_marks_sensitive() -> None:
    class _OppositeDirSensitiveStore(_StubNeo4jStore):
        def __init__(self):
            super().__init__()
            self.kg_schema = schema_from_dict(
                {
                    "version": "v1",
                    "default_domain": "default",
                    "domains": {"default": {"direction_sensitive_relations": ["OWNS"]}},
                }
            )

        def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
            text = str(query or "")
            if "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)" in text:
                return [
                    {
                        "source_id": "entity-a",
                        "target_id": "entity-b",
                        "rel_type": "RELATES_TO",
                        "weight": 1.0,
                        "predicate": "OWNS",
                        "fact_id": "fact-ab",
                        "source_chunk_ids": [["chunk-1"]],
                        "schema_version": "v1",
                        "domain": "default",
                        "source_type": "entity",
                        "target_type": "entity",
                        "source_name": "A公司",
                        "target_name": "B公司",
                    },
                    {
                        "source_id": "entity-b",
                        "target_id": "entity-a",
                        "rel_type": "RELATES_TO",
                        "weight": 1.0,
                        "predicate": "OWNS",
                        "fact_id": "fact-ba",
                        "source_chunk_ids": [["chunk-2"]],
                        "schema_version": "v1",
                        "domain": "default",
                        "source_type": "entity",
                        "target_type": "entity",
                        "source_name": "B公司",
                        "target_name": "A公司",
                    },
                ]
            return super()._execute_query(query, params)

    store = _OppositeDirSensitiveStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-a", "entity-b"},
        owner_id="owner-1",
        directed_edges=False,
        preserve_multi_edges=False,
        max_edges=50,
    )
    owns_edges = [e for e in payload["edges"] if e.get("relation") == "OWNS"]
    assert len(owns_edges) == 2
    assert all(e.get("directed") is True for e in owns_edges)
