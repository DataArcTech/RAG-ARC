from typing import Any, Dict, List

import encapsulation.database.utils.graph_export_utils_neo4j as export_mod
from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j
from core.knowledge_graph.schema import schema_from_dict


class _StubStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self):
        self.config = type("_Cfg", (), {"kg_schema_path": "./kg_schema.yml"})()
        self.kg_schema = schema_from_dict({"version": "v1", "default_domain": "default", "domains": {"default": {}}})

    def _owner_key(self, owner_id: str) -> str:  # noqa: D401
        return str(owner_id)

    def _execute_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:  # noqa: ARG002
        text = str(query or "")
        if "MATCH (e:Entity)" in text and "RETURN e.entity_id AS node_id" in text:
            return [
                {
                    "node_id": "entity-num",
                    "node_type": "entity",
                    "content": None,
                    "metadata": None,
                    "entity_name": "2024-06-01",
                    "entity_type": "Date",
                }
            ]
        if "MATCH (m:KGIngestMeta" in text:
            return []
        return []


def test_export_subgraph_filters_numeric_time_entities_by_default() -> None:
    store = _StubStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-num"},
        owner_id="owner-1",
        max_edges=0,
    )
    assert payload["nodes"] == []


def test_export_subgraph_can_include_numeric_time_entities(monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "GRAPH_EXPORT_FILTER_NUMERIC_TIME_ENTITIES", False)
    store = _StubStore()
    payload = GraphExporterNeo4j.export_subgraph(
        graph_store=store,
        subgraph_node_ids={"entity-num"},
        owner_id="owner-1",
        max_edges=0,
    )
    assert payload["nodes"]
    assert payload["nodes"][0]["name"] == "2024-06-01"

