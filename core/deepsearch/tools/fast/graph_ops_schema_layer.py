"""Deterministic tools over schema-layer nodes (Neo4j Cypher).

Schema-layer nodes are derived from HippoRAG2 mindmap extraction:
- [concept] / [process] / [instance] tags in mindmap TSV lines
- persisted as (:SchemaNode) with per-chunk (:Chunk)-[:HAS_SCHEMA_NODE]->(:SchemaNode)

This tool makes the schema-layer queryable inside DeepSearch (audit/explainability).
"""

from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_CYPHER, SCOPE_OWNER
from .graph_ops_common import limit_int


class GraphSchemaNodesTool(GraphTool):
    """Fetch schema-layer nodes linked to a chunk, or search by term."""

    descriptor = ToolDescriptor(
        name="graph.schema_nodes",
        channel="graph",
        description=(
            "Deterministic schema-layer lookup (mindmap-derived Concept/Process/Instance scaffolding) backed by Neo4j Cypher. "
            "Supports chunk→SchemaNode expansion and term search over SchemaNode.text_normalized."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("schema", "mindmap", "concept", "process", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_schema_nodes",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "chunk_id": {"type": "string", "description": "Chunk id to expand into schema nodes (optional)."},
                "term": {"type": "string", "description": "Search term (optional)."},
                "layer": {"type": "string", "description": "Optional layer filter: concept/process/instance/unknown."},
                "limit": {"type": "integer", "minimum": 1, "description": "Max nodes returned."},
            },
            required_extra_fields=(),
        ),
        example_args={
            "question": "What schema nodes mention 'risk'?",
            "plan_step": "plan_08",
            "extra": {"term": "risk", "layer": "concept", "limit": 20},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Schema-node lookup requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        chunk_id = str(request.extra.get("chunk_id") or "").strip()
        term = str(request.extra.get("term") or "").strip()
        layer = str(request.extra.get("layer") or "").strip().lower()
        if layer not in {"concept", "process", "instance", "unknown", ""}:
            layer = ""
        limit = limit_int(request.extra.get("limit"), 50, max_value=300)

        if not chunk_id and not term:
            return ToolResult(summary="Schema-node lookup requires chunk_id or term.")

        layer_clause = "AND s.layer = $layer" if layer else ""

        if chunk_id:
            cypher = f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})
            WHERE COALESCE(c.owner_id, $global_owner) = $owner_id
            MATCH (c)-[r:HAS_SCHEMA_NODE]->(s:SchemaNode)
            WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
              AND COALESCE(s.owner_id, $global_owner) = $owner_id
              {layer_clause}
            RETURN s.schema_id AS schema_id,
                   s.layer AS layer,
                   s.text AS text,
                   s.text_normalized AS text_normalized,
                   r.level AS level
            ORDER BY r.level ASC, s.text_normalized ASC
            LIMIT $limit
            """
            params = {"chunk_id": chunk_id, "layer": layer, "limit": limit}
        else:
            cypher = f"""
            MATCH (s:SchemaNode)
            WHERE COALESCE(s.owner_id, $global_owner) = $owner_id
              AND s.text_normalized CONTAINS $term
              {layer_clause}
            RETURN s.schema_id AS schema_id,
                   s.layer AS layer,
                   s.text AS text,
                   s.text_normalized AS text_normalized,
                   NULL AS level
            ORDER BY s.text_normalized ASC
            LIMIT $limit
            """
            params = {"term": term.lower(), "layer": layer, "limit": limit}

        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)

        nodes: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            nodes.append(
                {
                    "schema_id": row.get("schema_id"),
                    "layer": row.get("layer"),
                    "text": text,
                    "text_normalized": row.get("text_normalized"),
                    "level": row.get("level"),
                }
            )

        summary = f"schema_nodes: count={len(nodes)}"
        label = "chunk" if chunk_id else "term"
        chunk = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label=label, content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk,
            source=self.descriptor.name,
            content=summary,
            provenance={"chunk_id": chunk_id or None, "term": term or None, "layer": layer or None, "nodes": nodes},
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"count": len(nodes)})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphSchemaNodesTool requires a GraphDeepSearchAdapter instance")
        return adapter
