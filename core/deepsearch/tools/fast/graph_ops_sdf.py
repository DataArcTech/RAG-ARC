"""Deterministic tools over SDF schema nodes (Neo4j Cypher)."""

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema

from .graph_ops_common import normalize_entity_name


class GraphSdfChildrenTool(GraphTool):
    """Fetch SDF subevents (children) for a given event."""

    descriptor = ToolDescriptor(
        name="graph.sdf_children",
        channel="graph",
        description="Deterministic retrieval of SDF subevents (children_gate + children importance) backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("sdf", "process_schema", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_sdf_children",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "event": {"type": "string", "description": "Event name."},
                "doc_namespace": {"type": "string", "description": "Optional document namespace to disambiguate."},
                "limit": {"type": "integer", "description": "Max children to return.", "minimum": 1, "maximum": 200},
            },
            required_extra_fields=("event",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="SDF children requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        event = normalize_entity_name(request.extra.get("event"))
        if not event:
            return ToolResult(summary="SDF children requires a non-empty event name.")

        doc_namespace = str(request.extra.get("doc_namespace") or "").strip()
        limit = int(request.extra.get("limit") or 50)
        limit = max(1, min(200, limit))

        cypher = """
        MATCH (t0:SDFEvent)
        WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
          AND t0.name_normalized = $event
          AND ($doc_namespace = '' OR t0.doc_namespace = $doc_namespace)
        WITH collect(t0) AS candidates
        WITH size(candidates) AS candidate_count,
             CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS t
        MATCH (t)-[r:SDF_HAS_SUBEVENT]->(c:SDFEvent)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
          AND COALESCE(c.owner_id, $global_owner) = $owner_id
        RETURN candidate_count AS candidate_count,
               t.children_gate AS gate,
               c.name AS child,
               c.sdf_event_id AS child_event_id,
               r.importance AS importance,
               r.source_chunk_ids AS source_chunk_ids
        ORDER BY importance DESC, child ASC
        LIMIT $limit
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"event": event, "doc_namespace": doc_namespace, "limit": limit},
                access_scope=request.access_scope,
            )

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 0)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "SDF children aborted due to ambiguous or missing event name. "
                    f"candidate_count={candidate_count}. Provide doc_namespace to disambiguate."
                ),
                diagnostics={"event": event, "candidate_count": candidate_count, "doc_namespace": doc_namespace or None},
            )

        gate = str((row0 or {}).get("gate") or "").strip() or None
        children = []
        source_chunk_ids: list[str] = []
        for row in rows or []:
            child = str((row or {}).get("child") or "").strip()
            if not child:
                continue
            children.append(
                {
                    "child": child,
                    "child_event_id": (row or {}).get("child_event_id"),
                    "importance": (row or {}).get("importance"),
                }
            )
            source_chunk_ids.extend(list((row or {}).get("source_chunk_ids") or []))

        summary = f"sdf_children: event={event} gate={gate or 'unknown'} children={len(children)}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="children", content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            provenance={
                "event": event,
                "doc_namespace": doc_namespace or None,
                "gate": gate,
                "children": children,
                "source_chunk_ids": sorted({c for c in source_chunk_ids if c}),
            },
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"children": children, "gate": gate})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphSdfChildrenTool requires a GraphDeepSearchAdapter instance")
        return adapter


class GraphSdfDependenciesTool(GraphTool):
    """Fetch immediate BEFORE dependencies around an SDF event."""

    descriptor = ToolDescriptor(
        name="graph.sdf_dependencies",
        channel="graph",
        description="Deterministic retrieval of SDF before/after neighbors backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("sdf", "temporal", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_sdf_dependencies",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "event": {"type": "string", "description": "Event name."},
                "doc_namespace": {"type": "string", "description": "Optional document namespace to disambiguate."},
                "limit": {"type": "integer", "description": "Max neighbors to return per side.", "minimum": 1, "maximum": 200},
            },
            required_extra_fields=("event",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="SDF dependencies requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        event = normalize_entity_name(request.extra.get("event"))
        if not event:
            return ToolResult(summary="SDF dependencies requires a non-empty event name.")

        doc_namespace = str(request.extra.get("doc_namespace") or "").strip()
        limit = int(request.extra.get("limit") or 50)
        limit = max(1, min(200, limit))

        cypher = """
        MATCH (t0:SDFEvent)
        WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
          AND t0.name_normalized = $event
          AND ($doc_namespace = '' OR t0.doc_namespace = $doc_namespace)
        WITH collect(t0) AS candidates
        WITH size(candidates) AS candidate_count,
             CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS t
        OPTIONAL MATCH (prev:SDFEvent)-[rb:SDF_BEFORE]->(t)
        WHERE COALESCE(rb.owner_id, $global_owner) = $owner_id
          AND COALESCE(prev.owner_id, $global_owner) = $owner_id
        WITH t, candidate_count,
             [x IN collect(DISTINCT CASE
                 WHEN prev IS NULL THEN NULL
                 ELSE {name: prev.name, event_id: prev.sdf_event_id, source_chunk_ids: rb.source_chunk_ids}
             END) WHERE x IS NOT NULL][..$limit] AS before_list
        OPTIONAL MATCH (t)-[ra:SDF_BEFORE]->(nxt:SDFEvent)
        WHERE COALESCE(ra.owner_id, $global_owner) = $owner_id
          AND COALESCE(nxt.owner_id, $global_owner) = $owner_id
        RETURN candidate_count AS candidate_count,
               before_list AS before,
               [x IN collect(DISTINCT CASE
                 WHEN nxt IS NULL THEN NULL
                 ELSE {name: nxt.name, event_id: nxt.sdf_event_id, source_chunk_ids: ra.source_chunk_ids}
               END) WHERE x IS NOT NULL][..$limit] AS after
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"event": event, "doc_namespace": doc_namespace, "limit": limit},
                access_scope=request.access_scope,
            )

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 0)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "SDF dependencies aborted due to ambiguous or missing event name. "
                    f"candidate_count={candidate_count}. Provide doc_namespace to disambiguate."
                ),
                diagnostics={"event": event, "candidate_count": candidate_count, "doc_namespace": doc_namespace or None},
            )

        before = list((row0 or {}).get("before") or [])
        after = list((row0 or {}).get("after") or [])

        summary = f"sdf_dependencies: event={event} before={len(before)} after={len(after)}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="deps", content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            provenance={"event": event, "doc_namespace": doc_namespace or None, "before": before, "after": after},
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"before": before, "after": after})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphSdfDependenciesTool requires a GraphDeepSearchAdapter instance")
        return adapter
