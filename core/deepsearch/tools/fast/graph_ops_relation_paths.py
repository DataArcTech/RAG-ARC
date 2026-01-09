"""Deterministic relation-path exploration/grounding tools backed by Neo4j Cypher."""
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_CYPHER, SCOPE_OWNER

from .graph_ops_common import (
    directionality_config,
    enforce_direction_for_sensitive_predicates,
    enforce_undirected_for_non_sensitive_predicates,
    limit_int,
    normalize_entity_name,
    normalize_predicates,
    rel_pattern_varlen,
)
from core.knowledge_graph.schema import normalize_relation_token
from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema


class GraphRelationPathExploreTool(GraphTool):
    """Enumerate predicate sequences reachable from a seed entity within N hops."""

    descriptor = ToolDescriptor(
        name="graph.relation_path_explore",
        channel="graph",
        description=(
            "Enumerate reachable relation-path predicate sequences from a seed entity (Cypher-backed). "
            "Evidence: citeable samples when `fact_ids/source_chunk_ids` are present in provenance."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("path", "relation_path", "explore", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_relation_path_explore",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "entity": {"type": "string", "description": "Seed entity name (anchor)."},
                "entity_type": {"type": "string", "description": "Optional entity_type for disambiguation."},
                "predicates": {"type": "array", "items": {"type": "string"}, "description": "Optional predicate allow-list."},
                "direction": {"type": "string", "enum": ["out", "in", "both"], "description": "Traversal direction."},
                "max_hops": {"type": "integer", "minimum": 1, "description": "Max hop count (1..5 recommended)."},
                "max_paths": {"type": "integer", "minimum": 1, "description": "Max paths sampled before grouping."},
                "max_sequences": {"type": "integer", "minimum": 1, "description": "Max unique predicate sequences returned."},
            },
            required_extra_fields=("entity",),
        ),
        example_args={
            "question": "What relation patterns are reachable from OpenAI within 2 hops?",
            "plan_step": "plan_01",
            "extra": {"entity": "OpenAI", "max_hops": 2, "max_sequences": 20, "direction": "out"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Relation path exploration requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        entity = normalize_entity_name(request.extra.get("entity"))
        if not entity:
            return ToolResult(summary="relation_path_explore requires a non-empty entity name.")

        predicates = normalize_predicates(request.extra.get("predicates"))
        entity_type = str(request.extra.get("entity_type") or "").strip()
        direction_raw = str(request.extra.get("direction") or "out")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw, predicates, directionality=directionality, default_direction="out"
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, predicates, directionality=directionality
        )
        max_hops = limit_int(request.extra.get("max_hops"), 3, max_value=5)
        max_paths = limit_int(request.extra.get("max_paths"), 200, max_value=2000)
        max_sequences = limit_int(request.extra.get("max_sequences"), 40, max_value=200)

        rel = rel_pattern_varlen(direction, rel_type="RELATES_TO", max_hops=max_hops)
        predicate_filter = (
            "AND ALL(r IN relationships(p) WHERE r.predicate IN $predicates)"
            if predicates
            else ""
        )
        cypher = f"""
        // relation_path_explore
        MATCH (s0:Entity)
        WHERE COALESCE(s0.owner_id, $global_owner) = $owner_id
          AND s0.entity_name_normalized = $entity
          AND ($entity_type = '' OR s0.entity_type = $entity_type)
        WITH collect(s0) AS candidates
        WITH size(candidates) AS candidate_count,
             CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS s
        MATCH p=(s){rel}(t:Entity)
        WHERE s IS NOT NULL
          AND COALESCE(t.owner_id, $global_owner) = $owner_id
          AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
          {predicate_filter}
        WITH candidate_count,
             [r IN relationships(p) | r.predicate] AS predicate_sequence,
             t.entity_name AS target_entity,
             [r IN relationships(p) | r.fact_id] AS fact_ids,
             [r IN relationships(p) | r.source_chunk_ids] AS source_chunk_ids
        LIMIT $max_paths
        WITH candidate_count,
             predicate_sequence,
             collect(DISTINCT target_entity)[0..5] AS targets,
             collect(fact_ids)[0..3] AS fact_ids_samples,
             collect(source_chunk_ids)[0..3] AS source_chunk_ids_samples,
             count(*) AS path_count
        RETURN candidate_count AS candidate_count,
               predicate_sequence AS predicate_sequence,
               targets AS targets,
               fact_ids_samples AS fact_ids_samples,
               source_chunk_ids_samples AS source_chunk_ids_samples,
               path_count AS path_count
        ORDER BY path_count DESC
        LIMIT $max_sequences
        """
        params = {
            "entity": entity,
            "entity_type": entity_type,
            "predicates": predicates,
            "max_paths": max_paths,
            "max_sequences": max_sequences,
        }
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 1)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "relation_path_explore aborted due to ambiguous entity. "
                    f"candidate_count={candidate_count}. Provide entity_type to disambiguate."
                ),
                diagnostics={"entity": entity, "candidate_count": candidate_count, "entity_type": entity_type or None},
            )

        sequences: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            seq = row.get("predicate_sequence")
            if not isinstance(seq, list) or not seq:
                continue
            sequences.append(
                {
                    "predicate_sequence": [str(item) for item in seq if str(item).strip()],
                    "path_count": int(row.get("path_count") or 0),
                    "targets": row.get("targets") if isinstance(row.get("targets"), list) else [],
                    "fact_ids_samples": row.get("fact_ids_samples") if isinstance(row.get("fact_ids_samples"), list) else [],
                    "source_chunk_ids_samples": row.get("source_chunk_ids_samples")
                    if isinstance(row.get("source_chunk_ids_samples"), list)
                    else [],
                }
            )

        summary = f"relation_path_explore: {entity} sequences={len(sequences)} (max_hops={max_hops})"
        content = f"relation_path_explore({entity}): {len(sequences)} sequences"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="relation_paths", content=content)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=content,
            provenance={
                "entity": entity,
                "entity_type": entity_type or None,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "max_hops": max_hops,
                "max_paths": max_paths,
                "max_sequences": max_sequences,
                "allowed_predicates": predicates,
                "relation_paths": sequences,
            },
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"sequence_count": len(sequences)})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphRelationPathExploreTool requires a GraphDeepSearchAdapter instance")
        return adapter


class GraphRelationPathGroundTool(GraphTool):
    """Ground a predicate sequence into concrete paths and frontier entities."""

    descriptor = ToolDescriptor(
        name="graph.relation_path_ground",
        channel="graph",
        description=(
            "Ground a predicate sequence into concrete paths + frontier entities (Cypher-backed). "
            "Evidence: citeable when `fact_ids/source_chunk_ids` are present in provenance."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("path", "relation_path", "ground", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_relation_path_ground",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "source": {"type": "string", "description": "Source entity name."},
                "source_type": {"type": "string", "description": "Optional source entity_type for disambiguation."},
                "predicate_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered predicate sequence to ground (canonical tokens).",
                },
                "direction": {"type": "string", "enum": ["out", "in", "both"], "description": "Traversal direction."},
                "max_paths": {"type": "integer", "minimum": 1, "description": "Max grounded paths returned."},
            },
            required_extra_fields=("source", "predicate_sequence"),
        ),
        example_args={
            "question": "Ground a 2-hop OWNS->OWNS path from A公司",
            "plan_step": "plan_01",
            "extra": {"source": "A公司", "predicate_sequence": ["OWNS", "OWNS"], "max_paths": 5, "direction": "out"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Relation path grounding requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        source = normalize_entity_name(request.extra.get("source"))
        if not source:
            return ToolResult(summary="relation_path_ground requires a non-empty source entity name.")

        raw_seq = request.extra.get("predicate_sequence")
        predicate_sequence = self._normalize_predicate_sequence(raw_seq)
        if not predicate_sequence:
            return ToolResult(summary="relation_path_ground requires a non-empty predicate_sequence.")

        source_type = str(request.extra.get("source_type") or "").strip()
        direction_raw = str(request.extra.get("direction") or "out")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw, predicate_sequence, directionality=directionality, default_direction="out"
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, predicate_sequence, directionality=directionality
        )
        max_paths = limit_int(request.extra.get("max_paths"), 25, max_value=200)
        hops = len(predicate_sequence)

        rel = rel_pattern_varlen(direction, rel_type="RELATES_TO", max_hops=hops)
        cypher = f"""
        // relation_path_ground
        MATCH (s0:Entity)
        WHERE COALESCE(s0.owner_id, $global_owner) = $owner_id
          AND s0.entity_name_normalized = $source
          AND ($source_type = '' OR s0.entity_type = $source_type)
        WITH collect(s0) AS source_nodes
        WITH size(source_nodes) AS candidate_count,
             CASE WHEN size(source_nodes) = 1 THEN source_nodes[0] ELSE NULL END AS s
        MATCH p=(s){rel}(t:Entity)
        WHERE s IS NOT NULL
          AND COALESCE(t.owner_id, $global_owner) = $owner_id
          AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
        WITH candidate_count, p, relationships(p) AS rels
        WHERE size(rels) = $hops
          AND all(i IN range(0, $hops - 1) WHERE rels[i].predicate = $predicates[i])
        RETURN candidate_count AS candidate_count,
               [n IN nodes(p) | n.entity_name] AS nodes,
               [r IN rels | r.predicate] AS predicates,
               [r IN rels | r.fact_id] AS fact_ids,
               [r IN rels | r.source_chunk_ids] AS source_chunk_ids
        LIMIT $max_paths
        """
        params = {
            "source": source,
            "source_type": source_type,
            "predicates": list(predicate_sequence),
            "hops": hops,
            "max_paths": max_paths,
        }
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 1)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "relation_path_ground aborted due to ambiguous source entity. "
                    f"candidate_count={candidate_count}. Provide source_type to disambiguate."
                ),
                diagnostics={"source": source, "candidate_count": candidate_count, "source_type": source_type or None},
            )

        grounded: List[Dict[str, Any]] = []
        frontier: List[str] = []
        seen_frontier: set[str] = set()
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            nodes = row.get("nodes") if isinstance(row.get("nodes"), list) else []
            if nodes:
                leaf = str(nodes[-1])
                if leaf and leaf not in seen_frontier:
                    seen_frontier.add(leaf)
                    frontier.append(leaf)
            grounded.append(
                {
                    "nodes": nodes,
                    "predicates": row.get("predicates") if isinstance(row.get("predicates"), list) else [],
                    "fact_ids": row.get("fact_ids") if isinstance(row.get("fact_ids"), list) else [],
                    "source_chunk_ids": row.get("source_chunk_ids") if isinstance(row.get("source_chunk_ids"), list) else [],
                }
            )

        summary = f"relation_path_ground: {source} grounded_paths={len(grounded)} hops={hops}"
        content = f"relation_path_ground({source}): {len(grounded)} grounded paths"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="grounded_paths", content=content)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=content,
            provenance={
                "source": source,
                "source_type": source_type or None,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "predicate_sequence": list(predicate_sequence),
                "grounded_paths": grounded,
                "frontier_entities": frontier,
            },
        )
        return ToolResult(
            summary=summary,
            evidences=[evidence],
            diagnostics={"grounded_path_count": len(grounded), "frontier_entity_count": len(frontier)},
        )

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphRelationPathGroundTool requires a GraphDeepSearchAdapter instance")
        return adapter

    @staticmethod
    def _normalize_predicate_sequence(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for item in raw:
            token = normalize_relation_token(str(item or ""))
            if token:
                out.append(token)
        return out
