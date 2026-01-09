"""Deterministic rule-check tool backed by Neo4j Cypher."""
from typing import Any, Dict, List, Mapping

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_CYPHER, SCOPE_OWNER
from .graph_ops_common import (
    directionality_config,
    enforce_direction_for_sensitive_predicates,
    enforce_undirected_for_non_sensitive_predicates,
    limit_int,
    normalize_entity_name,
    normalize_predicates,
    rel_pattern,
)


class GraphRuleCheckTool(GraphTool):
    """Deterministic rule checking over fact edges."""

    descriptor = ToolDescriptor(
        name="graph.rule_check",
        channel="graph",
        description=(
            "Deterministic rule checker (AND of edge-existence predicates) backed by Neo4j Cypher. "
            "Evidence: citeable when matches include `fact_id/source_chunk_ids`."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("rules", "deterministic", "compliance", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_rule_check",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "conditions": {
                    "type": "array",
                    "description": "List of required edge patterns (AND semantics).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "head": {"type": "string"},
                            "head_type": {"type": "string", "description": "Optional head entity_type for disambiguation."},
                            "predicate": {"type": "string"},
                            "tail": {"type": "string"},
                            "tail_type": {"type": "string", "description": "Optional tail entity_type for disambiguation."},
                            "direction": {"type": "string", "enum": ["out", "in", "both"]},
                        },
                        "required": ["head", "predicate", "tail"],
                    },
                },
                "limit": {"type": "integer", "minimum": 1, "description": "Max evidence facts per condition."},
            },
            required_extra_fields=("conditions",),
        ),
        example_args={
            "question": "Check compliance rule: A OWNS B and B OWNS C",
            "plan_step": "plan_10",
            "extra": {
                "conditions": [
                    {"head": "A公司", "predicate": "OWNS", "tail": "B公司", "direction": "out"},
                    {"head": "B公司", "predicate": "OWNS", "tail": "C公司", "direction": "out"},
                ],
                "limit": 5,
            },
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Rule check requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        conditions = request.extra.get("conditions") or []
        if not isinstance(conditions, list) or not conditions:
            return ToolResult(summary="Rule check requires a non-empty conditions list.")
        limit = limit_int(request.extra.get("limit"), 5, max_value=50)

        results: List[Dict[str, Any]] = []
        evidences: List[EvidenceChunk] = []
        all_ok = True
        valid_conditions = 0
        invalid_indices: List[int] = []

        async with adapter_locked(adapter):
            directionality = directionality_config(adapter)
            for idx, cond in enumerate(conditions):
                if not isinstance(cond, Mapping):
                    invalid_indices.append(idx)
                    all_ok = False
                    continue
                head = normalize_entity_name(cond.get("head"))
                head_type = str(cond.get("head_type") or "").strip()
                tail = normalize_entity_name(cond.get("tail"))
                tail_type = str(cond.get("tail_type") or "").strip()
                predicate_list = normalize_predicates(cond.get("predicate"))
                predicate = predicate_list[0] if predicate_list else None
                direction_raw = str(cond.get("direction") or "out")
                direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
                    direction_raw,
                    [predicate] if predicate else [],
                    directionality=directionality,
                    default_direction="out",
                )
                direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
                    direction,
                    [predicate] if predicate else [],
                    directionality=directionality,
                )
                if not head or not tail or not predicate:
                    invalid_indices.append(idx)
                    all_ok = False
                    continue
                valid_conditions += 1

                rel = rel_pattern(direction, rel_var="r", rel_type="RELATES_TO")
                cypher = f"""
                MATCH (h0:Entity)
                WHERE COALESCE(h0.owner_id, $global_owner) = $owner_id
                  AND h0.entity_name_normalized = $head
                  AND ($head_type = '' OR h0.entity_type = $head_type)
                WITH collect(h0) AS head_nodes
                WITH size(head_nodes) AS head_candidates,
                     CASE WHEN size(head_nodes) = 1 THEN head_nodes[0] ELSE NULL END AS h
                MATCH (t0:Entity)
                WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
                  AND t0.entity_name_normalized = $tail
                  AND ($tail_type = '' OR t0.entity_type = $tail_type)
                WITH head_candidates, h, collect(t0) AS tail_nodes
                WITH head_candidates,
                     size(tail_nodes) AS tail_candidates,
                     h,
                     CASE WHEN size(tail_nodes) = 1 THEN tail_nodes[0] ELSE NULL END AS t
                OPTIONAL MATCH (h){rel}(t)
                WHERE head_candidates = 1
                  AND tail_candidates = 1
                  AND COALESCE(r.owner_id, $global_owner) = $owner_id
                  AND r.predicate = $predicate
                RETURN head_candidates AS head_candidates,
                       tail_candidates AS tail_candidates,
                       r.fact_id AS fact_id,
                       r.source_chunk_ids AS source_chunk_ids,
                       r.text AS text
                LIMIT $limit
                """
                rows = await adapter.acypher(
                    cypher,
                    {
                        "head": head,
                        "tail": tail,
                        "predicate": predicate,
                        "limit": limit,
                        "head_type": head_type,
                        "tail_type": tail_type,
                    },
                    access_scope=request.access_scope,
                )
                row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
                head_candidates = int((row0 or {}).get("head_candidates") or 1)
                tail_candidates = int((row0 or {}).get("tail_candidates") or 1)
                if head_candidates != 1 or tail_candidates != 1:
                    ok = False
                    rows = []
                else:
                    ok = any(bool((row or {}).get("fact_id")) for row in (rows or []))
                all_ok = all_ok and ok
                results.append(
                    {
                        "condition": {
                            "head": head,
                            "head_type": head_type or None,
                            "predicate": predicate,
                            "tail": tail,
                            "tail_type": tail_type or None,
                            "direction": direction,
                            "direction_forced_sensitive": forced_sensitive,
                            "direction_forced_undirected": forced_undirected,
                            "head_candidates": head_candidates,
                            "tail_candidates": tail_candidates,
                        },
                        "ok": ok,
                        "matches": rows or [],
                    }
                )
                content = f"rule_condition[{idx}] ok={ok}: {head} -[{predicate}]-> {tail}"
                chunk_id = derived_chunk_id(
                    tool_name=self.descriptor.name, plan_step=request.plan_step, label=f"cond_{idx}", content=content
                )
                evidences.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        source=self.descriptor.name,
                        content=content,
                        provenance={"condition": results[-1]["condition"], "matches": rows or []},
                    )
                )

        if valid_conditions == 0:
            return ToolResult(
                summary="Rule check FAILED (no valid conditions to evaluate).",
                evidences=evidences,
                diagnostics={"ok": False, "results": results, "invalid_indices": invalid_indices},
            )

        summary = "Rule check PASSED (all conditions satisfied)." if all_ok else "Rule check FAILED (some conditions not satisfied)."
        return ToolResult(
            summary=summary,
            evidences=evidences,
            diagnostics={"ok": all_ok, "results": results, "invalid_indices": invalid_indices},
        )

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphRuleCheckTool requires a GraphDeepSearchAdapter instance")
        return adapter
