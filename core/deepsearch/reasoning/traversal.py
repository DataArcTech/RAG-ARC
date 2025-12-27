"""
Graph traversal executor used by DeepSearch.

Runs planner steps through GraphDeepSearchAdapter (prepare → query → filter → summarize → chain traversal)
and returns traversal/evidence/reasoning records for downstream gap detection and reporting.
Keeps the adapter abstraction swappable so semantic or relational strategies can be configured per run.
"""
import logging
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from encapsulation.data_model.deepsearch import (
    EvidenceChunk,
    GraphQueryContext,
    GraphTraversalRecord,
    PlanSpec,
    ReasoningStepRecord,
)
from core.graph_adapter.base import GraphDeepSearchAdapter
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.trace import emit_trace
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.utils.query_clean import clean_query
from core.utils.json_safe import json_safe

logger = logging.getLogger(__name__)


@dataclass
class GraphTraversalSettings:
    """Strategy configuration used by the traversal executor."""

    strategy_name: str
    allow_semantic_channel: bool
    chain_depth: int
    parallel_branches: int
    step_summary_max_chars: int


class GraphTraversalExecutor:
    """Run GraphDeepSearchAdapter traversals according to Planner output."""

    def __init__(self, adapter: GraphDeepSearchAdapter, settings: GraphTraversalSettings | None = None):
        self.adapter = adapter
        if settings is None:
            raise ValueError("GraphTraversalExecutor requires explicit settings (no implicit defaults).")
        self.settings = settings

    async def prepare(self, context: GraphQueryContext) -> None:
        """Ensure the adapter is warmed up for the provided context."""

        scope = context.resolve_scope()
        async with adapter_locked(self.adapter):
            await self.adapter.prepare(context.question or "", access_scope=scope)

    async def run(
        self,
        plan_steps: Sequence[PlanSpec],
        context: GraphQueryContext,
        *,
        tool_args_map: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_name: str,
    ) -> Tuple[List[GraphTraversalRecord], List[ReasoningStepRecord], List[EvidenceChunk]]:
        """Execute plan steps against the graph adapter."""

        await self.prepare(context)
        traversals: List[GraphTraversalRecord] = []
        reasoning_steps: List[ReasoningStepRecord] = []
        evidences: List[EvidenceChunk] = []

        for step in plan_steps:
            traversal_record, reasoning_entry, step_evidences = await self.run_step(
                step,
                context,
                tool_args=tool_args_map.get(step.step_id) if tool_args_map else None,
                tool_name=tool_name,
            )
            if traversal_record:
                traversals.append(traversal_record)
            reasoning_steps.append(reasoning_entry)
            evidences.extend(step_evidences)

        return traversals, reasoning_steps, evidences

    async def run_step(
        self,
        step: PlanSpec,
        context: GraphQueryContext,
        *,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_name: str,
    ) -> Tuple[Optional[GraphTraversalRecord], ReasoningStepRecord, List[EvidenceChunk]]:
        """Execute a single plan step so reasoning can interleave with other channels."""

        resolved_tool = str(tool_name or "").strip()
        if not resolved_tool:
            raise ValueError("tool_name is required for GraphTraversalExecutor.run_step")

        scope = context.resolve_scope()
        reasoning_entry = ReasoningStepRecord(
            step_id=step.step_id,
            description=step.description,
            channel=step.channel,
            status="running",
        )
        evidences: List[EvidenceChunk] = []
        traversal_record: Optional[GraphTraversalRecord] = None
        call_id = uuid.uuid4().hex
        start = time.perf_counter()
        try:
            merged_seed_entities = self._merge_seed_entities(context, tool_args)
            query = self._resolve_query(step.description, tool_args)

            await emit_trace(
                "tool_call",
                json.dumps(
                    json_safe(
                        {
                            "call_id": call_id,
                            "tool_name": resolved_tool,
                            "plan_step": step.step_id,
                            "channel": step.channel,
                            "query": query,
                            "seed_entities": merged_seed_entities,
                            "settings": {
                                "strategy": self.settings.strategy_name,
                                "chain_depth": self.settings.chain_depth,
                                "allow_semantic_channel": bool(self.settings.allow_semantic_channel),
                            },
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                meta={"call_id": call_id, "tool_name": resolved_tool, "plan_step": step.step_id},
            )

            async with adapter_locked(self.adapter):
                subgraph = await self.adapter.aquery_subgraph(
                    query,
                    channel=step.channel,
                    access_scope=scope,
                )
                filter_type = "semantic" if self.settings.allow_semantic_channel else "relational"
                filtered = await self.adapter.context_filter(
                    subgraph,
                    filter_type=filter_type,
                    access_scope=scope,
                )
                summary = await self.adapter.summarize(step.channel, filtered, access_scope=scope)
                chain_payload = {
                    "strategy": self.settings.strategy_name,
                    "max_depth": self.settings.chain_depth,
                    "plan_step": step.step_id,
                    "question": context.question,
                }
                if merged_seed_entities:
                    chain_payload["seed_entities"] = merged_seed_entities
                if tool_args:
                    chain_payload["tool_args"] = tool_args
                chain_result = await self.adapter.chain_traverse(chain_payload, access_scope=scope)

            latency_ms = int((time.perf_counter() - start) * 1000)

            summary_text = summary if isinstance(summary, str) else str(summary)
            summary_limit = max(0, int(self.settings.step_summary_max_chars))
            if summary_limit > 0 and len(summary_text) > summary_limit:
                summary_text = summary_text[: max(0, summary_limit - 3)].rstrip() + "..."
            subgraph_info = self._extract_subgraph_info(filtered, subgraph)
            triples = self._extract_triples(filtered, subgraph)
            adapter_source = getattr(getattr(self.adapter, "metadata", lambda: None)(), "adapter_name", None)  # type: ignore[misc]
            source = str(adapter_source or context.adapter_name or "graph").strip() or "graph"
            chunks = self._extract_chunks(filtered, subgraph)
            evidences = self._chunks_to_evidences(
                chunks,
                source=source,
                plan_step=step.step_id,
                query=query,
                triples=triples,
                subgraph_info=subgraph_info,
                filter_type=filter_type,
                chain_result=chain_result,
                tool_args=tool_args,
                latency_ms=latency_ms,
            )

            traversal_record = GraphTraversalRecord(
                step_id=step.step_id,
                strategy=self.settings.strategy_name,
                hop_count=self._infer_hops(filtered, chain_result),
                visited_nodes=self._collect_nodes(filtered),
                visited_edges=self._collect_edges(filtered),
                seed_entities=merged_seed_entities,
                retrieved_chunks=[ev.chunk_id for ev in evidences],
                metadata={
                    "channel": step.channel,
                    "filter_type": filter_type,
                    "tool_args": tool_args or {},
                    "latency_ms": latency_ms,
                },
            )

            reasoning_entry.status = "done"
            reasoning_entry.output_summary = summary_text
            reasoning_entry.diagnostics.setdefault("latency_ms", latency_ms)
            if evidences:
                reasoning_entry.produced_evidence_ids.extend([ev.chunk_id for ev in evidences])
            await emit_trace(
                "tool_response",
                json.dumps(
                    json_safe(
                        {
                            "call_id": call_id,
                            "tool_name": resolved_tool,
                            "plan_step": step.step_id,
                            "channel": step.channel,
                            "query": query,
                            "latency_ms": latency_ms,
                            "summary": summary_text,
                            "traversal": (traversal_record.model_dump(exclude_none=True) if traversal_record else None),
                            "evidences": [ev.model_dump(exclude_none=True) for ev in evidences],
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                meta={
                    "call_id": call_id,
                    "tool_name": resolved_tool,
                    "plan_step": step.step_id,
                    "ok": True,
                    "evidence_count": len(evidences),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Graph traversal failed for %s: %s", step.step_id, exc)
            reasoning_entry.status = "failed"
            reasoning_entry.diagnostics["error"] = str(exc)
            await emit_trace(
                "tool_response",
                json.dumps(
                    json_safe(
                        {
                            "call_id": call_id,
                            "tool_name": resolved_tool,
                            "plan_step": step.step_id,
                            "error": str(exc),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                meta={"call_id": call_id, "tool_name": resolved_tool, "plan_step": step.step_id, "ok": False},
            )
        return traversal_record, reasoning_entry, evidences

    @staticmethod
    def _resolve_query(description: str, tool_args: Optional[Dict[str, Any]]) -> str:
        if tool_args and isinstance(tool_args.get("query"), str):
            return clean_query(tool_args["query"], max_chars=360) or tool_args["query"]
        return clean_query(description, max_chars=360) or description

    @staticmethod
    def _merge_seed_entities(
        context: GraphQueryContext,
        tool_args: Optional[Dict[str, Any]],
    ) -> List[str]:
        seeds: List[str] = []

        def _extend(values: Optional[List[str]]) -> None:
            if not values:
                return
            for value in values:
                token = str(value).strip()
                if token and token not in seeds:
                    seeds.append(token)

        _extend(getattr(context, "seed_entities", []) or [])
        metadata = getattr(context, "metadata", {}) or {}
        planner_hints = metadata.get("planner_hints") or {}
        if isinstance(planner_hints, dict):
            _extend(planner_hints.get("seed_entities"))
        if tool_args:
            candidate_seeds = tool_args.get("seed_entities")
            if isinstance(candidate_seeds, list):
                _extend(candidate_seeds)
        return seeds

    @staticmethod
    def _collect_nodes(payload: Dict[str, object]) -> List[str]:
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(nodes, list):
            return []
        result = []
        for node in nodes:
            if isinstance(node, dict):
                result.append(str(node.get("id") or node.get("name")))
        return [n for n in result if n]

    @staticmethod
    def _collect_edges(payload: Dict[str, object]) -> List[str]:
        edges = payload.get("edges") if isinstance(payload, dict) else None
        if not isinstance(edges, list):
            return []
        result = []
        for edge in edges:
            if isinstance(edge, dict):
                result.append(str(edge.get("id") or edge.get("relation")))
        return [e for e in result if e]

    @staticmethod
    def _infer_hops(filtered: Dict[str, object], chain_result: Dict[str, object]) -> int:
        hops = chain_result.get("hops") if isinstance(chain_result, dict) else None
        if isinstance(hops, int):
            return hops
        edges = filtered.get("edges") if isinstance(filtered, dict) else None
        if isinstance(edges, list):
            return len(edges)
        return 0

    @staticmethod
    def _extract_subgraph_info(filtered: Any, subgraph: Any) -> Optional[Dict[str, Any]]:
        """Extract a compact _subgraph_info payload for downstream evidence rendering."""

        candidates: List[Any] = []
        if isinstance(filtered, dict):
            candidates.append((filtered.get("metadata") or {}).get("subgraph_info"))
        if isinstance(subgraph, dict):
            candidates.append((subgraph.get("metadata") or {}).get("subgraph_info"))
        for candidate in candidates:
            if isinstance(candidate, dict):
                return {
                    key: candidate.get(key)
                    for key in (
                        "subgraph_nodes",
                        "seed_entity_ids",
                        "retrieved_chunk_ids",
                        "node_ppr_scores",
                    )
                    if candidate.get(key) is not None
                }
        return None

    @staticmethod
    def _extract_chunks(filtered: Any, subgraph: Any) -> List[Dict[str, Any]]:
        """Prefer filtered chunks (post filter) and fall back to raw subgraph chunks."""

        for payload in (filtered, subgraph):
            if isinstance(payload, dict):
                chunks = payload.get("chunks")
                if isinstance(chunks, list):
                    return [chunk for chunk in chunks if isinstance(chunk, dict)]
        return []

    @staticmethod
    def _chunks_to_evidences(
        chunks: List[Dict[str, Any]],
        *,
        source: str,
        plan_step: str,
        query: str,
        triples: List[Dict[str, str]],
        subgraph_info: Optional[Dict[str, Any]],
        filter_type: str,
        chain_result: Any,
        tool_args: Optional[Dict[str, Any]],
        latency_ms: int,
    ) -> List[EvidenceChunk]:
        """Convert adapter-returned chunks into EvidenceChunk objects (chunk-first evidence)."""

        evidences: List[EvidenceChunk] = []
        seen: set[str] = set()

        for chunk in chunks:
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            chunk_id = GraphTraversalExecutor._extract_chunk_id(chunk, source=source, content=content)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            score = chunk.get("score")
            if score is None and isinstance(chunk.get("metadata"), dict):
                score = chunk["metadata"].get("score")
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=source,
                    content=content,
                    score=score,
                    provenance={
                        "plan_step": plan_step,
                        "query": query,
                        "triples": triples,
                        "metadata": {
                            "_subgraph_info": subgraph_info,
                            "filter_type": filter_type,
                            "chain_result": GraphTraversalExecutor._compact_chain_result(chain_result),
                            "tool_args": tool_args or {},
                            "latency_ms": latency_ms,
                            "chunk_metadata": (chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}),
                        },
                    },
                )
            )
        return evidences

    @staticmethod
    def _extract_chunk_id(chunk: Dict[str, Any], *, source: str, content: str) -> str:
        candidates: List[Any] = []
        for key in ("id", "chunk_id", "chunkId"):
            candidates.append(chunk.get(key))
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            for key in ("chunk_id", "chunkId", "id"):
                candidates.append(metadata.get(key))
        for candidate in candidates:
            token = str(candidate or "").strip()
            if token:
                return token
        return hashed_chunk_id(source=source, content=content)

    @staticmethod
    def _extract_triples(filtered: Any, subgraph: Any) -> List[Dict[str, str]]:
        """Normalize adapter edge exports into head/relation/tail triples."""

        candidates: List[Any] = []
        if isinstance(filtered, dict):
            candidates.append(filtered)
        if isinstance(subgraph, dict):
            candidates.append(subgraph)

        triples: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for payload in candidates:
            edges = payload.get("edges") if isinstance(payload, dict) else None
            if not isinstance(edges, list):
                continue
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                relation = str(edge.get("relation") or "").strip()
                if not relation or relation == "mentions":
                    continue
                head = str(edge.get("source") or "").strip()
                tail = str(edge.get("target") or "").strip()
                if not head or not tail:
                    continue
                key = (head, relation, tail)
                if key in seen:
                    continue
                seen.add(key)
                triples.append({"head": head, "relation": relation, "tail": tail})
        return triples

    @staticmethod
    def _compact_chain_result(chain_result: Any) -> Any:
        return chain_result
