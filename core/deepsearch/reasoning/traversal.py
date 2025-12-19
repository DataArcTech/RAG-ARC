"""
Graph traversal executor used by DeepSearch.

Runs planner steps through GraphDeepSearchAdapter (prepare → query → filter → summarize → chain traversal)
and returns traversal/evidence/reasoning records for downstream gap detection and reporting.
Keeps the adapter abstraction swappable so semantic or relational strategies can be configured per run.
"""
import asyncio
import logging
import time
import uuid
import weakref
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

logger = logging.getLogger(__name__)

_GLOBAL_ADAPTER_LOCKS: "weakref.WeakKeyDictionary[GraphDeepSearchAdapter, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _global_adapter_lock(adapter: GraphDeepSearchAdapter) -> asyncio.Lock:
    lock = _GLOBAL_ADAPTER_LOCKS.get(adapter)
    if lock is None:
        lock = asyncio.Lock()
        _GLOBAL_ADAPTER_LOCKS[adapter] = lock
    return lock


@dataclass
class GraphTraversalSettings:
    """Strategy configuration used by the traversal executor."""

    strategy_name: str = "ppr_chain"
    allow_semantic_channel: bool = True
    chain_depth: int = 4
    parallel_branches: int = 1


class GraphTraversalExecutor:
    """Run GraphDeepSearchAdapter traversals according to Planner output."""

    def __init__(self, adapter: GraphDeepSearchAdapter, settings: GraphTraversalSettings | None = None):
        self.adapter = adapter
        self.settings = settings or GraphTraversalSettings()
        self._adapter_lock = _global_adapter_lock(adapter)

    async def prepare(self, context: GraphQueryContext) -> None:
        """Ensure the adapter is warmed up for the provided context."""

        scope = context.resolve_scope()
        async with self._adapter_lock:
            await self.adapter.prepare(context.question or "", access_scope=scope)

    async def run(
        self,
        plan_steps: Sequence[PlanSpec],
        context: GraphQueryContext,
        *,
        tool_args_map: Optional[Dict[str, Dict[str, Any]]] = None,
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
    ) -> Tuple[Optional[GraphTraversalRecord], ReasoningStepRecord, List[EvidenceChunk]]:
        """Execute a single plan step so reasoning can interleave with other channels."""

        scope = context.resolve_scope()
        reasoning_entry = ReasoningStepRecord(
            step_id=step.step_id,
            description=step.description,
            channel=step.channel,
            status="running",
        )
        evidences: List[EvidenceChunk] = []
        traversal_record: Optional[GraphTraversalRecord] = None
        start = time.perf_counter()
        try:
            merged_seed_entities = self._merge_seed_entities(context, tool_args)
            query = self._resolve_query(step.description, tool_args)
            async with self._adapter_lock:
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

            chunk_id = f"{step.step_id}-{uuid.uuid4().hex[:8]}"
            summary_text = summary if isinstance(summary, str) else str(summary)
            subgraph_info = self._extract_subgraph_info(filtered, subgraph)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source=context.adapter_name,
                content=summary_text,
                score=1.0,
                provenance={
                    "plan_step": step.step_id,
                    "query": query,
                    "metadata": {
                        "_subgraph_info": subgraph_info,
                        "filter_type": filter_type,
                        "chain_result": self._compact_chain_result(chain_result),
                        "tool_args": tool_args or {},
                        "latency_ms": latency_ms,
                    },
                },
            )

            traversal_record = GraphTraversalRecord(
                step_id=step.step_id,
                strategy=self.settings.strategy_name,
                hop_count=self._infer_hops(filtered, chain_result),
                visited_nodes=self._collect_nodes(filtered),
                visited_edges=self._collect_edges(filtered),
                seed_entities=merged_seed_entities,
                retrieved_chunks=[chunk_id],
                metadata={
                    "channel": step.channel,
                    "filter_type": filter_type,
                    "tool_args": tool_args or {},
                    "latency_ms": latency_ms,
                },
            )

            reasoning_entry.status = "done"
            reasoning_entry.produced_evidence_ids.append(chunk_id)
            reasoning_entry.output_summary = summary_text
            reasoning_entry.diagnostics.setdefault("latency_ms", latency_ms)
            evidences.append(evidence)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Graph traversal failed for %s: %s", step.step_id, exc)
            reasoning_entry.status = "failed"
            reasoning_entry.diagnostics["error"] = str(exc)
        return traversal_record, reasoning_entry, evidences

    @staticmethod
    def _resolve_query(description: str, tool_args: Optional[Dict[str, Any]]) -> str:
        if tool_args and isinstance(tool_args.get("query"), str):
            return tool_args["query"]
        return description

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
            return min(len(edges), 10)
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
    def _compact_chain_result(chain_result: Any) -> Any:
        if not isinstance(chain_result, dict):
            return chain_result
        allowed = {"strategy", "hops", "visited", "scope"}
        return {key: chain_result.get(key) for key in allowed if chain_result.get(key) is not None}
