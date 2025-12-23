"""Graph-first reasoning loop that orchestrates adapter traversals and tool calls."""
import asyncio
import contextvars
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from encapsulation.data_model.deepsearch import (
    EvidenceChunk,
    GraphQueryContext,
    GraphTraversalRecord,
    PlanSpec,
    ReasoningStepRecord,
    ThinkNote,
    ToolExecutionLog,
    ToolResultPayload,
)
from core.deepsearch.tooling.protocols import ToolInvoker
from core.graph_adapter.base import GraphAccessScope, GraphDeepSearchAdapter
from core.graph_adapter.scope_provider import require_scope

from .parallel import ParallelExecutionManager
from .subagent import PlanSubAgent, SubAgentOutcome
from .traversal import GraphTraversalExecutor, GraphTraversalSettings

logger = logging.getLogger(__name__)


_RUN_EVIDENCES: contextvars.ContextVar[List[EvidenceChunk] | None] = contextvars.ContextVar(
    "deepsearch_run_evidences",
    default=None,
)
_RUN_EVIDENCE_LOCK: contextvars.ContextVar[asyncio.Lock | None] = contextvars.ContextVar(
    "deepsearch_run_evidence_lock",
    default=None,
)
_RUN_TOTAL_STEPS: contextvars.ContextVar[int] = contextvars.ContextVar("deepsearch_run_total_steps", default=0)
_RUN_THINK_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("deepsearch_run_think_count", default=0)


class GraphReasoningLoop:
    """Run multi-step graph reasoning using adapters, graph tools, and MCP routing."""

    def __init__(
        self,
        adapter: GraphDeepSearchAdapter,
        llm_connector,
        strategy_config,
        *,
        tool_manager: ToolInvoker | None = None,
    ):
        # adapter: dynamically injected HippoRAG/other graphrag implementation
        self.adapter = adapter
        # llm_connector: reserved for prompts or LLM backed tools (kept for parity with tool configs)
        self.llm_connector = llm_connector
        # strategy_config: Chain-of-Exploration parameters controlling traversal depth/filters
        self.strategy_config = strategy_config
        self.tool_manager = tool_manager
        self.traversal_settings = self._build_traversal_settings(strategy_config)
        self.traversal_executor = GraphTraversalExecutor(
            adapter=self.adapter,
            settings=self.traversal_settings,
        )
        self._adapter_metadata = self._resolve_adapter_metadata()
        self._think_config = self._build_think_config(strategy_config)
        self.parallel_branches = self._resolve_parallel_branches(strategy_config)
        self.max_parallel_branches = self._resolve_max_parallel(strategy_config)
        self._active_parallel_branches = max(1, self.parallel_branches or 1)
        self._tool_timeout = self._resolve_tool_timeout(strategy_config)

    async def run(
        self,
        question: str,
        plan_steps: Sequence[Dict[str, Any] | PlanSpec],
        *,
        graph_context: Optional[GraphQueryContext] = None,
    ) -> Dict[str, Any]:
        """Execute planner steps, trigger adapter traversals, and return a structured trace."""

        if not question:
            raise ValueError("question must be a non-empty string")
        normalized_steps = self._normalize_plan_steps(plan_steps)
        if not normalized_steps:
            raise ValueError("plan_steps must contain at least one step")

        context = self._prepare_graph_context(graph_context, question, normalized_steps)
        traversals: List[GraphTraversalRecord] = []
        evidences: List[EvidenceChunk] = []
        evidence_lock = asyncio.Lock()
        tool_runs: List[Dict[str, Any]] = []
        think_notes: List[Dict[str, Any]] = []
        pending_external: List[Dict[str, Any]] = []
        reasoning_results: Dict[int, ReasoningStepRecord] = {}
        aux_reasoning: List[ReasoningStepRecord] = []
        completed_internal_steps = 0
        coverage_metrics: Dict[str, Any] = {}

        if any(entry["run_with_adapter"] for entry in normalized_steps):
            await self.traversal_executor.prepare(context)

        evidences_token = _RUN_EVIDENCES.set(evidences)
        lock_token = _RUN_EVIDENCE_LOCK.set(evidence_lock)
        total_steps_token = _RUN_TOTAL_STEPS.set(len(normalized_steps))
        think_count_token = _RUN_THINK_COUNT.set(0)
        parallel_branches = self._determine_parallel_branches(normalized_steps)
        self._active_parallel_branches = parallel_branches
        sub_agents = [
            PlanSubAgent(owner=self, step_index=idx, entry=entry, question=question, context=context)
            for idx, entry in enumerate(normalized_steps)
        ]
        exec_manager = ParallelExecutionManager(max_concurrency=parallel_branches)

        async def _sequential_iter():
            for agent in sub_agents:
                yield agent.step_index, await agent.run()

        try:
            iterator = exec_manager.run(sub_agents) if parallel_branches > 1 else _sequential_iter()
            async for idx, outcome in iterator:
                reasoning_results[idx] = outcome.reasoning
                if outcome.traversal:
                    traversals.append(outcome.traversal)
                if outcome.pending_external:
                    pending_external.append(outcome.pending_external)
                if outcome.evidences:
                    await self._extend_shared_evidences(outcome.evidences)
                if outcome.tool_runs:
                    tool_runs.extend(outcome.tool_runs)
                if outcome.think_notes:
                    think_notes.extend(outcome.think_notes)

                if outcome.reasoning.status == "done" and not outcome.pending_external:
                    completed_internal_steps += 1

                coverage_metrics = self._coverage_snapshot(
                    evidence_count=len(evidences),
                    source_labels=[chunk.source for chunk in evidences],
                    completed_steps=completed_internal_steps,
                    total_steps=len(normalized_steps),
                )
                ordered_log = [reasoning_results[i] for i in sorted(reasoning_results)]
                think_record = await self._maybe_run_periodic_think(
                    question=question,
                    context=context,
                    evidences=evidences,
                    reasoning_log=ordered_log,
                    tool_runs=tool_runs,
                    think_notes=think_notes,
                    coverage_metrics=coverage_metrics,
                    completed_steps=completed_internal_steps,
                    total_steps=len(normalized_steps),
                )
                if think_record:
                    aux_reasoning.append(think_record)
        finally:
            _RUN_EVIDENCES.reset(evidences_token)
            _RUN_EVIDENCE_LOCK.reset(lock_token)
            _RUN_TOTAL_STEPS.reset(total_steps_token)
            _RUN_THINK_COUNT.reset(think_count_token)

        ordered_reasoning: List[ReasoningStepRecord] = []
        for idx, entry in enumerate(normalized_steps):
            record = reasoning_results.get(idx)
            if record is None:
                placeholder = self._empty_record(entry["spec"])
                placeholder.status = "skipped"
                placeholder.diagnostics.setdefault("reason", "sub_agent_missing")
                record = placeholder
            ordered_reasoning.append(record)
        combined_reasoning = ordered_reasoning + aux_reasoning

        return {
            "question": question,
            "graph_context": context.model_dump(exclude_none=True),
            "adapter_metadata": self._adapter_metadata,
            "plan_steps": [entry["spec"].model_dump() for entry in normalized_steps],
            "graph_traversals": [record.model_dump() for record in traversals],
            "reasoning_steps": [record.model_dump() for record in combined_reasoning],
            "evidences": [chunk.model_dump() for chunk in evidences],
            "tool_results": tool_runs,
            "pending_external": pending_external,
            "think_notes": think_notes,
            "coverage_metrics": coverage_metrics,
        }

    # ------------------------------------------------------------------
    async def _execute_plan_entry(
        self,
        *,
        step_index: int,
        entry: Dict[str, Any],
        question: str,
        context: GraphQueryContext,
    ) -> SubAgentOutcome:
        spec = entry["spec"]
        record = self._empty_record(spec)
        traversal_record: GraphTraversalRecord | None = None
        new_evidences: List[EvidenceChunk] = []
        tool_runs: List[Dict[str, Any]] = []
        think_notes: List[Dict[str, Any]] = []
        pending_external_payload: Dict[str, Any] | None = None

        record.diagnostics.setdefault("sub_agent", f"sub_agent_{step_index + 1:02d}")

        if not entry["enabled"]:
            record.status = "skipped"
            record.diagnostics.setdefault("reason", "disabled_by_planner")
            return SubAgentOutcome(step_index, record, None, [], [], [], None)

        if entry["requires_external"]:
            record.status = "pending_external"
            record.diagnostics.setdefault("reason", "requires_external_channel")
            pending_external_payload = self._pending_external_payload(entry)
            return SubAgentOutcome(step_index, record, None, [], [], [], pending_external_payload)

        if entry["run_with_adapter"]:
            traversal_record, reasoning_record, new_evidences = await self.traversal_executor.run_step(
                spec,
                context,
                tool_args=entry["tool_args"],
            )
            reasoning_record.diagnostics.setdefault("tool", entry["tool"] or "graph_adapter.query")
            return SubAgentOutcome(step_index, reasoning_record, traversal_record, new_evidences, [], [], None)

        if entry["should_invoke_tool"] and not self.tool_manager:
            record.status = "skipped"
            record.diagnostics.setdefault("reason", "tool_manager_disabled")
            record.diagnostics.setdefault("tool", entry["tool"])
            return SubAgentOutcome(step_index, record, None, [], [], [], None)

        if entry["should_invoke_tool"]:
            evidence_snapshot = await self._snapshot_evidences()
            coverage_hint = self._coverage_hint_for_step(step_index, evidence_snapshot)
            try:
                result, latency_ms = await self._invoke_tool(
                    tool_name=entry["tool"],
                    step=entry,
                    context=context,
                    question=question,
                    accumulated_evidence=evidence_snapshot,
                    coverage_hint=coverage_hint,
                )
            except asyncio.TimeoutError:
                logger.warning("Tool %s timed out for %s", entry["tool"], spec.step_id)
                record.status = "failed"
                record.diagnostics.setdefault("reason", "tool_timeout")
                record.diagnostics.setdefault("latency_ms", int(self._tool_timeout * 1000) if self._tool_timeout else None)
                return SubAgentOutcome(step_index, record, None, [], [], [], None)
            except Exception as exc:  # pragma: no cover - defensive guardrails
                logger.warning("Tool %s failed for %s: %s", entry["tool"], spec.step_id, exc)
                record.status = "failed"
                record.diagnostics.setdefault("error", str(exc))
                record.diagnostics.setdefault("reason", "tool_failure")
                return SubAgentOutcome(step_index, record, None, [], [], [], None)

            new_evidences = list(result.evidences)
            record.status = "done"
            record.output_summary = result.summary
            record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
            record.diagnostics.setdefault("tool", entry["tool"])
            record.diagnostics.setdefault("latency_ms", latency_ms)
            log_entry = ToolExecutionLog(
                tool_name=result.tool_name,
                server_name=None,
                arguments_snapshot=entry["tool_args"],
                response_excerpt=result.summary[:200] if result.summary else None,
                latency_ms=latency_ms,
                graph_context=context,
                extra={
                    "channel": spec.channel,
                    "profile": result.profile,
                    "determinism": result.determinism,
                },
            )
            record.tool_logs.append(log_entry)
            tool_runs.append(
                {
                    "plan_step_id": spec.step_id,
                    "tool_name": result.tool_name,
                    "channel": spec.channel,
                    "result": result.model_dump(),
                }
            )
            for note in result.think_notes:
                think_notes.append(note.model_dump(exclude_none=True))
            return SubAgentOutcome(step_index, record, None, new_evidences, tool_runs, think_notes, None)

        record.status = "skipped"
        record.diagnostics.setdefault("reason", "no_tool_available")
        return SubAgentOutcome(step_index, record, None, [], [], [], None)

    def _build_traversal_settings(self, config) -> GraphTraversalSettings:
        if isinstance(config, GraphTraversalSettings):
            return config
        if hasattr(config, "model_dump"):
            payload = config.model_dump()
        elif isinstance(config, dict):
            payload = dict(config)
        else:
            payload = getattr(config, "__dict__", {})
        allowed = {
            key: payload.get(key)
            for key in ("strategy_name", "allow_semantic_channel", "chain_depth", "parallel_branches")
            if key in payload and payload.get(key) is not None
        }
        return GraphTraversalSettings(**allowed)

    def _build_think_config(self, config) -> Dict[str, Any]:
        defaults = {
            "tool_name": "graph.think",
            "cadence": 0,
            "min_coverage": 0.75,
        }
        if not config:
            return defaults
        if hasattr(config, "model_dump"):
            try:
                payload = config.model_dump()
            except TypeError:
                payload = config.model_dump(exclude_none=True)
        elif isinstance(config, dict):
            payload = dict(config)
        else:
            payload = getattr(config, "__dict__", {})
        think_section = payload.get("think") if isinstance(payload.get("think"), dict) else {}
        tool_name = think_section.get("tool_name") or payload.get("think_tool_name") or defaults["tool_name"]
        cadence = think_section.get("every_n_steps") or payload.get("think_every_n_steps") or defaults["cadence"]
        min_coverage = (
            think_section.get("min_coverage")
            or payload.get("think_min_coverage")
            or defaults["min_coverage"]
        )
        cfg = {
            "tool_name": str(tool_name).strip() if tool_name else "",
            "cadence": int(cadence) if cadence is not None else 0,
            "min_coverage": float(min_coverage) if min_coverage is not None else defaults["min_coverage"],
        }
        if cfg["cadence"] < 0:
            cfg["cadence"] = 0
        return cfg

    def _resolve_adapter_metadata(self) -> Dict[str, Any]:
        metadata = getattr(self.adapter, "metadata", None)
        if callable(metadata):
            raw = metadata()
        else:
            raw = None
        if raw is None:
            return {}
        if hasattr(raw, "model_dump"):
            return raw.model_dump(exclude_none=True)
        if is_dataclass(raw):
            return asdict(raw)
        return {key: value for key, value in raw.__dict__.items() if not key.startswith("_")}

    def _build_graph_context(
        self,
        question: str,
        *,
        access_scope: GraphAccessScope,
        seed_entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphQueryContext:
        adapter_name = self._adapter_metadata.get("adapter_name") or getattr(self.adapter, "name", "graph_adapter")
        context_metadata = {
            "strategy": self.traversal_settings.strategy_name,
            "chain_depth": self.traversal_settings.chain_depth,
        }
        if metadata:
            context_metadata.update(metadata)
        return GraphQueryContext(
            adapter_name=adapter_name,
            question=question,
            metadata=context_metadata,
            seed_entities=list(seed_entities or []),
            access_scope=access_scope,
        )

    def _prepare_graph_context(
        self,
        context: Optional[GraphQueryContext],
        question: str,
        normalized_steps: Sequence[Dict[str, Any]],
    ) -> GraphQueryContext:
        if context is None:
            scope = require_scope()
            context = self._build_graph_context(question, access_scope=scope)
        else:
            # copy existing context to avoid mutating caller-owned instances
            try:
                context = context.model_copy()
            except AttributeError:
                context = GraphQueryContext(**context.model_dump())

        scope = context.resolve_scope()
        if scope is None:
            scope = require_scope()
            context = context.model_copy(update={"access_scope": scope})

        seed_entities = self._collect_seed_entities(context, normalized_steps)
        scheduler_hints = self._collect_scheduler_hints(normalized_steps)
        metadata = dict(context.metadata or {})
        if scheduler_hints:
            planner_hints = metadata.setdefault("planner_hints", {})
            planner_hints.setdefault("per_step", {}).update(scheduler_hints)
        if seed_entities:
            metadata.setdefault("planner_hints", {}).setdefault("seed_entities", seed_entities)
        return context.model_copy(update={"seed_entities": seed_entities, "metadata": metadata})

    def _normalize_plan_steps(self, plan_steps: Sequence[Dict[str, Any] | PlanSpec]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, raw in enumerate(plan_steps):
            if isinstance(raw, PlanSpec):
                spec = raw
                raw_dict = {
                    "step_id": spec.step_id,
                    "description": spec.description,
                    "channel": spec.channel,
                    "metadata": spec.metadata,
                }
            else:
                metadata = dict(raw.get("metadata") or {})
                spec = PlanSpec(
                    step_id=str(raw.get("step_id") or f"plan_{idx+1:02d}"),
                    description=str(raw.get("description") or ""),
                    channel=str(raw.get("channel") or "graph"),
                    metadata=metadata,
                )
                raw_dict = dict(raw)

            tool = raw_dict.get("tool") or spec.metadata.get("tool")
            tool_args = raw_dict.get("tool_args") or spec.metadata.get("tool_args") or {}
            enabled = raw_dict.get("enabled")
            requires_external = raw_dict.get("requires_external") or spec.metadata.get("requires_external")
            channel = (spec.channel or "graph").lower()
            requires_external = bool(requires_external or channel == "web")
            run_with_adapter = bool(
                not requires_external
                and (channel == "graph")
                and (not tool or tool == "graph_adapter.query")
            )
            should_invoke_tool = bool(
                not requires_external
                and channel in {"graph", "text"}
                and tool
                and tool != "graph_adapter.query"
            )
            normalized.append(
                {
                    "spec": spec,
                    "tool": tool,
                    "tool_args": tool_args if isinstance(tool_args, dict) else {},
                    "enabled": True if enabled is None else bool(enabled),
                    "requires_external": requires_external,
                    "run_with_adapter": run_with_adapter,
                    "should_invoke_tool": should_invoke_tool,
                    "channel": channel,
                    "raw": raw_dict,
                }
            )
        return normalized

    async def _invoke_tool(
        self,
        *,
        tool_name: str,
        step: Dict[str, Any],
        context: GraphQueryContext,
        question: str,
        accumulated_evidence: List[EvidenceChunk],
        coverage_hint: Dict[str, Any],
    ) -> tuple[ToolResultPayload, int]:
        if not self.tool_manager:
            raise RuntimeError("GraphReasoningLoop cannot invoke tools without a tool manager")
        payload = self._build_tool_payload(
            plan_step_id=step["spec"].step_id,
            question=question,
            context=context,
            evidences=accumulated_evidence,
            coverage_hint=coverage_hint,
            extra=step["tool_args"],
        )
        start = time.perf_counter()
        invoke_task = self.tool_manager.invoke(tool_name, payload=payload)
        if self._tool_timeout and self._tool_timeout > 0:
            result = await asyncio.wait_for(invoke_task, timeout=self._tool_timeout)
        else:
            result = await invoke_task
        latency_ms = int((time.perf_counter() - start) * 1000)
        return result, latency_ms

    def _build_tool_payload(
        self,
        *,
        plan_step_id: str,
        question: str,
        context: GraphQueryContext,
        evidences: List[EvidenceChunk],
        coverage_hint: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "plan_step": plan_step_id,
            "context_evidences": [chunk.model_dump() for chunk in evidences],
            "adapter": self.adapter,
            "access_scope": context.resolve_scope(),
            "extra": dict(extra or {}),
            "graph_context": context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_hint,
        }

    @staticmethod
    def _empty_record(spec: PlanSpec) -> ReasoningStepRecord:
        return ReasoningStepRecord(
            step_id=spec.step_id,
            description=spec.description,
            channel=spec.channel,
            status="pending",
        )

    @staticmethod
    def _pending_external_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
        spec: PlanSpec = entry["spec"]
        return {
            "step_id": spec.step_id,
            "description": spec.description,
            "channel": spec.channel,
            "tool": entry["tool"],
            "tool_args": entry["tool_args"],
            "metadata": spec.metadata,
        }

    @staticmethod
    def _coverage_snapshot(
        *,
        evidence_count: int,
        source_labels: Optional[Sequence[str]],
        completed_steps: int,
        total_steps: int,
    ) -> Dict[str, Any]:
        unique_sources = len({label for label in source_labels or [] if label})
        plan_progress_ratio = (completed_steps / total_steps) if total_steps else 0.0
        expected_min_chunks = 3
        try:
            # Optional override used for heuristic think windows (gap detector remains authoritative).
            import os

            raw = os.getenv("DEEPSEARCH_GAP_EXPECTED_MIN_CHUNKS")
            if raw is not None:
                expected_min_chunks = max(1, int(raw))
        except Exception:
            expected_min_chunks = 3
        evidence_ratio = evidence_count / max(1, expected_min_chunks)
        coverage_score = min(1.0, evidence_ratio)
        coverage_ratio = coverage_score
        return {
            "evidence_count": evidence_count,
            "unique_source_count": unique_sources,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "coverage_ratio": round(coverage_ratio, 3),
            "plan_progress_ratio": round(plan_progress_ratio, 3),
            "expected_min_chunks": expected_min_chunks,
            "coverage_score": round(coverage_score, 3),
            "confidence_score": None,
            "missing_topics": [],
        }

    async def _maybe_run_periodic_think(
        self,
        *,
        question: str,
        context: GraphQueryContext,
        evidences: List[EvidenceChunk],
        reasoning_log: List[ReasoningStepRecord],
        tool_runs: List[Dict[str, Any]],
        think_notes: List[Dict[str, Any]],
        coverage_metrics: Dict[str, Any],
        completed_steps: int,
        total_steps: int,
    ) -> Optional[ReasoningStepRecord]:
        if not self._should_run_think(completed_steps, coverage_metrics):
            return None
        if not self.tool_manager or not self._think_config["tool_name"]:
            return None
        next_count = _RUN_THINK_COUNT.get() + 1
        _RUN_THINK_COUNT.set(next_count)
        think_step_id = f"think_auto_{next_count:02d}"
        record = ReasoningStepRecord(
            step_id=think_step_id,
            description="Periodic think checkpoint",
            channel="graph",
            status="running",
        )
        reasoning_log.append(record)
        think_evidences = list(evidences[-6:]) if evidences else []
        payload = self._build_tool_payload(
            plan_step_id=think_step_id,
            question=question,
            context=context,
            evidences=think_evidences,
            coverage_hint=coverage_metrics,
            extra={
                "trigger": "periodic_think",
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "context_window": {"evidence_items": len(think_evidences)},
            },
        )
        try:
            start = time.perf_counter()
            invocation = self.tool_manager.invoke(self._think_config["tool_name"], payload=payload)
            if self._tool_timeout and self._tool_timeout > 0:
                result = await asyncio.wait_for(invocation, timeout=self._tool_timeout)
            else:
                result = await invocation
            latency_ms = int((time.perf_counter() - start) * 1000)
        except asyncio.TimeoutError:
            record.status = "done"
            record.output_summary = "Think tool timed out; proceeding with a heuristic checkpoint."
            record.diagnostics.setdefault("reason", "tool_timeout_fallback")
            record.diagnostics.setdefault("trigger", "periodic_think")
            note = ThinkNote(
                plan_step_id=think_step_id,
                reasoning="Periodic think checkpoint fell back due to tool timeout.",
                confidence_delta=None,
                coverage_delta=None,
                next_actions=["Continue execution; consider rerunning think with a smaller context if needed."],
                metadata={
                    "trigger": "periodic_think",
                    "fallback": True,
                    "timeout_seconds": self._tool_timeout,
                    "coverage_metrics": coverage_metrics,
                    "context_window": {"evidence_items": len(think_evidences)},
                },
            )
            think_notes.append(note.model_dump(exclude_none=True))
            return record
        except Exception as exc:  # pragma: no cover - defensive guardrail
            record.status = "failed"
            record.diagnostics.setdefault("error", str(exc))
            record.diagnostics.setdefault("reason", "periodic_think")
            return record

        await self._extend_shared_evidences(result.evidences)
        record.status = "done"
        record.output_summary = result.summary
        record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
        record.diagnostics.setdefault("reason", "periodic_think")
        record.diagnostics.setdefault("latency_ms", latency_ms)
        log_entry = ToolExecutionLog(
            tool_name=result.tool_name,
            server_name=None,
            arguments_snapshot={"trigger": "periodic_think"},
            response_excerpt=result.summary[:200] if result.summary else None,
            latency_ms=latency_ms,
            graph_context=context,
            extra={
                "channel": "graph",
                "profile": result.profile,
                "determinism": result.determinism,
                "trigger": "periodic_think",
            },
        )
        record.tool_logs.append(log_entry)
        tool_runs.append(
            {
                "plan_step_id": think_step_id,
                "tool_name": result.tool_name,
                "channel": "graph",
                "result": result.model_dump(),
            }
        )
        for note in result.think_notes:
            think_notes.append(note.model_dump(exclude_none=True))
        return record

    def _should_run_think(self, completed_steps: int, coverage_metrics: Dict[str, Any]) -> bool:
        cadence = self._think_config["cadence"]
        if cadence <= 0 or completed_steps <= 0 or (completed_steps % cadence) != 0:
            return False
        coverage_ratio = coverage_metrics.get("coverage_ratio") or 0.0
        return coverage_ratio < self._think_config["min_coverage"]

    def _collect_seed_entities(
        self,
        context: GraphQueryContext,
        steps: Sequence[Dict[str, Any]],
    ) -> List[str]:
        seeds: Set[str] = set()

        def _ingest(value: Any) -> None:
            if not value:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _ingest(item)
                return
            token = str(value).strip()
            if token:
                seeds.add(token)

        for existing in getattr(context, "seed_entities", []) or []:
            _ingest(existing)

        metadata_payload = getattr(context, "metadata", {}) or {}
        _ingest(metadata_payload.get("seed_entities"))
        planner_hint = metadata_payload.get("planner_hints") or {}
        _ingest(planner_hint.get("seed_entities"))

        for entry in steps:
            spec: PlanSpec = entry["spec"]
            _ingest(spec.metadata.get("seed_entities"))
            tool_args = entry.get("tool_args") or {}
            _ingest(tool_args.get("seed_entities"))
            _ingest(spec.metadata.get("seed_nodes"))
            _ingest(tool_args.get("seed_nodes"))

        return list(seeds)

    @staticmethod
    def _collect_scheduler_hints(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        hints: Dict[str, Any] = {}
        for entry in steps:
            spec: PlanSpec = entry["spec"]
            collected: Dict[str, Any] = {}
            metadata = spec.metadata or {}
            for key in ("scheduler", "scheduler_hint", "priority", "tool_profile"):
                if metadata.get(key) is not None:
                    collected[key] = metadata[key]
            tool_args = entry.get("tool_args") or {}
            if tool_args.get("scheduler") is not None:
                collected["scheduler"] = tool_args["scheduler"]
            if collected:
                hints[spec.step_id] = collected
        return hints

    async def _extend_shared_evidences(self, additions: Sequence[EvidenceChunk]) -> None:
        if not additions:
            return
        evidences, lock = self._run_evidence_state()
        async with lock:
            evidences.extend(additions)

    async def _snapshot_evidences(self) -> List[EvidenceChunk]:
        evidences, lock = self._run_evidence_state()
        async with lock:
            return list(evidences)

    def _coverage_hint_for_step(self, step_index: int, snapshot: Optional[List[EvidenceChunk]] = None) -> Dict[str, Any]:
        total = _RUN_TOTAL_STEPS.get() or 1
        snapshot = snapshot or []
        return self._coverage_snapshot(
            evidence_count=len(snapshot),
            source_labels=[chunk.source for chunk in snapshot],
            completed_steps=min(step_index, total),
            total_steps=total,
        )

    @staticmethod
    def _run_evidence_state() -> tuple[List[EvidenceChunk], asyncio.Lock]:
        evidences = _RUN_EVIDENCES.get()
        lock = _RUN_EVIDENCE_LOCK.get()
        if evidences is None or lock is None:
            raise RuntimeError("GraphReasoningLoop evidence state is not initialised; call run() first")
        return evidences, lock

    def _resolve_parallel_branches(self, config) -> int:
        if not config:
            return 1
        if hasattr(config, "parallel_branches"):
            value = getattr(config, "parallel_branches")
        elif isinstance(config, dict):
            value = config.get("parallel_branches")
        else:
            value = getattr(config, "parallel_branches", 1)
        try:
            numeric = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 1
        return numeric if numeric >= 0 else 0

    def _resolve_max_parallel(self, config) -> int:
        if not config:
            return 4
        if hasattr(config, "max_parallel_branches"):
            value = getattr(config, "max_parallel_branches")
        elif isinstance(config, dict):
            value = config.get("max_parallel_branches")
        else:
            value = getattr(config, "max_parallel_branches", 4)
        try:
            numeric = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 4
        return max(1, numeric)

    def _resolve_tool_timeout(self, config) -> float:
        if not config:
            return 45.0
        if isinstance(config, dict):
            value = config.get("tool_timeout_seconds")
        else:
            value = getattr(config, "tool_timeout_seconds", None)
        try:
            numeric = float(value) if value is not None else 45.0
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 45.0
        return max(0.0, numeric)

    def _determine_parallel_branches(self, steps: Sequence[Dict[str, Any]]) -> int:
        configured = self.parallel_branches
        if configured > 0:
            return configured
        step_count = len(steps)
        if step_count <= 1:
            return 1
        if not self._auto_parallel_allowed(steps):
            return 1
        target = min(self.max_parallel_branches, step_count)
        return max(1, target)

    @staticmethod
    def _auto_parallel_allowed(steps: Sequence[Dict[str, Any]]) -> bool:
        """Return True only when planner marks steps as safe to run concurrently.

        Defaults to serial execution unless the plan explicitly opts in.
        """

        saw_parallel_hint = False
        for entry in steps:
            spec: PlanSpec = entry["spec"]
            metadata = spec.metadata or {}
            tool_args = entry.get("tool_args") or {}
            raw_hint = (
                metadata.get("scheduler")
                or metadata.get("scheduler_hint")
                or tool_args.get("scheduler")
                or tool_args.get("scheduler_hint")
            )
            hint = str(raw_hint or "").strip().lower()
            if hint in {"serial", "sequential"}:
                return False
            if hint in {"parallel", "concurrent", "auto_parallel"}:
                saw_parallel_hint = True
            if metadata.get("parallelizable") is True or tool_args.get("parallelizable") is True:
                saw_parallel_hint = True
        return saw_parallel_hint
