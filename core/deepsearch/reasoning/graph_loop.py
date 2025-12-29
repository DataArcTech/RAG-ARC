"""Graph-first reasoning loop that orchestrates adapter traversals and tool calls."""
import asyncio
import contextvars
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from config.core.deepsearch.reasoning_defaults import (
    TRACE_REFLECTION_DEFAULT_EVIDENCE_PREVIEW_CHARS,
    TRACE_REFLECTION_DEFAULT_EVIDENCE_SAMPLE_COUNT,
    TRACE_REFLECTION_DEFAULT_MAX_LINES,
    TRACE_REFLECTION_DEFAULT_NEW_EVIDENCE_ID_COUNT,
    TRACE_REFLECTION_DEFAULT_TEMPERATURE,
)
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
from core.deepsearch.tools.base import call_llm_async
from core.deepsearch.tooling.protocols import ToolInvoker
from core.deepsearch.trace import emit_trace
from core.deepsearch.utils.compression import compact_evidences, resolve_compaction_config
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
_RUN_REFLECT_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("deepsearch_run_reflect_count", default=0)
_RUN_THINK_TOOL_SIGNATURES: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "deepsearch_run_think_tool_signatures",
    default=None,
)


class GraphReasoningLoop:
    """Run multi-step graph reasoning using adapters, graph tools, and MCP routing."""

    def __init__(
        self,
        adapter: GraphDeepSearchAdapter,
        llm_connector,
        strategy_config,
        *,
        tool_manager: ToolInvoker | None = None,
        graph_channel_tool: str,
    ):
        # adapter: dynamically injected HippoRAG/other graphrag implementation
        self.adapter = adapter
        # llm_connector: reserved for prompts or LLM backed tools (kept for parity with tool configs)
        self.llm_connector = llm_connector
        # strategy_config: Chain-of-Exploration parameters controlling traversal depth/filters
        self.strategy_config = strategy_config
        self.tool_manager = tool_manager
        self.graph_channel_tool = str(graph_channel_tool).strip()
        if not self.graph_channel_tool:
            raise ValueError("graph_channel_tool is required for GraphReasoningLoop")
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
        self._tool_context_max_items = self._resolve_tool_context_max_items(strategy_config)
        self._tool_context_max_chars = self._resolve_tool_context_max_chars(strategy_config)
        self._coverage_expected_min_chunks = self._resolve_expected_min_chunks(strategy_config)
        self._trace_reflection_enabled = self._resolve_trace_reflection_enabled(strategy_config)
        self._trace_reflection_max = self._resolve_trace_reflection_max(strategy_config)

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
        reflect_count_token = _RUN_REFLECT_COUNT.set(0)
        think_sig_token = _RUN_THINK_TOOL_SIGNATURES.set(set())
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
                await self._emit_trace_reflection(
                    question=question,
                    context=context,
                    outcome=outcome,
                    accumulated_evidences=evidences,
                    coverage_metrics=coverage_metrics,
                )
                ordered_log = [reasoning_results[i] for i in sorted(reasoning_results)]
                think_records = await self._maybe_run_periodic_think(
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
                if think_records:
                    aux_reasoning.extend(think_records)
        finally:
            _RUN_EVIDENCES.reset(evidences_token)
            _RUN_EVIDENCE_LOCK.reset(lock_token)
            _RUN_TOTAL_STEPS.reset(total_steps_token)
            _RUN_THINK_COUNT.reset(think_count_token)
            _RUN_REFLECT_COUNT.reset(reflect_count_token)
            _RUN_THINK_TOOL_SIGNATURES.reset(think_sig_token)

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
    async def _emit_trace_reflection(
        self,
        *,
        question: str,
        context: GraphQueryContext,
        outcome: SubAgentOutcome,
        accumulated_evidences: List[EvidenceChunk],
        coverage_metrics: Dict[str, Any],
    ) -> None:
        """Emit a user-visible reflection after a step completes.

        This is intentionally short and action-oriented (not chain-of-thought).
        """

        if not self._trace_reflection_enabled:
            return
        if self.llm_connector is None:
            return
        if self._trace_reflection_max <= 0:
            return
        next_count = _RUN_REFLECT_COUNT.get() + 1
        if next_count > self._trace_reflection_max:
            return
        _RUN_REFLECT_COUNT.set(next_count)

        reasoning = outcome.reasoning
        step_id = reasoning.step_id
        tool_name = None
        tool_logs = reasoning.tool_logs or []
        if tool_logs:
            tool_name = tool_logs[-1].tool_name
        if not tool_name and reasoning.diagnostics.get("tool"):
            tool_name = str(reasoning.diagnostics.get("tool"))
        tool_name = tool_name or (self.graph_channel_tool if outcome.traversal else "unknown")

        new_evidences = list(outcome.evidences or [])
        ev_samples: List[Dict[str, Any]] = []
        sample_count = int(TRACE_REFLECTION_DEFAULT_EVIDENCE_SAMPLE_COUNT)
        preview_chars = int(TRACE_REFLECTION_DEFAULT_EVIDENCE_PREVIEW_CHARS)
        for ev in (new_evidences[:sample_count] if new_evidences else accumulated_evidences[:sample_count]):
            try:
                ev_samples.append(
                    {
                        "chunk_id": ev.chunk_id,
                        "source": ev.source,
                        "score": ev.score,
                        "preview": (ev.content or "")[:preview_chars],
                    }
                )
            except Exception:
                continue

        traversal = outcome.traversal.model_dump(exclude_none=True) if outcome.traversal else None
        input_payload = {
            "step": {
                "step_id": step_id,
                "description": reasoning.description,
                "channel": reasoning.channel,
                "status": reasoning.status,
                "tool": tool_name,
                "output_summary": reasoning.output_summary,
                "produced_evidence_ids": reasoning.produced_evidence_ids,
            },
            "evidence_delta": {
                "new_evidence_count": len(new_evidences),
                "new_evidence_ids": [
                    ev.chunk_id for ev in new_evidences[: int(TRACE_REFLECTION_DEFAULT_NEW_EVIDENCE_ID_COUNT)]
                ],
                "samples": ev_samples,
            },
            "traversal": traversal,
            "coverage": {
                "evidence_count": coverage_metrics.get("evidence_count"),
                "coverage_ratio": coverage_metrics.get("coverage_ratio"),
                "coverage_score": coverage_metrics.get("coverage_score"),
                "completed_steps": coverage_metrics.get("completed_steps"),
                "total_steps": coverage_metrics.get("total_steps"),
            },
            "graph_context": context.model_dump(exclude_none=True),
        }

        system = (
            "You are writing a user-visible trace reflection for a research agent.\n"
            "Write concise, action-oriented notes about what was learned from the last step and what to do next.\n"
            "Do NOT reveal private chain-of-thought. Do NOT invent facts.\n"
            f"Return plain text (no JSON), at most {int(TRACE_REFLECTION_DEFAULT_MAX_LINES)} lines."
        )
        user = "Question:\n{q}\n\nLast step snapshot:\n{payload}\n\nWrite the reflection now.".format(
            q=str(question or "").strip(),
            payload=json.dumps(input_payload, ensure_ascii=False, indent=2, default=str),
        )
        text = await call_llm_async(
            self.llm_connector,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=float(TRACE_REFLECTION_DEFAULT_TEMPERATURE),
        )
        rendered = (text or "").strip()
        if not rendered:
            return
        await emit_trace(
            "think",
            rendered,
            meta={
                "stage": "reflection",
                "step_id": step_id,
                "tool": tool_name,
                "reflection_index": next_count,
            },
        )

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
                tool_name=self.graph_channel_tool,
            )
            reasoning_record.diagnostics.setdefault("tool", entry["tool"] or self.graph_channel_tool)
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
                response_excerpt=result.summary if result.summary else None,
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
        required_keys = ("strategy_name", "allow_semantic_channel", "chain_depth", "parallel_branches", "step_summary_max_chars")
        missing = [key for key in required_keys if payload.get(key) is None]
        if missing:
            raise ValueError(f"graph_reasoning config missing required keys: {missing}")
        return GraphTraversalSettings(
            strategy_name=str(payload["strategy_name"]),
            allow_semantic_channel=bool(payload["allow_semantic_channel"]),
            chain_depth=int(payload["chain_depth"]),
            parallel_branches=int(payload["parallel_branches"]),
            step_summary_max_chars=int(payload["step_summary_max_chars"]),
        )

    def _build_think_config(self, config) -> Dict[str, Any]:
        think = None
        if hasattr(config, "think"):
            think = getattr(config, "think")
        elif isinstance(config, dict) and isinstance(config.get("think"), dict):
            think = config.get("think")

        if think is None:
            raise ValueError("strategy_config.think is required for GraphReasoningLoop")

        def _get(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key)

        include_llm_tools = _get(think, "include_llm_tools")
        if include_llm_tools is None:
            raise ValueError("think.include_llm_tools is required for GraphReasoningLoop")

        cfg = {
            "tool_name": str(_get(think, "tool_name") or "").strip(),
            "cadence": int(_get(think, "every_n_steps") or 0),
            "min_coverage": float(_get(think, "min_coverage")),
            "enable_tool_calls": bool(_get(think, "enable_tool_calls")),
            "max_tool_calls": int(_get(think, "max_tool_calls")),
            "tool_call_concurrency": int(_get(think, "tool_call_concurrency")),
            "tool_catalog_max_items": int(_get(think, "tool_catalog_max_items")),
            "max_rounds_per_checkpoint": max(1, int(_get(think, "max_rounds_per_checkpoint") or 1)),
            "include_llm_tools": bool(include_llm_tools),
        }
        if cfg["enable_tool_calls"] and cfg["tool_catalog_max_items"] <= 0:
            raise ValueError("think.tool_catalog_max_items must be > 0 when think.enable_tool_calls is true")
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
        compression = self._resolve_strategy_compression_schema()
        if compression:
            current = metadata.get("compression") if isinstance(metadata.get("compression"), dict) else None
            metadata["compression"] = self._merge_compression_defaults(defaults=compression, overrides=current)
        return context.model_copy(update={"seed_entities": seed_entities, "metadata": metadata})

    def _resolve_strategy_compression_schema(self) -> Dict[str, Any]:
        cfg = self.strategy_config
        if hasattr(cfg, "compression"):
            fields_set = getattr(cfg, "model_fields_set", None)
            if isinstance(fields_set, set) and "compression" not in fields_set:
                return {}
            raw = getattr(cfg, "compression")
            if hasattr(raw, "model_dump"):
                try:
                    payload = raw.model_dump(exclude_none=True)
                except TypeError:
                    payload = raw.model_dump()
            elif isinstance(raw, dict):
                payload = dict(raw)
            else:
                payload = {}
        elif isinstance(cfg, dict) and isinstance(cfg.get("compression"), dict):
            payload = dict(cfg.get("compression") or {})
        else:
            payload = {}

        allowed: Dict[str, Any] = {}
        for key in ("tool_context", "think"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                allowed[key] = dict(value)
        return allowed

    @staticmethod
    def _merge_compression_defaults(*, defaults: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
        if not overrides:
            return dict(defaults)
        merged: Dict[str, Any] = dict(defaults)
        for branch, override in overrides.items():
            if branch not in {"tool_context", "think"}:
                continue
            if isinstance(override, dict) and isinstance(merged.get(branch), dict):
                merged[branch] = {**dict(merged.get(branch) or {}), **dict(override)}
            else:
                merged[branch] = override
        return merged

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
                and (not tool or tool == self.graph_channel_tool)
            )
            should_invoke_tool = bool(
                not requires_external
                and channel in {"graph", "text"}
                and tool
                and tool != self.graph_channel_tool
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
            "context_evidences": self._context_window_evidences(evidences, context=context, extra=extra),
            "adapter": self.adapter,
            "access_scope": context.resolve_scope(),
            "extra": dict(extra or {}),
            "graph_context": context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_hint,
        }

    def _context_window_evidences(
        self,
        evidences: List[EvidenceChunk],
        *,
        context: GraphQueryContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Bound evidence payload size for tool calls (recency retention).

        This prevents tool prompts from accidentally exceeding model context limits
        when the evidence pile grows large.
        """
        cfg = resolve_compaction_config(
            branch="tool_context",
            graph_context=context,
            extra=(extra or {}),
            default_max_items=int(self._tool_context_max_items or 0),
            default_max_chars=int(self._tool_context_max_chars or 0),
            default_mode="truncate",
            default_retention="tail",
            env_max_items="DEEPSEARCH_TOOL_CONTEXT_MAX_EVIDENCES",
            env_max_chars="DEEPSEARCH_TOOL_CONTEXT_MAX_CHARS",
        )
        compacted, _meta = compact_evidences(
            evidences or [],
            cfg=cfg,
            question=(context.question or ""),
            extra=(extra or {}),
            include_triple_count=False,
        )
        return compacted

    @staticmethod
    def _resolve_tool_context_max_items(config: Any) -> int:
        if hasattr(config, "compression"):
            fields_set = getattr(config, "model_fields_set", None)
            if not (isinstance(fields_set, set) and "compression" not in fields_set):
                comp = getattr(config, "compression")
                branch = getattr(comp, "tool_context", None) if comp is not None else None
                if branch is not None and hasattr(branch, "max_items") and getattr(branch, "max_items") is not None:
                    return max(1, int(getattr(branch, "max_items")))
        if isinstance(config, dict):
            comp = config.get("compression")
            if isinstance(comp, dict):
                branch = comp.get("tool_context")
                if isinstance(branch, dict) and branch.get("max_items") is not None:
                    return max(1, int(branch.get("max_items")))
        if hasattr(config, "tool_context_max_evidences"):
            return max(1, int(getattr(config, "tool_context_max_evidences")))
        if isinstance(config, dict) and config.get("tool_context_max_evidences") is not None:
            return max(1, int(config.get("tool_context_max_evidences")))
        raise ValueError("strategy_config.tool_context_max_evidences is required for GraphReasoningLoop")

    @staticmethod
    def _resolve_tool_context_max_chars(config: Any) -> int:
        if hasattr(config, "compression"):
            fields_set = getattr(config, "model_fields_set", None)
            if not (isinstance(fields_set, set) and "compression" not in fields_set):
                comp = getattr(config, "compression")
                branch = getattr(comp, "tool_context", None) if comp is not None else None
                if branch is not None and hasattr(branch, "max_chars") and getattr(branch, "max_chars") is not None:
                    return max(100, int(getattr(branch, "max_chars")))
        if isinstance(config, dict):
            comp = config.get("compression")
            if isinstance(comp, dict):
                branch = comp.get("tool_context")
                if isinstance(branch, dict) and branch.get("max_chars") is not None:
                    return max(100, int(branch.get("max_chars")))
        if hasattr(config, "tool_context_max_chars"):
            return max(100, int(getattr(config, "tool_context_max_chars")))
        if isinstance(config, dict) and config.get("tool_context_max_chars") is not None:
            return max(100, int(config.get("tool_context_max_chars")))
        raise ValueError("strategy_config.tool_context_max_chars is required for GraphReasoningLoop")

    @staticmethod
    def _resolve_expected_min_chunks(config: Any) -> int:
        if hasattr(config, "coverage_expected_min_chunks"):
            return max(1, int(getattr(config, "coverage_expected_min_chunks")))
        if isinstance(config, dict) and config.get("coverage_expected_min_chunks") is not None:
            return max(1, int(config.get("coverage_expected_min_chunks")))
        raise ValueError("strategy_config.coverage_expected_min_chunks is required for GraphReasoningLoop")

    @staticmethod
    def _resolve_trace_reflection_enabled(config: Any) -> bool:
        if hasattr(config, "trace_reflection_enabled"):
            return bool(getattr(config, "trace_reflection_enabled"))
        if isinstance(config, dict) and config.get("trace_reflection_enabled") is not None:
            return bool(config.get("trace_reflection_enabled"))
        raise ValueError("strategy_config.trace_reflection_enabled is required for GraphReasoningLoop")

    @staticmethod
    def _resolve_trace_reflection_max(config: Any) -> int:
        if hasattr(config, "trace_reflection_max"):
            return max(0, int(getattr(config, "trace_reflection_max")))
        if isinstance(config, dict) and config.get("trace_reflection_max") is not None:
            return max(0, int(config.get("trace_reflection_max")))
        raise ValueError("strategy_config.trace_reflection_max is required for GraphReasoningLoop")

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

    def _coverage_snapshot(
        self,
        *,
        evidence_count: int,
        source_labels: Optional[Sequence[str]],
        completed_steps: int,
        total_steps: int,
    ) -> Dict[str, Any]:
        unique_sources = len({label for label in source_labels or [] if label})
        plan_progress_ratio = (completed_steps / total_steps) if total_steps else 0.0
        expected_min_chunks = max(1, int(self._coverage_expected_min_chunks))
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
    ) -> Optional[List[ReasoningStepRecord]]:
        if not self._should_run_think(completed_steps, coverage_metrics):
            return None
        if not self.tool_manager or not self._think_config["tool_name"]:
            return None

        tool_catalog: List[Dict[str, Any]] = []
        available_tool_names: set[str] = set()
        limit = max(0, int(self._think_config.get("tool_catalog_max_items") or 0))
        if limit:
            from core.deepsearch.tooling import describe_available_tools

            adapter_hint = {
                "name": self.graph_channel_tool,
                "channel": "graph",
                "description": "Primary graph traversal via the configured graph adapter (prepare→query→filter→summarize→chain_traverse).",
                "profile": "X",
                "determinism": "adapter",
                "strategy_tags": ["graph", "adapter", "traversal"],
            }
            tool_catalog = describe_available_tools(
                extra_hints=[adapter_hint],
                include_llm_tools=bool(self._think_config["include_llm_tools"]),
            )[:limit]
            for entry in tool_catalog:
                if isinstance(entry, dict) and entry.get("name"):
                    available_tool_names.add(str(entry["name"]))

        max_rounds = max(1, int(self._think_config.get("max_rounds_per_checkpoint") or 1))
        checkpoint_records: List[ReasoningStepRecord] = []
        previous_tool_call_results: List[Dict[str, Any]] = []

        for round_idx in range(1, max_rounds + 1):
            next_count = _RUN_THINK_COUNT.get() + 1
            _RUN_THINK_COUNT.set(next_count)
            think_step_id = f"think_auto_{next_count:02d}"
            record = ReasoningStepRecord(
                step_id=think_step_id,
                description="Periodic think checkpoint" if round_idx == 1 else "Periodic think checkpoint (iterated)",
                channel="graph",
                status="running",
            )
            reasoning_log.append(record)
            checkpoint_records.append(record)

            think_evidences = list(evidences) if evidences else []
            payload = self._build_tool_payload(
                plan_step_id=think_step_id,
                question=question,
                context=context,
                evidences=think_evidences,
                coverage_hint=coverage_metrics,
                extra={
                    "trigger": "periodic_think",
                    "round": round_idx,
                    "completed_steps": completed_steps,
                    "total_steps": total_steps,
                    "context_window": {"evidence_items": len(think_evidences)},
                    "available_tools": tool_catalog,
                    "previous_tool_call_results": previous_tool_call_results,
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
                record.status = "failed"
                record.diagnostics.setdefault("reason", "tool_timeout")
                record.diagnostics.setdefault("trigger", "periodic_think")
                break
            except Exception as exc:  # pragma: no cover - defensive guardrail
                record.status = "failed"
                record.diagnostics.setdefault("error", str(exc))
                record.diagnostics.setdefault("reason", "periodic_think")
                break

            await self._extend_shared_evidences(result.evidences)
            record.status = "done"
            record.output_summary = result.summary
            record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
            record.diagnostics.setdefault("reason", "periodic_think")
            record.diagnostics.setdefault("latency_ms", latency_ms)
            record.tool_logs.append(
                ToolExecutionLog(
                    tool_name=result.tool_name,
                    server_name=None,
                    arguments_snapshot={"trigger": "periodic_think", "round": round_idx},
                    response_excerpt=result.summary if result.summary else None,
                    latency_ms=latency_ms,
                    graph_context=context,
                    extra={
                        "channel": "graph",
                        "profile": result.profile,
                        "determinism": result.determinism,
                        "trigger": "periodic_think",
                        "round": round_idx,
                    },
                )
            )
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

            if result.think_notes:
                lines: List[str] = []
                lines.append(f"Think checkpoint: {think_step_id}")
                for idx, note in enumerate(result.think_notes, start=1):
                    prefix = f"note_{idx}"
                    lines.append(f"{prefix}. reasoning={note.reasoning}")
                    if note.next_actions:
                        lines.append(f"{prefix}. next_actions={note.next_actions}")
                    if note.coverage_delta is not None:
                        lines.append(f"{prefix}. coverage_delta={note.coverage_delta}")
                    if note.confidence_delta is not None:
                        lines.append(f"{prefix}. confidence_delta={note.confidence_delta}")
                    missing = None
                    if isinstance(note.metadata, dict):
                        missing = note.metadata.get("missing_topics")
                    if isinstance(missing, list) and missing:
                        lines.append(f"{prefix}. missing_topics={missing}")
                await emit_trace(
                    "think",
                    "\n".join(lines),
                    meta={"stage": "think", "think_step_id": think_step_id, "tool_name": result.tool_name},
                )

            if not self._think_config.get("enable_tool_calls"):
                break
            tool_call_records, tool_call_summary = await self._execute_tool_calls_from_think(
                think_step_id=think_step_id,
                question=question,
                context=context,
                evidences=evidences,
                coverage_metrics=coverage_metrics,
                think_notes=result.think_notes or [],
                tool_runs=tool_runs,
                think_notes_out=think_notes,
                available_tool_names=available_tool_names,
            )
            checkpoint_records.extend(tool_call_records)
            proposed = int(tool_call_summary.get("proposed") or 0)
            previous_tool_call_results = list(tool_call_summary.get("results") or [])

            coverage_metrics.update(
                self._coverage_snapshot(
                    evidence_count=len(evidences),
                    source_labels=[chunk.source for chunk in evidences],
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                )
            )
            if proposed <= 0:
                break

        return checkpoint_records or None

    async def _execute_tool_calls_from_think(
        self,
        *,
        think_step_id: str,
        question: str,
        context: GraphQueryContext,
        evidences: List[EvidenceChunk],
        coverage_metrics: Dict[str, Any],
        think_notes: Sequence[ThinkNote],
        tool_runs: List[Dict[str, Any]],
        think_notes_out: List[Dict[str, Any]],
        available_tool_names: Set[str],
    ) -> tuple[List[ReasoningStepRecord], Dict[str, Any]]:
        """Execute tool calls proposed by the think tool (LLM-driven iteration loop)."""

        max_calls = max(0, int(self._think_config.get("max_tool_calls") or 0))
        if max_calls <= 0 or not self.tool_manager:
            return [], {"proposed": 0, "results": []}

        proposed: List[Dict[str, Any]] = []
        for note in think_notes or []:
            raw = note.metadata.get("raw") if isinstance(note.metadata, dict) else None
            calls = raw.get("tool_calls") if isinstance(raw, dict) else None
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        proposed.append(call)
        proposed = proposed[:max_calls]
        if not proposed:
            return [], {"proposed": 0, "results": []}

        concurrency = int(self._think_config.get("tool_call_concurrency") or 0)
        if concurrency <= 0:
            concurrency = len(proposed)
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run_one(idx: int, call: Dict[str, Any]) -> ReasoningStepRecord:
            tool_name = str(call.get("tool_name") or call.get("tool") or "").strip()
            tool_args = call.get("tool_args") if isinstance(call.get("tool_args"), dict) else {}
            if not tool_name:
                return ReasoningStepRecord(
                    step_id=f"{think_step_id}_call_{idx:02d}",
                    description="Think-proposed tool call (invalid)",
                    channel="graph",
                    status="failed",
                    diagnostics={"reason": "missing_tool_name"},
                )
            if available_tool_names and tool_name not in available_tool_names:
                return ReasoningStepRecord(
                    step_id=f"{think_step_id}_call_{idx:02d}",
                    description="Think-proposed tool call (unknown tool)",
                    channel="graph",
                    status="failed",
                    diagnostics={"reason": "unknown_tool", "tool_name": tool_name},
                )
            plan_step_id = f"{think_step_id}_call_{idx:02d}"

            # Dedupe repeated think-proposed tool calls within the same run.
            signatures = _RUN_THINK_TOOL_SIGNATURES.get()
            if signatures is not None:
                try:
                    sig = tool_name + ":" + json.dumps(tool_args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                except Exception:
                    sig = tool_name
                if sig in signatures:
                    return ReasoningStepRecord(
                        step_id=plan_step_id,
                        description="Think-proposed tool call (deduped)",
                        channel="graph",
                        status="skipped",
                        diagnostics={"reason": "deduped", "tool_name": tool_name},
                    )
                signatures.add(sig)

            record = ReasoningStepRecord(
                step_id=plan_step_id,
                description=str(call.get("rationale") or f"Think-proposed tool call: {tool_name}"),
                channel="graph",
                status="running",
            )
            async with semaphore:
                if tool_name == self.graph_channel_tool:
                    # Allow think tool to trigger the primary graph adapter traversal.
                    # This unlocks non-scan deepsearch actions directly from think checkpoints.
                    from encapsulation.data_model.deepsearch import PlanSpec

                    query = str(tool_args.get("query") or tool_args.get("focus_query") or "").strip()
                    if not query:
                        query = str(call.get("rationale") or "Graph adapter query").strip()
                    spec = PlanSpec(
                        step_id=plan_step_id,
                        description=query,
                        channel="graph",
                        metadata={"source": "think_tool_call"},
                    )
                    start = time.perf_counter()
                    traversal_record, reasoning_record, new_evidences = await self.traversal_executor.run_step(
                        spec,
                        context,
                        tool_args=tool_args,
                        tool_name=self.graph_channel_tool,
                    )
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    reasoning_record.diagnostics.setdefault("reason", "think_tool_call")
                    reasoning_record.diagnostics.setdefault("latency_ms", latency_ms)
                    if new_evidences:
                        await self._extend_shared_evidences(new_evidences)
                    tool_runs.append(
                        {
                            "plan_step_id": plan_step_id,
                            "tool_name": self.graph_channel_tool,
                            "channel": "graph",
                            "result": {
                                "summary": reasoning_record.output_summary,
                                "evidence_ids": [ev.chunk_id for ev in new_evidences],
                                "latency_ms": latency_ms,
                                "traversal": traversal_record.model_dump(exclude_none=True) if traversal_record else None,
                            },
                        }
                    )
                    return reasoning_record

                payload = self._build_tool_payload(
                    plan_step_id=plan_step_id,
                    question=question,
                    context=context,
                    evidences=evidences,
                    coverage_hint=coverage_metrics,
                    extra=tool_args,
                )
                start = time.perf_counter()
                invocation = self.tool_manager.invoke(tool_name, payload=payload)
                if self._tool_timeout and self._tool_timeout > 0:
                    result = await asyncio.wait_for(invocation, timeout=self._tool_timeout)
                else:
                    result = await invocation
                latency_ms = int((time.perf_counter() - start) * 1000)

            await self._extend_shared_evidences(result.evidences)
            record.status = "done"
            record.output_summary = result.summary
            record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
            record.diagnostics.setdefault("reason", "think_tool_call")
            record.diagnostics.setdefault("latency_ms", latency_ms)
            record.tool_logs.append(
                ToolExecutionLog(
                    tool_name=result.tool_name,
                    server_name=None,
                    arguments_snapshot=tool_args,
                    response_excerpt=result.summary if result.summary else None,
                    latency_ms=latency_ms,
                    graph_context=context,
                    extra={
                        "channel": result.channel,
                        "profile": result.profile,
                        "determinism": result.determinism,
                        "trigger": "think_tool_call",
                        "parent_think_step_id": think_step_id,
                    },
                )
            )
            tool_runs.append(
                {
                    "plan_step_id": plan_step_id,
                    "tool_name": result.tool_name,
                    "channel": result.channel,
                    "result": result.model_dump(),
                }
            )
            for note in result.think_notes:
                think_notes_out.append(note.model_dump(exclude_none=True))
            return record

        results = await asyncio.gather(
            *[run_one(idx + 1, call) for idx, call in enumerate(proposed)],
            return_exceptions=True,
        )
        records: List[ReasoningStepRecord] = []
        summary_rows: List[Dict[str, Any]] = []
        for res in results:
            if isinstance(res, Exception):
                records.append(
                    ReasoningStepRecord(
                        step_id=f"{think_step_id}_call_err",
                        description="Think-proposed tool call failed",
                        channel="graph",
                        status="failed",
                        diagnostics={"error": str(res), "reason": "think_tool_call"},
                    )
                )
                summary_rows.append({"status": "failed", "error": str(res)})
            else:
                records.append(res)
                summary_rows.append(
                    {
                        "status": res.status,
                        "step_id": res.step_id,
                        "produced_evidence_count": len(res.produced_evidence_ids or []),
                        "tool": (res.tool_logs[-1].tool_name if res.tool_logs else None),
                    }
                )
        return records, {"proposed": len(proposed), "results": summary_rows}

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
        if isinstance(config, dict):
            if "parallel_branches" not in config:
                raise ValueError("strategy_config.parallel_branches is required for GraphReasoningLoop")
            value = config["parallel_branches"]
        else:
            if not hasattr(config, "parallel_branches"):
                raise ValueError("strategy_config.parallel_branches is required for GraphReasoningLoop")
            value = getattr(config, "parallel_branches")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("strategy_config.parallel_branches must be an integer") from exc

    def _resolve_max_parallel(self, config) -> int:
        if isinstance(config, dict):
            if "max_parallel_branches" not in config:
                raise ValueError("strategy_config.max_parallel_branches is required for GraphReasoningLoop")
            value = config["max_parallel_branches"]
        else:
            if not hasattr(config, "max_parallel_branches"):
                raise ValueError("strategy_config.max_parallel_branches is required for GraphReasoningLoop")
            value = getattr(config, "max_parallel_branches")
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("strategy_config.max_parallel_branches must be an integer") from exc
        if numeric < 1:
            raise ValueError("strategy_config.max_parallel_branches must be >= 1")
        return numeric

    def _resolve_tool_timeout(self, config) -> float:
        if isinstance(config, dict):
            if "tool_timeout_seconds" not in config:
                raise ValueError("strategy_config.tool_timeout_seconds is required for GraphReasoningLoop")
            value = config["tool_timeout_seconds"]
        else:
            if not hasattr(config, "tool_timeout_seconds"):
                raise ValueError("strategy_config.tool_timeout_seconds is required for GraphReasoningLoop")
            value = getattr(config, "tool_timeout_seconds")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("strategy_config.tool_timeout_seconds must be a float") from exc
        if numeric < 0:
            raise ValueError("strategy_config.tool_timeout_seconds must be >= 0")
        return numeric

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
