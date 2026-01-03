"""Shared runtime methods for the DeepSearch graph reasoning loop.

This module hosts heavier orchestration methods split out from `graph_loop.py` for
maintainability and unit-level testing.
"""
import asyncio
import json
import logging
import time
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
    ReasoningStepRecord,
    ThinkNote,
    ToolExecutionLog,
)
from core.deepsearch.tools.base import call_llm_async
from core.deepsearch.trace import emit_trace

from .graph_loop_state import _RUN_REFLECT_COUNT, _RUN_THINK_COUNT, _RUN_THINK_TOOL_SIGNATURES
from .subagent import SubAgentOutcome

logger = logging.getLogger(__name__)


class GraphLoopRuntimeMixin:
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
                registry = None
                try:
                    registry = getattr(getattr(self.tool_manager, "local_registry", None), "tool_hint_registry", None)
                except Exception:
                    registry = None
                tool_catalog = describe_available_tools(
                    extra_hints=[adapter_hint],
                    registry=registry,
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


__all__ = ["GraphLoopRuntimeMixin"]
