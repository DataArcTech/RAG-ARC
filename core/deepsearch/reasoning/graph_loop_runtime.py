"""Shared runtime helpers for the think-driven DeepSearch loop."""
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Sequence, Set

from encapsulation.data_model.deepsearch import (
    EvidenceChunk,
    GraphQueryContext,
    ReasoningStepRecord,
    ThinkNote,
    ToolExecutionLog,
)
from core.deepsearch.memory.plan_state import update_plan_from_think_notes
from core.deepsearch.trace import emit_trace
from config.core.deepsearch import tool_defaults

from .graph_loop_state import _RUN_THINK_COUNT, _RUN_THINK_TOOL_SIGNATURES, _run_plan_state

logger = logging.getLogger(__name__)


class GraphLoopRuntimeMixin:
    async def _emit_plan_update(self, *, plan_state, stage: str, plan_step_id: str | None = None) -> None:
        markdown = str(getattr(plan_state, "markdown", "") or "").strip()
        if not markdown:
            return
        await emit_trace(
            "write_outline",
            markdown,
            meta={
                "stage": stage,
                "plan_step_id": plan_step_id,
                "plan_version": getattr(plan_state, "version", None),
                "plan_items": list(getattr(plan_state, "items", []) or []),
            },
        )

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
                if tool_name == "logic.check":
                    plan_state = _run_plan_state()
                    tool_args = dict(tool_args)
                    tool_args["runtime_snapshot"] = self._build_logic_check_snapshot(
                        tool_runs=tool_runs,
                        plan_state=plan_state,
                    )
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
            if result.think_notes:
                plan_state = _run_plan_state()
                if plan_state is not None and update_plan_from_think_notes(plan_state, think_notes=result.think_notes):
                    await self._emit_plan_update(
                        plan_state=plan_state,
                        stage="think_tool_call",
                        plan_step_id=plan_step_id,
                    )
            return record

        results = await asyncio.gather(
            *[run_one(idx + 1, call) for idx, call in enumerate(proposed)],
            return_exceptions=True,
        )
        records: List[ReasoningStepRecord] = []
        summary_rows: List[Dict[str, Any]] = []
        for idx, res in enumerate(results, start=1):
            if isinstance(res, Exception):
                records.append(
                    ReasoningStepRecord(
                        step_id=f"{think_step_id}_call_err",
                        description="Think-proposed tool call failed",
                        channel="graph",
                        status="failed",
                        diagnostics={"reason": "exception", "error": str(res)},
                    )
                )
                summary_rows.append({"status": "failed", "step_id": f"{think_step_id}_call_{idx:02d}"})
                continue
            records.append(res)
            tool_name = None
            tool_args = None
            if res.tool_logs:
                tool_name = res.tool_logs[-1].tool_name
                tool_args = res.tool_logs[-1].arguments_snapshot
            output_summary = str(res.output_summary or "").strip()
            if len(output_summary) > 400:
                output_summary = output_summary[:399] + "…"
            summary_rows.append(
                {
                    "status": res.status,
                    "step_id": res.step_id,
                    "produced_evidence_count": len(res.produced_evidence_ids or []),
                    "tool": (res.tool_logs[-1].tool_name if res.tool_logs else tool_name),
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "output_summary": output_summary or None,
                    "failure_reason": res.diagnostics.get("reason") if isinstance(res.diagnostics, dict) else None,
                }
            )
        return records, {"proposed": len(proposed), "results": summary_rows}

    @staticmethod
    def _summarize_recent_tool_runs(
        tool_runs: Sequence[Dict[str, Any]],
        *,
        max_items: int,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        if max_items <= 0 or not tool_runs:
            return []
        summaries: List[Dict[str, Any]] = []
        for run in tool_runs[-max_items:]:
            if not isinstance(run, dict):
                continue
            result = run.get("result") if isinstance(run.get("result"), dict) else {}
            summary = str(result.get("summary") or "").strip()
            if max_chars > 0 and len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."
            evidences = result.get("evidences")
            evidence_count = len(evidences) if isinstance(evidences, list) else 0
            diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
            failure_reason = None
            if isinstance(diagnostics, dict):
                failure_reason = diagnostics.get("reason") or diagnostics.get("error")
            summaries.append(
                {
                    "plan_step_id": run.get("plan_step_id"),
                    "tool_name": run.get("tool_name"),
                    "channel": run.get("channel"),
                    "summary": summary or None,
                    "evidence_count": evidence_count,
                    "failure_reason": failure_reason,
                }
            )
        return summaries

    def _build_logic_check_snapshot(
        self,
        *,
        tool_runs: Sequence[Dict[str, Any]],
        plan_state: Any,
    ) -> Dict[str, Any]:
        max_items = int(tool_defaults.LOGIC_CHECK_RECENT_TOOL_RUNS_MAX)
        max_chars = int(tool_defaults.LOGIC_CHECK_RECENT_TOOL_RUNS_MAX_CHARS)
        recent = self._summarize_recent_tool_runs(tool_runs, max_items=max_items, max_chars=max_chars)
        evidence_ids = self._collect_evidence_ids_from_runs(
            tool_runs,
            limit=int(tool_defaults.LOGIC_CHECK_EVIDENCE_ID_MAX),
        )
        tool_names = self._collect_tool_names_from_runs(tool_runs)
        return {
            "plan": list(getattr(plan_state, "items", []) or []),
            "recent_tool_runs": recent,
            "tool_names": sorted(tool_names),
            "evidence_ids": evidence_ids,
            "tool_run_count": len(tool_runs),
        }

    @staticmethod
    def _collect_tool_names_from_runs(tool_runs: Sequence[Dict[str, Any]]) -> Set[str]:
        names: Set[str] = set()
        for run in tool_runs:
            if not isinstance(run, dict):
                continue
            name = str(run.get("tool_name") or "").strip()
            if name:
                names.add(name)
        return names

    @staticmethod
    def _collect_evidence_ids_from_runs(
        tool_runs: Sequence[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[str]:
        collected: List[str] = []
        seen: Set[str] = set()
        for run in tool_runs:
            if not isinstance(run, dict):
                continue
            result = run.get("result") if isinstance(run.get("result"), dict) else None
            evidences = result.get("evidences") if isinstance(result, dict) else None
            if not isinstance(evidences, list):
                continue
            for item in evidences:
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                collected.append(chunk_id)
                if len(collected) >= max(0, limit):
                    return collected
        return collected


def _prioritize_tool_catalog(
    hints: List[Dict[str, Any]],
    *,
    always_include: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    by_name: Dict[str, Dict[str, Any]] = {}
    for hint in hints or []:
        if not isinstance(hint, dict):
            continue
        name = str(hint.get("name") or "").strip()
        if not name:
            continue
        by_name.setdefault(name, hint)

    for name in always_include or ():
        token = str(name or "").strip()
        if not token or token in seen:
            continue
        hint = by_name.get(token)
        if hint is None:
            continue
        seen.add(token)
        ordered.append(hint)
        if len(ordered) >= limit:
            return ordered

    for hint in hints or []:
        if not isinstance(hint, dict):
            continue
        name = str(hint.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(hint)
        if len(ordered) >= limit:
            break
    return ordered


__all__ = ["GraphLoopRuntimeMixin"]
