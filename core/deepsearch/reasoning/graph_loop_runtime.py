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
from core.deepsearch.utils.llm_envelope import try_parse_llm_envelope
from core.deepsearch.tooling.run_tool_memo import MemoEntry
from core.deepsearch.utils.ids import coerce_uuid_list

from .graph_loop_state import _RUN_THINK_COUNT, _run_plan_state
from .graph_loop_state import _RUN_TOOL_MEMO

logger = logging.getLogger(__name__)


class GraphLoopRuntimeMixin:
    @staticmethod
    def _extract_file_ids_from_tool_args(tool_name: str, tool_args: Any) -> list[str]:
        """Best-effort extraction of explicit file_id/file_ids from tool args.

        Priority: explicit tool args reflect the selected file(s) and should narrow file_scope.
        """

        if not isinstance(tool_args, dict):
            return []

        raw: list[Any] = []
        for key in ("file_ids", "file_id", "file", "source_file_ids", "source_file_id"):
            val = tool_args.get(key)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set, frozenset)):
                raw.extend(list(val))
            else:
                raw.append(val)

        valid, _invalid = coerce_uuid_list(raw)
        return list(valid)

    @staticmethod
    def _extract_file_ids_from_locate_diagnostics(diagnostics: Any) -> list[str]:
        if not isinstance(diagnostics, dict):
            return []
        results = diagnostics.get("results")
        if not isinstance(results, list):
            return []
        raw: list[Any] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            raw.append(row.get("file_id"))
        valid, _invalid = coerce_uuid_list(raw)
        return list(valid)

    @classmethod
    def _extract_file_ids_from_tool_result(cls, *, tool_name: str, result: Any) -> list[str]:
        """Extract candidate file_ids from tool results (locate inside explore)."""

        name = str(tool_name or "").strip()
        diag = getattr(result, "diagnostics", None)
        if name == "locate":
            out = cls._extract_file_ids_from_locate_diagnostics(diag)
            if out:
                return out
            # Fallback to parsing the JSON envelope in summary.
            summary = str(getattr(result, "summary", "") or "")
            env = try_parse_llm_envelope(summary)
            if isinstance(env, dict):
                answer = env.get("answer")
                valid, _invalid = coerce_uuid_list(answer if isinstance(answer, list) else [])
                return list(valid)
            return []

        return []

    async def _maybe_lock_in_file_scope(
        self,
        *,
        context: GraphQueryContext,
        tool_name: str,
        tool_args: Any,
        result: Any,
    ) -> None:
        """Lock file_scope into graph_context.metadata when we have strong signals.

        Rules:
        - Prefer explicit tool args (file_id/file_ids) since they represent a deliberate selection.
        - Otherwise, accept candidates from locate outputs (routing stage).
        """

        explicit = self._extract_file_ids_from_tool_args(tool_name, tool_args)
        candidates: list[str] = []
        if not explicit:
            candidates = self._extract_file_ids_from_tool_result(tool_name=tool_name, result=result)

        file_ids = explicit or candidates
        if not file_ids:
            return

        meta = dict(getattr(context, "metadata", None) or {})
        current = meta.get("file_scope") if isinstance(meta, dict) else None
        current_ids: list[str] = []
        if isinstance(current, dict):
            current_ids = list(current.get("file_ids") or [])

        if list(current_ids) == list(file_ids):
            return

        meta["file_scope"] = {
            "file_ids": list(file_ids),
            "filename_contains": [],
            "source": "tool_args" if explicit else "locate",
        }
        context.metadata = meta
        await emit_trace(
            "think",
            "Locked-in file_scope for subsequent tool calls.",
            meta={"tool": str(tool_name or ""), "file_ids": list(file_ids), "source": meta["file_scope"]["source"]},
        )

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
        proposed_total = len(proposed)

        # Dedupe policy (critical for agentic robustness):
        # - Only dedupe within the current think checkpoint to avoid repeated identical calls in a single response.
        # - Do NOT dedupe across checkpoints; the model must be allowed to re-read the same pages after discovering
        #   evidence gaps (e.g., tables not covered, wrong section, etc.).
        deduped_records: list[ReasoningStepRecord] = []
        unique: list[tuple[int, Dict[str, Any]]] = []
        seen: set[str] = set()
        for idx, call in enumerate(proposed, start=1):
            tool_name = str(call.get("tool_name") or call.get("tool") or "").strip()
            tool_args = call.get("tool_args") if isinstance(call.get("tool_args"), dict) else {}
            try:
                sig = tool_name + ":" + json.dumps(tool_args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except Exception:  # noqa: BLE001
                sig = tool_name
            if sig and sig in seen:
                deduped_records.append(
                    ReasoningStepRecord(
                        step_id=f"{think_step_id}_call_{idx:02d}",
                        description="Think-proposed tool call (deduped)",
                        channel="graph",
                        status="skipped",
                        diagnostics={"reason": "deduped", "tool_name": tool_name},
                    )
                )
                continue
            if sig:
                seen.add(sig)
            unique.append((idx, call))
        proposed = [call for _idx, call in unique]

        # Note: config uses 0 to mean "sequential" (avoid accidental full parallelism).
        concurrency = int(self._think_config.get("tool_call_concurrency") or 0)
        if concurrency <= 0:
            concurrency = 1
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run_one(orig_idx: int, call: Dict[str, Any]) -> ReasoningStepRecord:
            tool_name = str(call.get("tool_name") or call.get("tool") or "").strip()
            tool_args = call.get("tool_args") if isinstance(call.get("tool_args"), dict) else {}
            if not tool_name:
                return ReasoningStepRecord(
                    step_id=f"{think_step_id}_call_{orig_idx:02d}",
                    description="Think-proposed tool call (invalid)",
                    channel="graph",
                    status="failed",
                    diagnostics={"reason": "missing_tool_name"},
                )
            if available_tool_names and tool_name not in available_tool_names:
                return ReasoningStepRecord(
                    step_id=f"{think_step_id}_call_{orig_idx:02d}",
                    description="Think-proposed tool call (unknown tool)",
                    channel="graph",
                    status="failed",
                    diagnostics={"reason": "unknown_tool", "tool_name": tool_name},
                )
            plan_step_id = f"{think_step_id}_call_{orig_idx:02d}"

            record = ReasoningStepRecord(
                step_id=plan_step_id,
                description=str(call.get("rationale") or f"Think-proposed tool call: {tool_name}"),
                channel="graph",
                status="running",
            )

            memoizer = _RUN_TOOL_MEMO.get()
            memo_key = None
            if memoizer is not None and memoizer.is_cacheable(tool_name):
                owner_scope_id = ""
                try:
                    owner_scope_id = str(getattr(getattr(context, "access_scope", None), "scope_id", "") or "")
                except Exception:
                    owner_scope_id = ""
                file_scope_hint = {}
                try:
                    meta = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
                    raw_scope = meta.get("file_scope")
                    file_scope_hint = dict(raw_scope) if isinstance(raw_scope, dict) else {}
                except Exception:
                    file_scope_hint = {}
                memo_key = memoizer.make_key(
                    tool_name=tool_name,
                    owner_scope_id=owner_scope_id,
                    tool_args=dict(tool_args or {}),
                    file_scope_hint=file_scope_hint,
                )
                cached = memoizer.get(memo_key)
                if cached is not None:
                    # Replay: do not re-emit evidences (they were already added to EvidenceBank on first call).
                    record.status = "done"
                    record.output_summary = cached.result.summary
                    record.produced_evidence_ids = list(cached.produced_evidence_ids)
                    record.diagnostics.setdefault("reason", "tool_memoization_replay")
                    record.diagnostics.setdefault("latency_ms", 0)
                    record.tool_logs.append(
                        ToolExecutionLog(
                            tool_name=cached.result.tool_name,
                            server_name=None,
                            arguments_snapshot=tool_args,
                            response_excerpt=cached.result.summary if cached.result.summary else None,
                            latency_ms=0,
                            graph_context=context,
                            extra={
                                "channel": cached.result.channel,
                                "profile": cached.result.profile,
                                "determinism": cached.result.determinism,
                                "trigger": "tool_memoization_replay",
                                "parent_think_step_id": think_step_id,
                                "memoization": {"hit": True},
                            },
                        )
                    )
                    diag = dict(cached.result.diagnostics or {})
                    diag.setdefault("memoization", {})
                    diag["memoization"] = {"hit": True}
                    replay_payload = cached.result.model_copy(update={"diagnostics": diag, "evidences": []})
                    tool_runs.append(
                        {
                            "plan_step_id": plan_step_id,
                            "tool_name": replay_payload.tool_name,
                            "channel": replay_payload.channel,
                            "result": replay_payload.model_dump(),
                        }
                    )
                    for note in replay_payload.think_notes:
                        think_notes_out.append(note.model_dump(exclude_none=True))
                    if replay_payload.think_notes:
                        plan_state = _run_plan_state()
                        if plan_state is not None and update_plan_from_think_notes(plan_state, think_notes=replay_payload.think_notes):
                            await self._emit_plan_update(
                                plan_state=plan_state,
                                stage="tool_memoization_replay",
                                plan_step_id=plan_step_id,
                            )
                    return record

            async with semaphore:
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
                try:
                    if self._tool_timeout and self._tool_timeout > 0:
                        result = await asyncio.wait_for(invocation, timeout=self._tool_timeout)
                    else:
                        result = await invocation
                except Exception:  # noqa: BLE001
                    raise
                latency_ms = int((time.perf_counter() - start) * 1000)

            # Propagate/lock-in file scope between tool calls.
            try:
                await self._maybe_lock_in_file_scope(context=context, tool_name=tool_name, tool_args=tool_args, result=result)
            except Exception:  # noqa: BLE001
                # Never fail the run due to a scoping hint; keep it observable via trace only.
                await emit_trace(
                    "think",
                    "file_scope lock-in failed (continuing).",
                    meta={"tool": str(tool_name or ""), "stage": "file_scope_lockin"},
                )

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

            # Store into run-scoped memoization cache (after successful completion).
            if memoizer is not None and memo_key and memoizer.is_cacheable(result.tool_name):
                diag = dict(result.diagnostics or {})
                diag.setdefault("memoization", {})
                diag["memoization"] = {
                    "stored": True,
                    "original_evidence_count": len(result.evidences or []),
                }
                stored = result.model_copy(update={"diagnostics": diag, "evidences": []})
                memoizer.put(
                    memo_key,
                    MemoEntry(result=stored, produced_evidence_ids=tuple(record.produced_evidence_ids or [])),
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

        results = await asyncio.gather(*[run_one(orig_idx, call) for orig_idx, call in unique], return_exceptions=True)
        records: List[ReasoningStepRecord] = []
        records.extend(deduped_records)
        summary_rows: List[Dict[str, Any]] = []
        for (orig_idx, _call), res in zip(unique, results):
            if isinstance(res, Exception):
                err_text = str(res) or f"{res.__class__.__name__}"
                records.append(
                    ReasoningStepRecord(
                        step_id=f"{think_step_id}_call_err",
                        description="Think-proposed tool call failed",
                        channel="graph",
                        status="failed",
                        diagnostics={"reason": "exception", "error": err_text},
                    )
                )
                summary_rows.append(
                    {
                        "status": "failed",
                        "step_id": f"{think_step_id}_call_{orig_idx:02d}",
                        "failure_reason": "exception",
                        "error": err_text,
                    }
                )
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
        return records, {"proposed": proposed_total, "results": summary_rows}

    @staticmethod
    def _summarize_recent_tool_runs(
        tool_runs: Sequence[Dict[str, Any]],
        *,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        if max_items <= 0 or not tool_runs:
            return []
        summaries: List[Dict[str, Any]] = []
        for run in tool_runs[-max_items:]:
            if not isinstance(run, dict):
                continue
            result = run.get("result") if isinstance(run.get("result"), dict) else {}
            summary = str(result.get("summary") or "").strip()
            envelope = try_parse_llm_envelope(summary)
            # Keep JSON envelopes machine-readable for the next LLM step.
            if envelope is not None:
                summaries.append(
                    {
                        "plan_step_id": run.get("plan_step_id"),
                        "tool_name": run.get("tool_name"),
                        "channel": run.get("channel"),
                        "envelope": envelope,
                        "summary": None,  # Prefer structured envelope for JSON summaries.
                        "evidence_count": len(result.get("evidences") or []) if isinstance(result.get("evidences"), list) else 0,
                        "failure_reason": (
                            (result.get("diagnostics") or {}).get("reason") if isinstance(result.get("diagnostics"), dict) else None
                        ),
                    }
                )
                continue
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
