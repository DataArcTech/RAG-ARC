"""Lead/worker style orchestrator for DeepSearch reasoning."""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext

from .graph_loop import GraphReasoningLoop
from core.deepsearch.tooling.protocols import ToolInvoker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiAgentSettings:
    """Runtime knobs for lead/worker orchestration."""

    enabled: bool = True
    max_subagents: int = 4
    subagent_concurrency: int = 4
    enable_parallel_tool_probes: bool = True
    probe_tool_names: Sequence[str] = (
        "graph.chunk_scan",
        "graph.pattern_scan",
    )
    probe_concurrency: int = 4
    lead_tool_names: Sequence[str] = (
        "graph.context_rollup",
        "graph.evidence_crosscheck",
    )
    lead_tool_concurrency: int = 2
    worker_timeout_seconds: Optional[float] = None
    worker_retry_attempts: int = 0
    fail_fast: bool = False


@dataclass(frozen=True)
class _WorkerSpec:
    agent_id: str
    session_id: str
    focus: str
    primary_step_id: str
    plan_steps: List[Dict[str, Any]]


class MultiAgentGraphReasoningLoop:
    """Lead agent that fans out work to parallel worker agents and merges results."""

    def __init__(
        self,
        *,
        adapter,
        llm_connector,
        strategy_config,
        tool_manager: ToolInvoker | None = None,
        settings: MultiAgentSettings | Dict[str, Any] | None = None,
    ):
        self.adapter = adapter
        self.llm_connector = llm_connector
        self.strategy_config = strategy_config
        self.tool_manager = tool_manager
        self.settings = self._coerce_settings(settings)

    async def run(
        self,
        question: str,
        plan_steps: Sequence[Dict[str, Any]],
        *,
        graph_context: Optional[GraphQueryContext] = None,
    ) -> Dict[str, Any]:
        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must be a non-empty string")

        steps = list(plan_steps or [])
        if not steps:
            raise ValueError("plan_steps must contain at least one step")

        if not self.settings.enabled:
            worker = self._build_worker()
            return await worker.run(normalized_question, steps, graph_context=graph_context)

        context = graph_context.model_copy() if graph_context else None
        if context is not None and context.question is None:
            context = context.model_copy(update={"question": normalized_question})

        worker_specs = self._build_worker_specs(steps)
        started_at = time.perf_counter()
        worker_outcomes = await self._run_workers(worker_specs, normalized_question, base_context=context)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        merged = self._merge_worker_traces(worker_outcomes)
        merged["question"] = normalized_question
        if context is not None:
            merged["graph_context"] = context.model_copy(update={"question": normalized_question}).model_dump(exclude_none=True)
        merged["plan_steps"] = self._order_plan_steps(
            merged_plan_steps=list(merged.get("plan_steps") or []),
            original_plan_steps=steps,
        )
        merged.setdefault("coverage_metrics", {})
        merged["coverage_metrics"].setdefault("orchestration_latency_ms", latency_ms)
        merged["coverage_metrics"].setdefault("worker_count", len(worker_outcomes))
        merged["coverage_metrics"].setdefault("worker_error_count", 0)
        merged["coverage_metrics"].setdefault("worker_errors", [])
        merged["coverage_metrics"].setdefault(
            "worker_sessions",
            [outcome["agent_session"] for outcome in worker_outcomes],
        )

        worker_errors = self._summarize_worker_errors(worker_outcomes)
        if worker_errors:
            merged["coverage_metrics"]["worker_error_count"] = len(worker_errors)
            merged["coverage_metrics"]["worker_errors"] = worker_errors

        if self.tool_manager and self.settings.lead_tool_names:
            lead_tools = await self._run_lead_tools(
                normalized_question,
                merged,
                base_context=context,
            )
            merged = self._merge_lead_tools(merged, lead_tools)

        merged["agent_sessions"] = [outcome["agent_session"] for outcome in worker_outcomes]
        return merged

    # ------------------------------------------------------------------
    def _build_worker(self) -> GraphReasoningLoop:
        return GraphReasoningLoop(
            adapter=self.adapter,
            llm_connector=self.llm_connector,
            strategy_config=self.strategy_config,
            tool_manager=self.tool_manager,
        )

    @staticmethod
    def _coerce_settings(raw: MultiAgentSettings | Dict[str, Any] | None) -> MultiAgentSettings:
        if raw is None:
            return MultiAgentSettings()
        if isinstance(raw, MultiAgentSettings):
            return raw
        if isinstance(raw, dict):
            payload = dict(raw)
            return MultiAgentSettings(
                enabled=bool(payload.get("enabled", True)),
                max_subagents=int(payload.get("max_subagents", 4)),
                subagent_concurrency=int(payload.get("subagent_concurrency", 4)),
                enable_parallel_tool_probes=bool(payload.get("enable_parallel_tool_probes", True)),
                probe_tool_names=tuple(payload.get("probe_tool_names") or MultiAgentSettings.probe_tool_names),
                probe_concurrency=int(payload.get("probe_concurrency", 4)),
                lead_tool_names=tuple(payload.get("lead_tool_names") or MultiAgentSettings.lead_tool_names),
                lead_tool_concurrency=int(payload.get("lead_tool_concurrency", 2)),
                worker_timeout_seconds=payload.get("worker_timeout_seconds"),
                worker_retry_attempts=int(payload.get("worker_retry_attempts", 0) or 0),
                fail_fast=bool(payload.get("fail_fast", False)),
            )
        return MultiAgentSettings()

    @staticmethod
    def _order_plan_steps(*, merged_plan_steps: List[Dict[str, Any]], original_plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        order: List[str] = []
        for step in original_plan_steps:
            step_id = str(step.get("step_id") or "").strip()
            if step_id:
                order.append(step_id)

        index = {step_id: idx for idx, step_id in enumerate(order)}
        seen: set[str] = set()
        normalized: List[Tuple[int, Dict[str, Any]]] = []
        fallback: List[Dict[str, Any]] = []

        for step in merged_plan_steps:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("step_id") or "").strip()
            if not step_id or step_id in seen:
                continue
            seen.add(step_id)
            if step_id in index:
                normalized.append((index[step_id], step))
            else:
                fallback.append(step)

        normalized.sort(key=lambda item: item[0])
        ordered = [payload for _, payload in normalized] + fallback

        if len(seen) < len(order):
            for step in original_plan_steps:
                step_id = str(step.get("step_id") or "").strip()
                if not step_id or step_id in seen:
                    continue
                seen.add(step_id)
                ordered.append(
                    {
                        "step_id": step_id,
                        "description": str(step.get("description") or ""),
                        "channel": str(step.get("channel") or "graph"),
                        "metadata": dict(step.get("metadata") or {}),
                    }
                )
        return ordered

    def _build_worker_specs(self, plan_steps: List[Dict[str, Any]]) -> List[_WorkerSpec]:
        # Planner prompt semantics: steps are serial by default; only parallelize when explicitly marked.
        parallel_flags = [self._is_parallel_safe(step) for step in plan_steps]
        if not any(parallel_flags):
            return [
                _WorkerSpec(
                    agent_id="worker_01",
                    session_id=uuid.uuid4().hex,
                    focus=(str(plan_steps[0].get("description") or "") if plan_steps else "plan") or "plan",
                    primary_step_id=str(plan_steps[0].get("step_id") or "plan_01"),
                    plan_steps=list(plan_steps),
                )
            ]

        # Mixed serial+parallel plans are ambiguous without explicit dependencies; run serially to preserve
        # step ordering + evidence accumulation (chunk-first) rather than guessing.
        if not all(parallel_flags):
            return [
                _WorkerSpec(
                    agent_id="worker_01",
                    session_id=uuid.uuid4().hex,
                    focus=(str(plan_steps[0].get("description") or "") if plan_steps else "plan") or "plan",
                    primary_step_id=str(plan_steps[0].get("step_id") or "plan_01"),
                    plan_steps=list(plan_steps),
                )
            ]

        max_agents = max(1, int(self.settings.max_subagents))
        desired_workers = min(max_agents, max(1, len(plan_steps)))
        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(desired_workers)]
        for idx, step in enumerate(plan_steps):
            buckets[idx % desired_workers].append(step)

        specs: List[_WorkerSpec] = []
        for idx, bucket in enumerate(buckets, start=1):
            primary = bucket[0]
            agent_id = f"worker_{idx:02d}"
            primary_step_id = str(primary.get("step_id") or f"plan_{agent_id}")
            focus = (primary.get("description") or "").strip() or primary_step_id
            specs.append(
                _WorkerSpec(
                    agent_id=agent_id,
                    session_id=uuid.uuid4().hex,
                    focus=focus,
                    primary_step_id=primary_step_id,
                    plan_steps=bucket,
                )
            )
        return specs

    @staticmethod
    def _is_parallel_safe(step: Dict[str, Any]) -> bool:
        metadata = step.get("metadata") if isinstance(step, dict) else None
        if not isinstance(metadata, dict):
            metadata = {}
        tool_args = step.get("tool_args") if isinstance(step, dict) else None
        if not isinstance(tool_args, dict):
            tool_args = {}

        if metadata.get("parallelizable") is True or tool_args.get("parallelizable") is True:
            return True
        scheduler = str(metadata.get("scheduler") or tool_args.get("scheduler") or "").strip().lower()
        return scheduler in {"parallel", "concurrent"}

    async def _run_workers(
        self,
        worker_specs: List[_WorkerSpec],
        question: str,
        *,
        base_context: Optional[GraphQueryContext],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, self.settings.subagent_concurrency))

        async def _run_one(spec: _WorkerSpec) -> Dict[str, Any]:
            async with semaphore:
                return await self._run_worker(spec, question, base_context=base_context)

        tasks = [asyncio.create_task(_run_one(spec)) for spec in worker_specs]
        results: List[Dict[str, Any]] = []
        for task in asyncio.as_completed(tasks):
            try:
                results.append(await task)
            except Exception:
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                raise
        results.sort(key=lambda item: str((item.get("agent_session") or {}).get("agent_id") or ""))
        return results

    async def _run_worker(
        self,
        spec: _WorkerSpec,
        question: str,
        *,
        base_context: Optional[GraphQueryContext],
    ) -> Dict[str, Any]:
        agent_id = spec.agent_id
        step_id = spec.primary_step_id
        focus = spec.focus
        assigned_step_ids = [str(step.get("step_id") or "") for step in spec.plan_steps if str(step.get("step_id") or "")]

        worker_question = self._build_worker_question(question=question, focus=focus, step_id=step_id)
        worker_context = base_context.model_copy() if base_context else None
        if worker_context is None:
            worker_context = GraphQueryContext(adapter_name="unknown", question=question)
        metadata = dict(worker_context.metadata or {})
        metadata["agent_id"] = agent_id
        metadata["agent_session_id"] = spec.session_id
        metadata["assigned_step_ids"] = assigned_step_ids
        metadata["primary_step_id"] = step_id
        metadata["focus"] = focus
        worker_context = worker_context.model_copy(update={"question": worker_question, "metadata": metadata})

        worker = self._build_worker()
        trace, worker_latency_ms, error = await self._run_worker_with_retries(
            worker=worker,
            worker_question=worker_question,
            worker_plan_steps=spec.plan_steps,
            worker_context=worker_context,
        )

        probe_results: List[Dict[str, Any]] = []
        if (
            not error
            and self.tool_manager
            and self.settings.enable_parallel_tool_probes
            and self.settings.probe_tool_names
        ):
            probe_results = await self._run_parallel_probes(
                agent_id=agent_id,
                focus_query=focus,
                question=worker_question,
                graph_context=trace.get("graph_context"),
                evidences=trace.get("evidences") or [],
            )
            trace = self._merge_probe_results(trace, probe_results)
        debrief = self._build_worker_debrief(trace, focus=focus, assigned_step_ids=assigned_step_ids, error=error)

        return {
            "agent_session": {
                "agent_id": agent_id,
                "primary_step_id": step_id,
                "focus": focus,
                "session_id": spec.session_id,
                "assigned_step_ids": assigned_step_ids,
                "probe_tools": list(self.settings.probe_tool_names),
                "latency_ms": worker_latency_ms,
                "error": error,
                "debrief": debrief,
            },
            "trace": trace,
            "probe_results": probe_results,
        }

    @staticmethod
    def _build_worker_question(*, question: str, focus: str, step_id: str) -> str:
        base = (question or "").strip()
        focus_line = (focus or "").strip() or step_id
        return f"{base}\n\nSubtask: {focus_line}"

    async def _run_worker_with_retries(
        self,
        *,
        worker: GraphReasoningLoop,
        worker_question: str,
        worker_plan_steps: List[Dict[str, Any]],
        worker_context: GraphQueryContext,
    ) -> Tuple[Dict[str, Any], int, Optional[str]]:
        attempts = max(0, int(self.settings.worker_retry_attempts))
        timeout = self.settings.worker_timeout_seconds

        last_error: Optional[str] = None
        for attempt in range(attempts + 1):
            start = time.perf_counter()
            try:
                run_task = worker.run(worker_question, worker_plan_steps, graph_context=worker_context)
                if timeout and timeout > 0:
                    trace = await asyncio.wait_for(run_task, timeout=timeout)
                else:
                    trace = await run_task
                latency_ms = int((time.perf_counter() - start) * 1000)
                return trace, latency_ms, None
            except asyncio.TimeoutError:
                last_error = "worker_timeout"
                logger.warning("Worker %s timed out on attempt %s", worker_context.metadata.get("agent_id"), attempt + 1)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc) or exc.__class__.__name__
                logger.warning(
                    "Worker %s failed on attempt %s: %s",
                    worker_context.metadata.get("agent_id"),
                    attempt + 1,
                    last_error,
                )

        if self.settings.fail_fast:
            raise RuntimeError(last_error or "worker_failed")
        return {"question": worker_question, "graph_context": worker_context.model_dump(exclude_none=True)}, 0, last_error

    @staticmethod
    def _summarize_worker_errors(worker_outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for outcome in worker_outcomes:
            agent_session = outcome.get("agent_session") or {}
            error = agent_session.get("error")
            if not error:
                continue
            errors.append(
                {
                    "agent_id": agent_session.get("agent_id"),
                    "session_id": agent_session.get("session_id"),
                    "primary_step_id": agent_session.get("primary_step_id"),
                    "assigned_step_ids": agent_session.get("assigned_step_ids") or [],
                    "focus": agent_session.get("focus"),
                    "error": error,
                    "latency_ms": agent_session.get("latency_ms"),
                }
            )
        return errors

    @staticmethod
    def _build_worker_debrief(
        trace: Dict[str, Any],
        *,
        focus: str,
        assigned_step_ids: List[str],
        error: Optional[str],
    ) -> Dict[str, Any]:
        evidences = trace.get("evidences") or []
        evidence_ids: List[str] = []
        sources: List[str] = []
        for item in evidences:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            source = str(item.get("source") or "").strip()
            if chunk_id:
                evidence_ids.append(chunk_id)
            if source:
                sources.append(source)
        reasoning_steps = trace.get("reasoning_steps") or []
        status_counts: Dict[str, int] = {}
        for step in reasoning_steps:
            if isinstance(step, dict):
                status = str(step.get("status") or "").strip() or "unknown"
                status_counts[status] = status_counts.get(status, 0) + 1
        pending_external = trace.get("pending_external") or []
        return {
            "focus": focus,
            "assigned_step_ids": assigned_step_ids,
            "error": error,
            "evidence_count": len(evidences),
            "evidence_ids": evidence_ids[:12],
            "unique_sources": sorted({s for s in sources if s})[:12],
            "reasoning_status_counts": status_counts,
            "pending_external_count": len(pending_external) if isinstance(pending_external, list) else 0,
        }

    async def _run_parallel_probes(
        self,
        *,
        agent_id: str,
        focus_query: str,
        question: str,
        graph_context: Any,
        evidences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, self.settings.probe_concurrency))

        async def _invoke(tool_name: str) -> Dict[str, Any]:
            async with semaphore:
                payload = {
                    "question": question,
                    "plan_step": f"{agent_id}:{tool_name}",
                    "context_evidences": evidences,
                    "adapter": self.adapter,
                    "access_scope": (graph_context or {}).get("access_scope") if isinstance(graph_context, dict) else None,
                    "extra": {"focus_query": focus_query},
                    "graph_context": graph_context,
                    "coverage_metrics": {},
                }
                result = await self.tool_manager.invoke(tool_name, payload=payload)  # type: ignore[union-attr]
                return {
                    "tool_name": tool_name,
                    "channel": result.channel,
                    "result": result.model_dump(),
                }

        tasks = [asyncio.create_task(_invoke(name)) for name in self.settings.probe_tool_names]
        results: List[Dict[str, Any]] = []
        for task in asyncio.as_completed(tasks):
            try:
                results.append(await task)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Probe tool failed (%s): %s", agent_id, exc)
        return results

    @staticmethod
    def _merge_probe_results(trace: Dict[str, Any], probe_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not probe_runs:
            return trace
        merged = dict(trace)
        merged.setdefault("tool_results", [])
        merged["tool_results"] = list(merged.get("tool_results") or []) + probe_runs
        merged.setdefault("evidences", [])
        evidences = list(merged.get("evidences") or [])
        for run in probe_runs:
            result = run.get("result") or {}
            for chunk in result.get("evidences") or []:
                if isinstance(chunk, dict):
                    evidences.append(chunk)
        merged["evidences"] = evidences
        return merged

    def _merge_worker_traces(self, worker_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "question": None,
            "graph_context": None,
            "adapter_metadata": {},
            "plan_steps": [],
            "graph_traversals": [],
            "reasoning_steps": [],
            "evidences": [],
            "tool_results": [],
            "pending_external": [],
            "think_notes": [],
            "coverage_metrics": {},
        }
        seen_evidence: set[str] = set()

        for outcome in worker_outcomes:
            agent_session = outcome.get("agent_session") or {}
            agent_id = str(agent_session.get("agent_id") or "")
            trace = outcome.get("trace") or {}
            if merged["question"] is None:
                merged["question"] = trace.get("question")
            if merged["graph_context"] is None:
                merged["graph_context"] = trace.get("graph_context")
            merged["adapter_metadata"] = trace.get("adapter_metadata") or merged["adapter_metadata"]
            merged["plan_steps"].extend(list(trace.get("plan_steps") or []))
            merged["graph_traversals"].extend(list(trace.get("graph_traversals") or []))
            for step in trace.get("reasoning_steps") or []:
                if not isinstance(step, dict):
                    continue
                diagnostics = step.get("diagnostics")
                if isinstance(diagnostics, dict) and agent_id:
                    diagnostics.setdefault("agent_id", agent_id)
                merged["reasoning_steps"].append(step)
            merged["tool_results"].extend(list(trace.get("tool_results") or []))
            merged["pending_external"].extend(list(trace.get("pending_external") or []))
            merged["think_notes"].extend(list(trace.get("think_notes") or []))

            for ev in trace.get("evidences") or []:
                if not isinstance(ev, dict):
                    continue
                chunk_id = str(ev.get("chunk_id") or "")
                source = str(ev.get("source") or "")
                if not chunk_id:
                    continue
                key = f"{source}::{chunk_id}"
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                if agent_id:
                    provenance = ev.get("provenance")
                    if isinstance(provenance, dict):
                        provenance.setdefault("agent_id", agent_id)
                    else:
                        ev["provenance"] = {"agent_id": agent_id}
                merged["evidences"].append(ev)

        merged["question"] = str(merged["question"] or "")
        merged["graph_context"] = merged["graph_context"] or {}
        return merged

    async def _run_lead_tools(
        self,
        question: str,
        merged_trace: Dict[str, Any],
        *,
        base_context: Optional[GraphQueryContext],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, self.settings.lead_tool_concurrency))
        evidences = list(merged_trace.get("evidences") or [])
        graph_context = merged_trace.get("graph_context") or (base_context.model_dump(exclude_none=True) if base_context else {})

        async def _invoke(tool_name: str) -> Dict[str, Any]:
            async with semaphore:
                payload = {
                    "question": question,
                    "plan_step": f"lead:{tool_name}:{uuid.uuid4().hex[:6]}",
                    "context_evidences": evidences,
                    "adapter": self.adapter,
                    "access_scope": (graph_context or {}).get("access_scope") if isinstance(graph_context, dict) else None,
                    "extra": {},
                    "graph_context": graph_context,
                    "coverage_metrics": merged_trace.get("coverage_metrics") or {},
                }
                result = await self.tool_manager.invoke(tool_name, payload=payload)  # type: ignore[union-attr]
                return {
                    "tool_name": tool_name,
                    "channel": result.channel,
                    "result": result.model_dump(),
                }

        tasks = [asyncio.create_task(_invoke(name)) for name in self.settings.lead_tool_names]
        results: List[Dict[str, Any]] = []
        for task in asyncio.as_completed(tasks):
            try:
                results.append(await task)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lead tool failed (%s): %s", question, exc)
        return results

    @staticmethod
    def _merge_lead_tools(merged_trace: Dict[str, Any], lead_tool_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not lead_tool_runs:
            return merged_trace
        merged = dict(merged_trace)
        merged.setdefault("tool_results", [])
        merged["tool_results"] = list(merged.get("tool_results") or []) + lead_tool_runs
        merged.setdefault("evidences", [])
        evidences = list(merged.get("evidences") or [])
        for run in lead_tool_runs:
            result = run.get("result") or {}
            for chunk in result.get("evidences") or []:
                if isinstance(chunk, dict):
                    evidences.append(chunk)
        merged["evidences"] = evidences
        merged.setdefault("coverage_metrics", {})
        merged["coverage_metrics"]["lead_tools"] = [run.get("tool_name") for run in lead_tool_runs]
        return merged
