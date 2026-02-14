"""Graph-first reasoning loop that executes think-driven tool calls."""
import asyncio
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from encapsulation.data_model.deepsearch import (
    EvidenceChunk,
    GraphQueryContext,
    ReasoningStepRecord,
    ThinkNote,
    ToolExecutionLog,
    ToolResultPayload,
)
from core.deepsearch.memory.plan_state import PlanState, update_plan_from_think_notes
from core.deepsearch.tooling.budget import attach_tool_budget_metadata, get_tool_budget
from core.deepsearch.tooling.protocols import ToolInvoker
from core.deepsearch.trace import emit_trace
from core.deepsearch.utils.evidence_kinds import count_evidences_by_kind, is_primary_evidence
from core.deepsearch.utils.evidence_cards import evidence_cards
from config.core.deepsearch.reasoning_defaults import (
    REPORT_HARD_GATE_MIN_PRIMARY_PAGE_EVIDENCE,
    THINK_RECENT_TOOL_RUNS_MAX,
    THINK_TOOL_CATALOG_ALWAYS_INCLUDE,
)
from config.core.deepsearch.runtime_messages import MISSING_PRIMARY_EVIDENCE_HARD_GATE_MESSAGE
from core.graph_adapter.base import GraphAccessScope, GraphDeepSearchAdapter
from core.graph_adapter.scope_provider import require_scope

from .graph_loop_runtime import GraphLoopRuntimeMixin, _prioritize_tool_catalog
from .graph_loop_state import (
    _RUN_EVIDENCES,
    _RUN_EVIDENCE_LOCK,
    _RUN_REFLECT_COUNT,
    _RUN_THINK_COUNT,
    _RUN_TOTAL_STEPS,
    _RUN_PLAN_STATE,
    _RUN_TOOL_MEMO,
    _run_evidence_state,
)
from core.deepsearch.tooling.run_tool_memo import RunToolMemoizer

logger = logging.getLogger(__name__)


class GraphReasoningLoop(GraphLoopRuntimeMixin):
    """Run think-driven tool loops without plan-step traversal."""

    def __init__(
        self,
        adapter: GraphDeepSearchAdapter,
        llm_connector,
        strategy_config,
        *,
        tool_manager: ToolInvoker | None = None,
    ):
        self.adapter = adapter
        self.llm_connector = llm_connector
        self.strategy_config = strategy_config
        self.tool_manager = tool_manager
        self._adapter_metadata = self._resolve_adapter_metadata()
        self._think_config = self._build_think_config(strategy_config)
        self._tool_timeout = self._resolve_tool_timeout(strategy_config)
        self._tool_context_max_items = self._resolve_tool_context_max_items(strategy_config)
        self._coverage_expected_min_chunks = self._resolve_expected_min_chunks(strategy_config)

    async def run_think_loop(
        self,
        question: str,
        *,
        graph_context: Optional[GraphQueryContext] = None,
        initial_think_notes: Sequence[ThinkNote] | None = None,
        seed_evidences: Sequence[EvidenceChunk] | None = None,
        plan_items: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Run a think → tool-calls loop without plan-step traversal."""

        if not question:
            raise ValueError("question must be a non-empty string")
        if not self.tool_manager:
            raise RuntimeError("GraphReasoningLoop cannot invoke tools without a tool manager")

        context = self._prepare_graph_context(graph_context, question)
        evidences: List[EvidenceChunk] = list(seed_evidences or [])
        evidence_lock = asyncio.Lock()
        tool_runs: List[Dict[str, Any]] = []
        think_notes: List[Dict[str, Any]] = []
        reasoning_steps: List[ReasoningStepRecord] = []
        coverage_metrics: Dict[str, Any] = {}

        evidences_token = _RUN_EVIDENCES.set(evidences)
        lock_token = _RUN_EVIDENCE_LOCK.set(evidence_lock)
        total_steps_token = _RUN_TOTAL_STEPS.set(0)
        think_count_token = _RUN_THINK_COUNT.set(0)
        reflect_count_token = _RUN_REFLECT_COUNT.set(0)
        tool_memo = RunToolMemoizer()
        tool_memo_token = _RUN_TOOL_MEMO.set(tool_memo if tool_memo.enabled() else None)
        plan_state = PlanState()
        if isinstance(context.metadata, dict):
            plan_state.update(context.metadata.get("runtime_plan"))
        if plan_items:
            plan_state.update(plan_items)
        plan_token = _RUN_PLAN_STATE.set(plan_state)

        tool_catalog: List[Dict[str, Any]] = []
        available_tool_names: set[str] = set()
        limit = max(0, int(self._think_config.get("tool_catalog_max_items") or 0))
        if limit:
            from core.deepsearch.tooling import describe_available_tools

            registry = None
            try:
                registry = getattr(getattr(self.tool_manager, "local_registry", None), "tool_hint_registry", None)
            except Exception:
                registry = None
            all_hints = describe_available_tools(
                extra_hints=[],
                registry=registry,
                include_llm_tools=bool(self._think_config["include_llm_tools"]),
            )
            allowlist = self._think_config.get("tool_catalog_allowlist")
            if isinstance(allowlist, (list, tuple, set)):
                allowed = {str(name).strip() for name in allowlist if str(name).strip()}
                all_hints = [hint for hint in all_hints if str(hint.get("name") or "").strip() in allowed]
            tool_catalog = _prioritize_tool_catalog(
                all_hints,
                always_include=THINK_TOOL_CATALOG_ALWAYS_INCLUDE,
                limit=limit,
            )
            if self._think_config.get("tool_name"):
                tool_catalog = [
                    hint for hint in tool_catalog if str(hint.get("name") or "").strip() != self._think_config["tool_name"]
                ]
            for entry in tool_catalog:
                if isinstance(entry, dict) and entry.get("name"):
                    available_tool_names.add(str(entry["name"]))

        if initial_think_notes:
            for note in initial_think_notes:
                think_notes.append(note.model_dump(exclude_none=True))
            coverage_metrics = self._coverage_snapshot(
                evidences=evidences,
                completed_steps=0,
                total_steps=0,
            )
            tool_call_records, tool_call_summary = await self._execute_tool_calls_from_think(
                think_step_id="think_init",
                question=question,
                context=context,
                evidences=evidences,
                coverage_metrics=coverage_metrics,
                think_notes=initial_think_notes,
                tool_runs=tool_runs,
                think_notes_out=think_notes,
                available_tool_names=available_tool_names,
            )
            reasoning_steps.extend(tool_call_records)
            previous_tool_call_results = list(tool_call_summary.get("results") or [])
        else:
            previous_tool_call_results = []

        max_rounds = max(1, int(self._think_config.get("max_rounds_per_checkpoint") or 1))
        try:
            for round_idx in range(1, max_rounds + 1):
                coverage_metrics = self._coverage_snapshot(
                    evidences=evidences,
                    completed_steps=len(reasoning_steps),
                    total_steps=0,
                )
                recent_tool_runs = self._summarize_recent_tool_runs(
                    tool_runs,
                    max_items=int(self._think_config.get("recent_tool_runs_max") or 0),
                )
                next_count = _RUN_THINK_COUNT.get() + 1
                _RUN_THINK_COUNT.set(next_count)
                think_step_id = f"think_loop_{next_count:02d}"
                record = ReasoningStepRecord(
                    step_id=think_step_id,
                    description="Think loop checkpoint",
                    channel="graph",
                    status="running",
                )
                reasoning_steps.append(record)
                payload = self._build_tool_payload(
                    plan_step_id=think_step_id,
                    question=question,
                    context=context,
                    evidences=evidences,
                    coverage_hint=coverage_metrics,
                    extra={
                        "trigger": "think_loop",
                        "round": round_idx,
                        "available_tools": tool_catalog,
                        "previous_tool_call_results": previous_tool_call_results,
                        "recent_tool_runs": recent_tool_runs,
                        "current_plan": list(plan_state.items),
                        "think_mode": "normal",
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
                except Exception as exc:  # noqa: BLE001
                    record.status = "failed"
                    record.diagnostics.setdefault("reason", "think_loop")
                    record.diagnostics.setdefault("error", str(exc))
                    # Robustness: do not abort the whole DeepSearch run on a single think parsing/routing failure.
                    # Feed the error back into the next think checkpoint so the model can retry with corrected JSON.
                    previous_tool_call_results = [
                        {
                            "status": "failed",
                            "step_id": think_step_id,
                            "failure_reason": "think_tool_error",
                            "error": str(exc),
                            "suggested_next_steps": [
                                "Retry think and return ONLY valid JSON matching the schema.",
                                "If you propose tool calls, emit them under tool_calls (array of objects).",
                                "Do not stop (tool_calls=[]) until at least one read.pages evidence exists for report-style DeepSearch.",
                            ],
                        }
                    ]
                    await emit_trace(
                        "think",
                        "Think tool failed (continuing loop).",
                        meta={"stage": "think_loop_error", "round": round_idx, "error": str(exc)},
                    )
                    if round_idx < max_rounds:
                        continue
                    break

                await self._extend_shared_evidences(result.evidences)
                record.status = "done"
                record.output_summary = result.summary
                record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
                record.diagnostics.setdefault("reason", "think_loop")
                record.diagnostics.setdefault("latency_ms", latency_ms)
                record.tool_logs.append(
                    ToolExecutionLog(
                        tool_name=result.tool_name,
                        server_name=None,
                        arguments_snapshot={"trigger": "think_loop", "round": round_idx},
                        response_excerpt=result.summary if result.summary else None,
                        latency_ms=latency_ms,
                        graph_context=context,
                        extra={
                            "channel": "graph",
                            "profile": result.profile,
                            "determinism": result.determinism,
                            "trigger": "think_loop",
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
                    if update_plan_from_think_notes(plan_state, think_notes=result.think_notes):
                        await self._emit_plan_update(
                            plan_state=plan_state,
                            stage="think_loop",
                            plan_step_id=think_step_id,
                        )

                # Let the model explicitly stop the reasoning loop (report-style DeepSearch),
                # but never allow stopping before collecting at least one read.pages evidence.
                think_is_final = self._extract_is_final_from_think_notes(result.think_notes or [])
                if think_is_final is True:
                    if self._should_require_primary_page_evidence(context) and not self._has_primary_page_evidence(evidences):
                        previous_tool_call_results = [
                            {
                                "status": "failed",
                                "step_id": f"{think_step_id}_final_gate",
                                "failure_reason": "missing_primary_page_evidence",
                                "error": MISSING_PRIMARY_EVIDENCE_HARD_GATE_MESSAGE,
                                "suggested_next_steps": [
                                    "Finalization requires citeable page evidence: call read.pages at least once.",
                                    "Use explore + search.file to obtain a real file_id (UUID) if not already known.",
                                    "Use toc.tree / tree.root / tree.open / section.select (or search.scoped) to locate likely pages, then read.pages.",
                                ],
                            }
                        ]
                        await emit_trace(
                            "think",
                            "Finalization requested but evidence gate unmet (missing read.pages). Continuing think loop.",
                            meta={"stage": "final_gate", "round": round_idx},
                        )
                        continue
                    await emit_trace(
                        "think",
                        "Think requested finalization (stopping reasoning loop and proceeding to report).",
                        meta={"stage": "finalize", "round": round_idx},
                    )
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
                reasoning_steps.extend(tool_call_records)
                proposed = int(tool_call_summary.get("proposed") or 0)
                previous_tool_call_results = list(tool_call_summary.get("results") or [])
                if proposed <= 0:
                    # Hard evidence gate (report-style DeepSearch): do not stop the loop until we have at least one
                    # successful read.pages evidence. This keeps final reports grounded in full-page context.
                    if self._should_require_primary_page_evidence(context) and not self._has_primary_page_evidence(evidences):
                        previous_tool_call_results = [
                            {
                                "status": "failed",
                                "step_id": f"{think_step_id}_evidence_gate",
                                "failure_reason": "missing_primary_page_evidence",
                                "error": MISSING_PRIMARY_EVIDENCE_HARD_GATE_MESSAGE,
                                "suggested_next_steps": [
                                    "Use explore + search.file to obtain a real file_id (UUID).",
                                    "Use toc.tree / tree.root / tree.open / section.select (or search.scoped) to locate likely pages.",
                                    "Call read.pages on those pages (at least once) before stopping.",
                                ],
                            }
                        ]
                        await emit_trace(
                            "think",
                            "Evidence gate: missing read.pages (continuing think loop).",
                            meta={"stage": "evidence_gate", "round": round_idx},
                        )
                        continue
                    break
        finally:
            _RUN_EVIDENCES.reset(evidences_token)
            _RUN_EVIDENCE_LOCK.reset(lock_token)
            _RUN_TOTAL_STEPS.reset(total_steps_token)
            _RUN_THINK_COUNT.reset(think_count_token)
            _RUN_REFLECT_COUNT.reset(reflect_count_token)
            _RUN_PLAN_STATE.reset(plan_token)
            _RUN_TOOL_MEMO.reset(tool_memo_token)

        coverage_metrics = self._coverage_snapshot(
            evidences=evidences,
            completed_steps=len(reasoning_steps),
            total_steps=0,
        )
        return {
            "question": question,
            "graph_context": context.model_dump(exclude_none=True),
            "adapter_metadata": self._adapter_metadata,
            "plan_steps": [],
            "graph_traversals": [],
            "reasoning_steps": [record.model_dump() for record in reasoning_steps],
            "evidences": [chunk.model_dump() for chunk in evidences],
            "tool_results": tool_runs,
            "think_notes": think_notes,
            "runtime_plan": plan_state.to_payload(),
            "coverage_metrics": coverage_metrics,
            "tool_memoization": tool_memo.stats(),
        }

    @staticmethod
    def _extract_is_final_from_think_notes(think_notes: Sequence[ThinkNote]) -> bool | None:
        if not think_notes:
            return None
        for note in reversed(list(think_notes)):
            raw = note.metadata.get("raw") if isinstance(note.metadata, dict) else None
            if not isinstance(raw, dict):
                continue
            if "is_final" not in raw:
                continue
            return bool(raw.get("is_final"))
        return None

    @staticmethod
    def _should_require_primary_page_evidence(context: GraphQueryContext) -> bool:
        meta = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
        report_needed = meta.get("report_needed")
        if report_needed is False:
            return False
        style = str(meta.get("report_style") or "").strip().lower()
        if style not in {"deepsearch", "research"}:
            return False
        try:
            return int(REPORT_HARD_GATE_MIN_PRIMARY_PAGE_EVIDENCE) >= 1
        except Exception:
            return True

    @staticmethod
    def _has_primary_page_evidence(evidences: Sequence[EvidenceChunk]) -> bool:
        for ev in evidences or []:
            try:
                if ev.source == "read.pages":
                    return True
            except Exception:
                continue
        return False

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

        recent_tool_runs_max = _get(think, "recent_tool_runs_max")
        if recent_tool_runs_max is None:
            recent_tool_runs_max = THINK_RECENT_TOOL_RUNS_MAX

        cfg = {
            "tool_name": str(_get(think, "tool_name") or "").strip(),
            "cadence": int(_get(think, "every_n_steps") or 0),
            "min_coverage": float(_get(think, "min_coverage")),
            "always_run": bool(_get(think, "always_run") or False),
            "enable_tool_calls": bool(_get(think, "enable_tool_calls")),
            "max_tool_calls": int(_get(think, "max_tool_calls")),
            "tool_call_concurrency": int(_get(think, "tool_call_concurrency")),
            "tool_catalog_max_items": int(_get(think, "tool_catalog_max_items")),
            "tool_catalog_allowlist": _get(think, "tool_catalog_allowlist"),
            "recent_tool_runs_max": max(0, int(recent_tool_runs_max)),
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
            try:
                return raw.model_dump(exclude_none=True)
            except TypeError:
                return raw.model_dump()
        if is_dataclass(raw):
            return asdict(raw)
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    def _build_graph_context(
        self,
        question: str,
        *,
        access_scope: GraphAccessScope,
        metadata: Optional[Dict[str, Any]] = None,
        seed_entities: Optional[Sequence[str]] = None,
    ) -> GraphQueryContext:
        adapter_name = "graph_adapter"
        meta = self._adapter_metadata
        if isinstance(meta, dict) and meta.get("adapter_name"):
            adapter_name = str(meta.get("adapter_name"))
        context_metadata = dict(metadata or {})
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
    ) -> GraphQueryContext:
        if context is None:
            scope = require_scope()
            context = self._build_graph_context(question, access_scope=scope)
        else:
            try:
                context = context.model_copy()
            except AttributeError:
                context = GraphQueryContext(**context.model_dump())

        scope = context.resolve_scope()
        if scope is None:
            scope = require_scope()
            context = context.model_copy(update={"access_scope": scope})

        metadata = dict(context.metadata or {})
        return context.model_copy(update={"metadata": metadata})

    def _build_tool_payload(
        self,
        *,
        plan_step_id: str,
        question: str,
        context: GraphQueryContext,
        evidences: Sequence[EvidenceChunk],
        coverage_hint: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # NOTE: Do not inline full evidence text in tool contexts.
        # EvidencePool/EvidenceBank stores full content; tools see cards (metadata-only) to decide next actions.
        cards = evidence_cards(evidences)
        max_items = max(0, int(self._tool_context_max_items))
        if max_items:
            cards = cards[-max_items:]
        payload: Dict[str, Any] = {
            "plan_step": plan_step_id,
            "question": question,
            "context_evidences": list(cards),
            "adapter": self.adapter,
            "access_scope": context.access_scope,
            "graph_context": context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_hint or {},
            "extra": extra or {},
        }
        budget = get_tool_budget()
        if budget is not None:
            attach_tool_budget_metadata(graph_context=payload["graph_context"], snapshot=budget.snapshot())
        return payload

    @staticmethod
    def _resolve_tool_context_max_items(config: Any) -> int:
        # NOTE: We used to require `tool_context_max_evidences` and default it to a small
        # recency window. This caused older evidence cards to "disappear" from the LLM's
        # decision context and led to premature stopping / incomplete reads.
        #
        # New default: unlimited cards (0 means "do not truncate").
        if isinstance(config, dict):
            value = config.get("tool_context_max_evidences", 0)
        else:
            value = getattr(config, "tool_context_max_evidences", 0)
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError("strategy_config.tool_context_max_evidences must be an integer") from exc
        if int_value < 0:
            raise ValueError("strategy_config.tool_context_max_evidences must be >= 0")
        return int_value

    @staticmethod
    def _resolve_expected_min_chunks(config: Any) -> int:
        if isinstance(config, dict):
            if "coverage_expected_min_chunks" not in config:
                raise ValueError("strategy_config.coverage_expected_min_chunks is required for GraphReasoningLoop")
            value = config["coverage_expected_min_chunks"]
        else:
            if not hasattr(config, "coverage_expected_min_chunks"):
                raise ValueError("strategy_config.coverage_expected_min_chunks is required for GraphReasoningLoop")
            value = getattr(config, "coverage_expected_min_chunks")
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError("strategy_config.coverage_expected_min_chunks must be an integer") from exc
        if int_value <= 0:
            raise ValueError("strategy_config.coverage_expected_min_chunks must be >= 1")
        return int_value

    @staticmethod
    def _resolve_tool_timeout(config: Any) -> float:
        if isinstance(config, dict):
            if "tool_timeout_seconds" not in config:
                raise ValueError("strategy_config.tool_timeout_seconds is required for GraphReasoningLoop")
            value = config["tool_timeout_seconds"]
        else:
            if not hasattr(config, "tool_timeout_seconds"):
                raise ValueError("strategy_config.tool_timeout_seconds is required for GraphReasoningLoop")
            value = getattr(config, "tool_timeout_seconds")
        try:
            float_value = float(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError("strategy_config.tool_timeout_seconds must be numeric") from exc
        if float_value < 0:
            raise ValueError("strategy_config.tool_timeout_seconds must be >= 0")
        return float_value

    def _coverage_snapshot(
        self,
        *,
        evidences: Sequence[EvidenceChunk],
        completed_steps: int,
        total_steps: int,
    ) -> Dict[str, Any]:
        stats = count_evidences_by_kind(evidences)
        total = len(evidences)
        primary = stats.get("primary", 0)
        derived = stats.get("derived", 0)
        diagnostic = stats.get("diagnostic", 0)
        coverage_ratio = 0.0
        if self._coverage_expected_min_chunks > 0:
            coverage_ratio = min(1.0, primary / float(self._coverage_expected_min_chunks))
        coverage_score = coverage_ratio
        confidence_score = None
        missing_topics: List[str] = []
        if total_steps > 0:
            plan_progress_ratio = min(1.0, completed_steps / float(total_steps))
        else:
            plan_progress_ratio = 0.0
        return {
            "evidence_count": total,
            "primary_evidence_count": primary,
            "derived_evidence_count": derived,
            "diagnostic_evidence_count": diagnostic,
            "total_evidence_count": total,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "coverage_ratio": coverage_ratio,
            "plan_progress_ratio": plan_progress_ratio,
            "expected_min_chunks": self._coverage_expected_min_chunks,
            "coverage_score": coverage_score,
            "confidence_score": confidence_score,
            "missing_topics": missing_topics,
        }

    async def _extend_shared_evidences(self, additions: Sequence[EvidenceChunk]) -> None:
        evidences, lock = _run_evidence_state()
        if not additions:
            return
        async with lock:
            evidences.extend(additions)

    async def _snapshot_evidences(self) -> List[EvidenceChunk]:
        evidences, lock = _run_evidence_state()
        async with lock:
            return list(evidences)
