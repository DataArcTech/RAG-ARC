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
from core.deepsearch.utils.compression import compact_evidences, resolve_compaction_config
from core.deepsearch.utils.evidence_kinds import count_evidences_by_kind, is_primary_evidence
from config.core.deepsearch.reasoning_defaults import (
    THINK_RECENT_TOOL_RUNS_MAX,
    THINK_RECENT_TOOL_RUN_SUMMARY_MAX_CHARS,
    THINK_TOOL_CATALOG_ALWAYS_INCLUDE,
)
from core.graph_adapter.base import GraphAccessScope, GraphDeepSearchAdapter
from core.graph_adapter.scope_provider import require_scope

from .graph_loop_runtime import GraphLoopRuntimeMixin, _prioritize_tool_catalog
from .graph_loop_state import (
    _RUN_EVIDENCES,
    _RUN_EVIDENCE_LOCK,
    _RUN_REFLECT_COUNT,
    _RUN_THINK_COUNT,
    _RUN_THINK_TOOL_SIGNATURES,
    _RUN_TOTAL_STEPS,
    _RUN_PLAN_STATE,
    _run_evidence_state,
)

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
        self._tool_context_max_chars = self._resolve_tool_context_max_chars(strategy_config)
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
        think_sig_token = _RUN_THINK_TOOL_SIGNATURES.set(set())
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
                    max_chars=int(self._think_config.get("recent_tool_run_summary_max_chars") or 0),
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
                    break
        finally:
            _RUN_EVIDENCES.reset(evidences_token)
            _RUN_EVIDENCE_LOCK.reset(lock_token)
            _RUN_TOTAL_STEPS.reset(total_steps_token)
            _RUN_THINK_COUNT.reset(think_count_token)
            _RUN_REFLECT_COUNT.reset(reflect_count_token)
            _RUN_THINK_TOOL_SIGNATURES.reset(think_sig_token)
            _RUN_PLAN_STATE.reset(plan_token)

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
        }

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
        recent_tool_run_summary_max_chars = _get(think, "recent_tool_run_summary_max_chars")
        if recent_tool_run_summary_max_chars is None:
            recent_tool_run_summary_max_chars = THINK_RECENT_TOOL_RUN_SUMMARY_MAX_CHARS

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
            "recent_tool_run_summary_max_chars": max(0, int(recent_tool_run_summary_max_chars)),
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
        compression = self._resolve_strategy_compression_schema()
        if compression:
            current = context_metadata.get("compression") if isinstance(context_metadata.get("compression"), dict) else None
            context_metadata["compression"] = self._merge_compression_defaults(defaults=compression, overrides=current)
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
        compression = self._resolve_strategy_compression_schema()
        if compression:
            current = metadata.get("compression") if isinstance(metadata.get("compression"), dict) else None
            metadata["compression"] = self._merge_compression_defaults(defaults=compression, overrides=current)
        return context.model_copy(update={"metadata": metadata})

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
        compaction = resolve_compaction_config(
            branch="tool_context",
            graph_context=context,
            extra=extra,
            default_max_items=self._tool_context_max_items,
            default_max_chars=self._tool_context_max_chars,
        )
        compacted, _ = compact_evidences(
            evidences=evidences,
            cfg=compaction,
            question=question,
            extra=extra or {},
            include_triple_count=False,
        )
        payload: Dict[str, Any] = {
            "plan_step": plan_step_id,
            "question": question,
            "context_evidences": list(compacted),
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
        if isinstance(config, dict):
            if "tool_context_max_evidences" not in config:
                raise ValueError("strategy_config.tool_context_max_evidences is required for GraphReasoningLoop")
            value = config["tool_context_max_evidences"]
        else:
            if not hasattr(config, "tool_context_max_evidences"):
                raise ValueError("strategy_config.tool_context_max_evidences is required for GraphReasoningLoop")
            value = getattr(config, "tool_context_max_evidences")
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError("strategy_config.tool_context_max_evidences must be an integer") from exc
        if int_value < 0:
            raise ValueError("strategy_config.tool_context_max_evidences must be >= 0")
        return int_value

    @staticmethod
    def _resolve_tool_context_max_chars(config: Any) -> int:
        if isinstance(config, dict):
            if "tool_context_max_chars" not in config:
                raise ValueError("strategy_config.tool_context_max_chars is required for GraphReasoningLoop")
            value = config["tool_context_max_chars"]
        else:
            if not hasattr(config, "tool_context_max_chars"):
                raise ValueError("strategy_config.tool_context_max_chars is required for GraphReasoningLoop")
            value = getattr(config, "tool_context_max_chars")
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError("strategy_config.tool_context_max_chars must be an integer") from exc
        if int_value < 0:
            raise ValueError("strategy_config.tool_context_max_chars must be >= 0")
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
