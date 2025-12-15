"""Service façade wiring DeepSearch planner, reasoning loop, and reporting."""
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.scope_provider import require_scope
from core.deepsearch.plan import DeepSearchPlanner
from core.deepsearch.reasoning import GraphReasoningLoop
from core.deepsearch.gap import GapDetectionEngine
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.tooling.protocols import ToolInvoker
from encapsulation.deepsearch.external import ExternalSearchChannel
from core.deepsearch.state import DeepSearchState


logger = logging.getLogger(__name__)


class DeepSearchService:
    """Application-layer facade with a shared async run() entry point for FastAPI/CLI/MCP."""

    def __init__(
        self,
        planner: DeepSearchPlanner,
        graph_loop: GraphReasoningLoop,
        gap_detector: GapDetectionEngine,
        reporter: DeepSearchReporter,
        tool_manager: ToolInvoker,
        *,
        external_channel: ExternalSearchChannel | None = None,
        state_cls: Type[DeepSearchState] = DeepSearchState,
        config: Dict[str, Any] | None = None,
    ):
        # planner: ReAct/IterResearch plan generator
        self.planner = planner
        # graph_loop: Executes GraphDeepSearchAdapter-driven traversals and tool calls
        self.graph_loop = graph_loop
        # gap_detector: Decides whether external search should run based on coverage/confidence
        self.gap_detector = gap_detector
        # reporter: Produces final answers/highlights/evidence payloads
        self.reporter = reporter
        # tool_manager: Schedules local and MCP tools
        self.tool_manager = tool_manager
        # external_channel: Optional Serper/Tavily orchestration triggered by gap detection
        self.external_channel = external_channel
        # state_cls: Allows swapping in richer state trackers when needed
        self.state_cls = state_cls
        # config: Injected via config/application/deepsearch_config.py to keep construction data-driven
        self.config = self._coerce_config(config)
        self.experiment_output_dir = self._resolve_experiment_dir()

    async def run(
        self,
        question: str,
        *,
        owner_id: Optional[str] = None,
        access_scope: Optional[GraphAccessScope] = None,
        graph_context: Optional[GraphQueryContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        stage_listener: Optional[Callable[[Dict[str, Any], DeepSearchState], None]] = None,
    ) -> Dict[str, Any]:
        """Plan → Graph reasoning → (optional) Gap/External → Report."""

        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must be a non-empty string")

        scope = self._resolve_scope(owner_id=owner_id, access_scope=access_scope, graph_context=graph_context)
        state = self._build_state(run_id=run_id, stage_listener=stage_listener)
        state.record_request_metadata(metadata)
        state.record_cost(
            "request_context",
            self._json_safe(
                {
                    "owner_id": owner_id,
                    "access_scope": getattr(scope, "scope_id", None),
                    "metadata": metadata or {},
                }
            ),
        )

        stage_timings: Dict[str, Any] = {}

        plan_result = await self._execute_stage(
            "plan",
            self._plan_stage,
            state=state,
            stage_timings=stage_timings,
            question=normalized_question,
            scope=scope,
        )
        state.record_plan(plan_result)
        plan_payload = plan_result.get("plan") or {}
        plan_steps: Sequence[Dict[str, Any]] = plan_payload.get("steps") or []
        graph_context_payload = plan_payload.get("graph_context") or {}
        reasoning_context = GraphQueryContext(**graph_context_payload)
        reasoning_context = self._attach_run_metadata(
            reasoning_context,
            run_id=state.run_id,
            metadata=metadata,
            external_allowed=self._external_allowed_flag(),
        )

        reasoning_trace = await self._execute_stage(
            "graph_reasoning",
            self._reasoning_stage,
            state=state,
            stage_timings=stage_timings,
            question=normalized_question,
            plan_steps=plan_steps,
            reasoning_context=reasoning_context,
        )
        state.record_reasoning(reasoning_trace)

        gap_result = await self._execute_stage(
            "gap_detection",
            self._gap_stage,
            state=state,
            stage_timings=stage_timings,
            reasoning_trace=reasoning_trace,
        )
        if gap_result:
            state.record_gap_result(gap_result)
            reasoning_trace["gap_result"] = gap_result

        external_payload = await self._execute_stage(
            "external_channel",
            self._run_external_if_needed,
            state=state,
            stage_timings=stage_timings,
            gap_result=gap_result,
            reasoning_trace=reasoning_trace,
            )
        external_logs: Sequence[Dict[str, Any]] = []
        external_evidences: Sequence[Dict[str, Any]] = []
        if external_payload:
            external_logs = external_payload.get("logs") or []
            external_evidences = external_payload.get("evidences") or []
        if external_logs:
            state.extend_external_calls(list(external_logs))

        report = await self._execute_stage(
            "report",
            self._report_stage,
            state=state,
            stage_timings=stage_timings,
            reasoning_trace=reasoning_trace,
            external_logs=external_evidences,
        )
        state.record_report(report)

        snapshot = state.snapshot()
        snapshot.setdefault("plan_metadata", plan_result.get("plan"))
        if stage_timings:
            state.record_cost("stage_timings", stage_timings)
        self._persist_experiment_snapshot(
            question=normalized_question,
            plan=plan_result,
            reasoning=reasoning_trace,
            report=report,
            snapshot=snapshot,
            stage_timings=stage_timings,
        )
        logger.info(
            "DeepSearch run %s completed (owner=%s, timings=%s)",
            snapshot.get("run_id"),
            owner_id,
            stage_timings,
        )
        return {
            "plan": plan_result,
            "reasoning": reasoning_trace,
            "report": report,
            "state": snapshot,
        }

    def _build_state(
        self,
        *,
        run_id: Optional[str],
        stage_listener: Optional[Callable[[Dict[str, Any], DeepSearchState], None]],
    ) -> DeepSearchState:
        kwargs: Dict[str, Any] = {"config_fingerprint": self._config_fingerprint()}
        if run_id:
            kwargs["run_id"] = run_id
        if stage_listener:
            kwargs["stage_listener"] = stage_listener
        try:
            return self.state_cls(**kwargs)
        except TypeError:
            # Backward-compatible fallback if custom state_cls does not accept stage_listener/run_id.
            state = self.state_cls(config_fingerprint=self._config_fingerprint())
            if run_id:
                try:
                    state.run_id = run_id  # type: ignore[misc]
                except Exception:
                    pass
            if stage_listener:
                try:
                    state.stage_listener = stage_listener  # type: ignore[misc]
                except Exception:
                    pass
            return state

    def _resolve_scope(
        self,
        *,
        owner_id: Optional[str],
        access_scope: Optional[GraphAccessScope],
        graph_context: Optional[GraphQueryContext],
    ) -> GraphAccessScope:
        if graph_context and graph_context.access_scope:
            return graph_context.access_scope
        if access_scope:
            return access_scope
        if owner_id:
            return GraphAccessScope(scope_id=str(owner_id), scope_type="owner")
        return require_scope()

    def _evaluate_gap(self, reasoning_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.gap_detector:
            return None
        try:
            return self.gap_detector.evaluate(reasoning_trace)
        except NotImplementedError:
            return None

    async def _run_external_if_needed(
        self,
        gap_result: Optional[Dict[str, Any]],
        reasoning_trace: Dict[str, Any],
    ) -> Optional[Dict[str, Sequence[Dict[str, Any]]]]:
        if not gap_result or not gap_result.get("should_trigger_external"):
            return None
        if not self.external_channel:
            return None
        tasks = list(reasoning_trace.get("pending_external") or [])
        if not tasks:
            synthesized = self._synthesize_external_tasks(
                question=reasoning_trace.get("question", ""),
                gap_result=gap_result,
                reasoning_trace=reasoning_trace,
            )
            if synthesized:
                tasks.extend(synthesized)
                reasoning_trace.setdefault("pending_external", []).extend(synthesized)
        if not tasks:
            return None
        try:
            return await self.external_channel.run(
                tasks,
                reasoning_trace=reasoning_trace,
                gap_result=gap_result,
            )
        except NotImplementedError:
            return None

    def _config_fingerprint(self) -> str:
        return str(self.config.get("fingerprint") or self.config.get("name") or "deepsearch-service")

    async def _plan_stage(self, *, question: str, scope: GraphAccessScope) -> Dict[str, Any]:
        return await self.planner.build_plan(question, access_scope=scope)

    async def _reasoning_stage(
        self,
        *,
        question: str,
        plan_steps: Sequence[Dict[str, Any]],
        reasoning_context: GraphQueryContext,
    ) -> Dict[str, Any]:
        return await self.graph_loop.run(
            question,
            plan_steps,
            graph_context=reasoning_context,
        )

    def _gap_stage(self, *, reasoning_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._evaluate_gap(reasoning_trace)

    def _report_stage(
        self,
        *,
        reasoning_trace: Dict[str, Any],
        external_logs: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return self.reporter.compose(reasoning_trace, external_logs)

    async def _execute_stage(
        self,
        label: str,
        func,
        *,
        state: DeepSearchState,
        stage_timings: Dict[str, Any],
        **kwargs: Any,
    ):
        start = time.perf_counter()
        try:
            result = func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("DeepSearch stage %s failed: %s", label, exc)
            state.append_error(str(exc), stage=label)
            state.mark_failed(label, details={"stage": label, "error": str(exc)})
            raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        stage_timings[label] = elapsed_ms
        state.record_cost(f"stage:{label}", {"latency_ms": elapsed_ms})
        return result

    @staticmethod
    def _attach_run_metadata(
        context: GraphQueryContext,
        *,
        run_id: str,
        metadata: Optional[Dict[str, Any]],
        external_allowed: Optional[bool] = None,
    ) -> GraphQueryContext:
        base = dict(context.metadata or {})
        base["run_id"] = run_id
        if external_allowed is not None:
            base["external_allowed"] = bool(external_allowed)
        if metadata:
            request_bucket = base.setdefault("request_metadata", {})
            if isinstance(request_bucket, dict):
                request_bucket.update(metadata)
        try:
            return context.model_copy(update={"metadata": base})
        except AttributeError:
            payload = context.model_dump(exclude_none=True)
            payload["metadata"] = base
            return GraphQueryContext(**payload)

    def _external_allowed_flag(self) -> bool:
        channel = self.external_channel
        if channel is None:
            return False
        checker = getattr(channel, "_is_enabled", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        enabled = getattr(channel, "enabled", None)
        return bool(enabled)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [DeepSearchService._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [DeepSearchService._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): DeepSearchService._json_safe(val) for key, val in value.items()}
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
            except TypeError:
                dumped = value.model_dump(exclude_none=True)
            return DeepSearchService._json_safe(dumped)
        if hasattr(value, "__dict__"):
            return DeepSearchService._json_safe(vars(value))
        return str(value)

    @staticmethod
    def _coerce_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        if hasattr(config, "model_dump"):
            try:
                return config.model_dump()
            except TypeError:
                return config.model_dump(exclude_none=True)
        if hasattr(config, "__dict__"):
            return {
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_")
            }
        return {"value": config}

    def _synthesize_external_tasks(
        self,
        *,
        question: str,
        gap_result: Dict[str, Any],
        reasoning_trace: Dict[str, Any],
    ) -> Sequence[Dict[str, Any]]:
        description = gap_result.get("reason") or "External search for missing coverage"
        normalized_question = (question or reasoning_trace.get("question") or "").strip()
        if not normalized_question:
            normalized_question = description
        metadata = {
            "provider": getattr(self.external_channel, "default_provider", None),
            "query": normalized_question,
            "source": "gap_detection",
            "gap_reason": gap_result.get("reason"),
            "missing_topics": gap_result.get("missing_topics") or [],
        }
        step_id = f"gap_web_{int(time.time() * 1000)}"
        return [
            {
                "step_id": step_id,
                "description": description,
                "channel": "web",
                "tool": "web.search",
                "metadata": metadata,
                "tool_args": {"query": normalized_question, "gap_reason": gap_result.get("reason")},
                "requires_external": True,
            }
        ]

    def _resolve_experiment_dir(self) -> Optional[Path]:
        candidate = None
        if isinstance(self.config, dict):
            candidate = self.config.get("experiment_output_dir")
        override = os.getenv("DEEPSEARCH_EXPERIMENT_OUTPUT_DIR")
        directory = override or candidate
        if not directory:
            return None
        return Path(str(directory)).expanduser()

    def _persist_experiment_snapshot(
        self,
        *,
        question: str,
        plan: Dict[str, Any],
        reasoning: Dict[str, Any],
        report: Dict[str, Any],
        snapshot: Dict[str, Any],
        stage_timings: Dict[str, Any],
    ) -> None:
        if not self.experiment_output_dir:
            return
        try:
            self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.warning("Failed to prepare experiment directory %s: %s", self.experiment_output_dir, exc)
            return

        plan_payload = plan.get("plan") or {}
        plan_id = plan.get("plan_id") or plan_payload.get("plan_id") or snapshot.get("plan_metadata", {}).get("plan_id")
        reasoning_steps = reasoning.get("reasoning_steps") or []
        experiment_record = {
            "question": question,
            "plan_id": plan_id,
            "config_fingerprint": snapshot.get("config_fingerprint"),
            "stage_timings": stage_timings,
            "coverage_metrics": reasoning.get("coverage_metrics"),
            "gap_result": reasoning.get("gap_result") or snapshot.get("gap_result"),
            "plan_steps": plan_payload.get("steps") or [],
            "reasoning_steps": reasoning_steps,
            "think_notes": reasoning.get("think_notes") or [],
            "tool_results": reasoning.get("tool_results") or [],
            "answer": report.get("answer"),
            "highlights": report.get("highlights"),
            "evidence_ids": [chunk.get("chunk_id") for chunk in report.get("evidences") or []],
            "request_metadata": snapshot.get("request_metadata"),
        }
        filename = plan_id or snapshot.get("run_id") or f"run_{int(time.time() * 1000)}"
        path = self.experiment_output_dir / f"{filename}.json"
        try:
            path.write_text(
                json.dumps(self._json_safe(experiment_record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.warning("Failed to persist DeepSearch experiment snapshot: %s", exc)
