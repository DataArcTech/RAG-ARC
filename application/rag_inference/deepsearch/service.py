"""Service façade wiring DeepSearch planner, reasoning loop, and reporting."""
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type, List

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.scope_provider import require_scope
from core.deepsearch.plan import DeepSearchPlanner
from core.deepsearch.reasoning import GraphReasoningLoop
from core.deepsearch.gap import GapDetectionEngine
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.report import DeepSearchQualityGate, QualityGateConfig
from core.deepsearch.tooling.protocols import ToolInvoker
from encapsulation.deepsearch.external import ExternalSearchChannel
from core.deepsearch.state import DeepSearchState
from core.deepsearch.trace import emit_trace
from core.utils.json_safe import json_safe
from application.deepsearch.artifacts import DeepSearchArtifactStore


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
        # external_channel: Optional Tavily orchestration triggered by gap detection
        self.external_channel = external_channel
        # state_cls: Allows swapping in richer state trackers when needed
        self.state_cls = state_cls
        # config: Injected via config/application/deepsearch_config.py to keep construction data-driven
        self.config = self._coerce_config(config)
        self.experiment_output_dir = self._resolve_experiment_dir()
        self.artifact_store = self._resolve_artifact_store()

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
        """Plan → (Iterative) Graph reasoning → Gap/External → Report → Quality gate → Iterate."""

        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must be a non-empty string")

        scope = self._resolve_scope(owner_id=owner_id, access_scope=access_scope, graph_context=graph_context)
        state = self._build_state(run_id=run_id, stage_listener=stage_listener)
        artifact_dir: str | None = None
        if self.artifact_store is not None:
            artifact_dir = str(self.artifact_store.ensure_run_dir(state.run_id))
        state.record_request_metadata(metadata)
        state.record_cost(
            "request_context",
            json_safe(
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
        if self.artifact_store is not None:
            try:
                self.artifact_store.write_json(state.run_id, "plan_result.json", json_safe(plan_result))
            except Exception:
                pass
        state.record_plan(plan_result)
        plan_payload = plan_result.get("plan") or {}
        plan_steps: Sequence[Dict[str, Any]] = plan_payload.get("steps") or []
        if not plan_steps:
            raise RuntimeError("Planner returned an empty plan (no fallback execution allowed).")
        graph_context_payload = plan_payload.get("graph_context") or {}
        reasoning_context = GraphQueryContext(**graph_context_payload)
        reasoning_context = self._attach_run_metadata(
            reasoning_context,
            run_id=state.run_id,
            metadata=metadata,
            external_allowed=self._external_allowed_flag(),
            artifact_dir=artifact_dir,
        )

        await self._emit_initial_think(
            question=normalized_question,
            scope=scope,
            reasoning_context=reasoning_context,
            plan_steps=plan_steps,
        )

        quality_cfg = self._resolve_quality_gate_config()
        quality_gate = DeepSearchQualityGate(
            getattr(self.reporter, "llm_connector", None),
            config=quality_cfg.model_dump(),
        )

        cumulative_reasoning: Dict[str, Any] | None = None
        external_evidences_all: List[Dict[str, Any]] = []
        final_report: Dict[str, Any] | None = None
        final_gap: Optional[Dict[str, Any]] = None
        quality_history: List[Dict[str, Any]] = []

        followup_steps: List[Dict[str, Any]] = []
        pending_rewrite_only = False
        max_rounds = max(1, int(quality_cfg.max_rounds))
        for round_idx in range(1, max_rounds + 1):
            # Round plan: base steps first; subsequent rounds run only follow-ups unless rewrite-only.
            active_steps = plan_steps if round_idx == 1 else followup_steps
            do_reasoning = bool(active_steps) and not pending_rewrite_only

            if do_reasoning:
                round_trace = await self._execute_stage(
                    f"graph_reasoning_r{round_idx}",
                    self._reasoning_stage,
                    state=state,
                    stage_timings=stage_timings,
                    question=normalized_question,
                    plan_steps=active_steps,
                    reasoning_context=reasoning_context,
                )
                if cumulative_reasoning is None:
                    cumulative_reasoning = round_trace
                    cumulative_reasoning["quality_loop"] = {"rounds": []}
                else:
                    cumulative_reasoning = self._merge_reasoning_traces(cumulative_reasoning, round_trace)
                cumulative_reasoning.setdefault("quality_loop", {}).setdefault("rounds", []).append(
                    {"round": round_idx, "plan_steps": list(active_steps)}
                )
                state.record_reasoning(cumulative_reasoning)
                self._surface_worker_failures(state, round_trace)
                try:
                    evidence_count = len(cumulative_reasoning.get("evidences") or [])
                except Exception:
                    evidence_count = 0
                await emit_trace(
                    "progress",
                    "\n".join(
                        [
                            f"Completed graph reasoning round {round_idx}.",
                            f"evidence_count={evidence_count}",
                        ]
                    ),
                    meta={"stage": "graph_reasoning", "round": round_idx, "evidence_count": evidence_count},
                )

            if cumulative_reasoning is None:
                raise RuntimeError("DeepSearch reasoning did not produce a trace")

            gap_result = await self._execute_stage(
                f"gap_detection_r{round_idx}",
                self._gap_stage,
                state=state,
                stage_timings=stage_timings,
                reasoning_trace=cumulative_reasoning,
            )
            if gap_result:
                final_gap = gap_result
                state.record_gap_result(gap_result)
                cumulative_reasoning["gap_result"] = gap_result
                await emit_trace(
                    "progress",
                    "\n".join(
                        [
                            f"Gap detection round {round_idx}.",
                            f"coverage_score={gap_result.get('coverage_score')}",
                            f"confidence_score={gap_result.get('confidence_score')}",
                            f"should_trigger_external={gap_result.get('should_trigger_external')}",
                            f"reason={gap_result.get('reason')}",
                            f"external_allowed={(gap_result.get('diagnostics') or {}).get('external_allowed')}",
                            f"external_decision_source={((gap_result.get('diagnostics') or {}).get('external_decision') or {}).get('source')}",
                            f"missing_topics={json.dumps(gap_result.get('missing_topics') or [], ensure_ascii=False)}",
                        ]
                    ),
                    meta={"stage": "gap_detection", "round": round_idx, "gap_result": json_safe(gap_result)},
                )

            external_payload = await self._execute_stage(
                f"external_channel_r{round_idx}",
                self._run_external_if_needed,
                state=state,
                stage_timings=stage_timings,
                gap_result=gap_result,
                reasoning_trace=cumulative_reasoning,
            )
            external_logs: Sequence[Dict[str, Any]] = []
            external_evidences: Sequence[Dict[str, Any]] = []
            if external_payload:
                external_logs = external_payload.get("logs") or []
                external_evidences = external_payload.get("evidences") or []
            if external_logs:
                state.extend_external_calls(list(external_logs))
            if external_evidences:
                external_evidences_all.extend([dict(item) for item in external_evidences if isinstance(item, dict)])
                await emit_trace(
                    "progress",
                    "\n".join(
                        [
                            f"External channel produced evidence (round {round_idx}).",
                            f"external_evidence_count={len(external_evidences)}",
                        ]
                    ),
                    meta={
                        "stage": "external_channel",
                        "round": round_idx,
                        "external_evidence_count": len(external_evidences),
                        "external_logs": json_safe(list(external_logs)),
                    },
                )
            elif gap_result and gap_result.get("reason") == "external_disabled":
                await emit_trace(
                    "progress",
                    "\n".join(
                        [
                            f"External channel was requested but is disabled (round {round_idx}).",
                            "No external tools were executed.",
                        ]
                    ),
                    meta={"stage": "external_channel", "round": round_idx, "blocked": True, "gap_result": json_safe(gap_result)},
                )
            elif gap_result and gap_result.get("should_trigger_external") is False:
                try:
                    await emit_trace(
                        "progress",
                        "\n".join(
                            [
                                f"External channel not triggered (round {round_idx}).",
                                f"external_allowed={(gap_result.get('diagnostics') or {}).get('external_allowed')}",
                                f"reason={gap_result.get('reason')}",
                            ]
                        ),
                        meta={"stage": "external_channel", "round": round_idx, "gap_result": json_safe(gap_result)},
                    )
                except Exception:
                    pass

            report = await self._execute_stage(
                f"report_r{round_idx}",
                self._report_stage,
                state=state,
                stage_timings=stage_timings,
                reasoning_trace=cumulative_reasoning,
                external_logs=external_evidences_all,
            )
            final_report = report
            state.record_report(report)
            await emit_trace(
                "progress",
                "\n".join(
                    [
                        f"Draft report generated (round {round_idx}).",
                        f"answer_length={len((report.get('answer') or '') if isinstance(report, dict) else '')}",
                        f"evidence_count={len((report.get('evidences') or []) if isinstance(report, dict) else [])}",
                    ]
                ),
                meta={"stage": "report", "round": round_idx},
            )

            structured_report = report.get("structured_report") if isinstance(report, dict) else None
            evidences_for_gate = list(report.get("evidences") or []) if isinstance(report, dict) else []
            quality_result = await self._execute_stage(
                f"quality_gate_r{round_idx}",
                self._quality_gate_stage,
                state=state,
                stage_timings=stage_timings,
                gate=quality_gate,
                question=normalized_question,
                structured_report=structured_report,
                evidences=evidences_for_gate,
                gap_result=final_gap,
                round_idx=round_idx,
            )
            if isinstance(report, dict):
                report.setdefault("metadata", {}).setdefault("quality_gate", quality_result)
            state.record_quality_gate(quality_result)
            quality_history.append(quality_result)
            await emit_trace(
                "progress",
                "\n".join(
                    [
                        f"Quality gate round {round_idx}.",
                        f"passed={quality_result.get('passed')}",
                        f"should_iterate={quality_result.get('should_iterate')}",
                        f"actions={json.dumps(quality_result.get('actions') or [], ensure_ascii=False)}",
                    ]
                ),
                meta={"stage": "quality_gate", "round": round_idx, "quality_result": json_safe(quality_result)},
            )

            passed = bool(quality_result.get("passed"))
            should_iterate = bool(quality_result.get("should_iterate")) and round_idx < max_rounds
            if passed or not should_iterate:
                break

            followup_steps = self._build_followup_plan_steps(
                actions=quality_result.get("actions") or [],
                round_idx=round_idx,
            )
            pending_rewrite_only = not bool(followup_steps)
            if followup_steps:
                await emit_trace(
                    "write_outline",
                    "\n".join(
                        [
                            f"Plan update from quality gate (round {round_idx + 1}).",
                            json.dumps(json_safe(followup_steps), ensure_ascii=False, indent=2, default=str),
                        ]
                    ),
                    meta={"stage": "plan_update", "round": round_idx + 1, "followup_steps": json_safe(followup_steps)},
                )
            extra_external_tasks = self._build_external_tasks_from_actions(
                actions=quality_result.get("actions") or [],
                round_idx=round_idx,
                question=normalized_question,
            )
            if extra_external_tasks:
                extra_payload = await self._execute_stage(
                    f"external_channel_post_gate_r{round_idx}",
                    self._run_external_tasks_direct,
                    state=state,
                    stage_timings=stage_timings,
                    tasks=extra_external_tasks,
                    reasoning_trace=cumulative_reasoning,
                    gap_result=final_gap,
                )
                if extra_payload:
                    extra_logs = extra_payload.get("logs") or []
                    extra_evs = extra_payload.get("evidences") or []
                    if extra_logs:
                        state.extend_external_calls(list(extra_logs))
                    if extra_evs:
                        external_evidences_all.extend([dict(item) for item in extra_evs if isinstance(item, dict)])

        if final_report is None:
            raise RuntimeError("DeepSearch did not produce a report")
        report = final_report
        if cumulative_reasoning is not None:
            cumulative_reasoning.setdefault("quality_loop", {})["quality_gate_history"] = quality_history
            cumulative_reasoning.setdefault("quality_loop", {})["external_evidence_count"] = len(external_evidences_all)
            # Avoid transitioning state back to "reasoned" after emitting the final report.
            state.reasoning_trace = cumulative_reasoning

        snapshot = state.snapshot()
        snapshot.setdefault("plan_metadata", plan_result.get("plan"))
        if stage_timings:
            state.record_cost("stage_timings", stage_timings)
        if self.artifact_store is not None:
            try:
                self.artifact_store.write_json(state.run_id, "stage_timings.json", json_safe(stage_timings))
            except Exception:
                pass
        self._persist_experiment_snapshot(
            question=normalized_question,
            plan=plan_result,
            reasoning=cumulative_reasoning or {},
            report=report,
            snapshot=snapshot,
            stage_timings=stage_timings,
        )
        if self.artifact_store is not None:
            try:
                self.artifact_store.write_json(state.run_id, "reasoning.json", json_safe(cumulative_reasoning or {}))
                self.artifact_store.write_json(state.run_id, "report.json", json_safe(report))
                if isinstance(report, dict) and isinstance(report.get("answer"), str):
                    self.artifact_store.write_text(state.run_id, "report.md", report.get("answer") or "")
                self.artifact_store.write_json(state.run_id, "state_snapshot.json", json_safe(snapshot))
            except Exception:
                pass
        logger.info(
            "DeepSearch run %s completed (owner=%s, timings=%s)",
            snapshot.get("run_id"),
            owner_id,
            stage_timings,
        )
        return {
            "plan": plan_result,
            "reasoning": cumulative_reasoning or {},
            "report": report,
            "state": snapshot,
        }

    async def _emit_initial_think(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        plan_steps: Sequence[Dict[str, Any]],
    ) -> None:
        if not self.tool_manager:
            return

        try:
            total_steps = len(plan_steps) if plan_steps is not None else 0
        except Exception:
            total_steps = 0

        payload = {
            "question": question,
            "plan_step": "think_init",
            "context_evidences": [],
            "adapter": getattr(self.graph_loop, "adapter", None),
            "access_scope": scope,
            "extra": {
                "trigger": "initial_think",
                "plan_steps": list(plan_steps),
            },
            "graph_context": reasoning_context.model_dump(exclude_none=True),
            "coverage_metrics": {
                "evidence_count": 0,
                "unique_source_count": 0,
                "completed_steps": 0,
                "total_steps": total_steps,
                "coverage_ratio": 0.0,
                "plan_progress_ratio": 0.0,
                "expected_min_chunks": int(self.config["coverage_expected_min_chunks"]),
                "coverage_score": 0.0,
                "confidence_score": None,
                "missing_topics": [],
            },
        }
        try:
            result = await self.tool_manager.invoke(self._tool_names()["think_tool"], payload=payload)
        except Exception:
            return

        notes = getattr(result, "think_notes", None)
        if not notes:
            return

        lines: List[str] = ["Initial think checkpoint (before execution)."]
        for idx, note in enumerate(notes, start=1):
            lines.append(f"note_{idx}. reasoning={note.reasoning}")
            if note.next_actions:
                lines.append(f"note_{idx}. next_actions={note.next_actions}")
            if note.coverage_delta is not None:
                lines.append(f"note_{idx}. coverage_delta={note.coverage_delta}")
            if note.confidence_delta is not None:
                lines.append(f"note_{idx}. confidence_delta={note.confidence_delta}")
            missing = None
            if isinstance(note.metadata, dict):
                missing = note.metadata.get("missing_topics")
            if isinstance(missing, list) and missing:
                lines.append(f"note_{idx}. missing_topics={missing}")

        await emit_trace(
            "think",
            "\n".join(lines),
            meta={"stage": "think_init", "plan_step": "think_init"},
        )

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
        return self.state_cls(**kwargs)

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
        return self.gap_detector.evaluate(reasoning_trace)

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
            return None
        return await self.external_channel.run(
            tasks,
            reasoning_trace=reasoning_trace,
            gap_result=gap_result,
        )

    def _config_fingerprint(self) -> str:
        if not isinstance(self.config, dict) or not str(self.config.get("fingerprint") or "").strip():
            raise ValueError("DeepSearchService config fingerprint is required")
        return str(self.config["fingerprint"])

    async def _plan_stage(self, *, question: str, scope: GraphAccessScope) -> Dict[str, Any]:
        return await self.planner.build_plan(question, access_scope=scope)

    async def _reasoning_stage(
        self,
        *,
        question: str,
        plan_steps: Sequence[Dict[str, Any]],
        reasoning_context: GraphQueryContext,
        settings_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        runner = getattr(self.graph_loop, "run", None)
        if not callable(runner):
            raise RuntimeError("graph_loop does not expose an async run() method")
        kwargs: Dict[str, Any] = {"graph_context": reasoning_context}
        try:
            sig = inspect.signature(runner)
            if "settings_override" in sig.parameters and settings_override:
                kwargs["settings_override"] = settings_override
        except Exception:
            pass
        return await runner(question, plan_steps, **kwargs)

    def _gap_stage(self, *, reasoning_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._evaluate_gap(reasoning_trace)

    @staticmethod
    def _surface_worker_failures(state: DeepSearchState, reasoning_trace: Dict[str, Any]) -> None:
        """Copy non-fatal multi-agent failures into state.errors for observability."""

        if not isinstance(reasoning_trace, dict):
            return
        coverage = reasoning_trace.get("coverage_metrics") or {}
        if not isinstance(coverage, dict):
            return
        errors = coverage.get("worker_errors") or []
        if not isinstance(errors, list) or not errors:
            return
        count = coverage.get("worker_error_count")
        try:
            count_int = int(count) if count is not None else len(errors)
        except (TypeError, ValueError):
            count_int = len(errors)
        previews: list[str] = []
        for entry in errors:
            if len(previews) >= 3:
                break
            if not isinstance(entry, dict):
                continue
            agent_id = entry.get("agent_id") or "worker"
            code = entry.get("error") or "error"
            previews.append(f"{agent_id}={code}")
        preview = ", ".join(previews)
        message = f"{count_int} worker agent(s) failed"
        if preview:
            message = f"{message}: {preview}"
        state.append_error(message, stage="graph_reasoning")

    def _report_stage(
        self,
        *,
        reasoning_trace: Dict[str, Any],
        external_logs: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return self.reporter.compose(reasoning_trace, external_logs)

    async def _quality_gate_stage(
        self,
        *,
        gate: DeepSearchQualityGate,
        question: str,
        structured_report: Optional[Dict[str, Any]],
        evidences: Sequence[Dict[str, Any]],
        gap_result: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> Dict[str, Any]:
        result = await gate.evaluate(
            question=question,
            structured_report=structured_report,
            evidences=evidences,
            gap_result=gap_result,
            external_allowed=self._external_allowed_flag(),
        )
        payload = result.model_dump()
        payload["round"] = round_idx
        return payload

    async def _run_external_tasks_direct(
        self,
        *,
        tasks: Sequence[Dict[str, Any]],
        reasoning_trace: Dict[str, Any],
        gap_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Sequence[Dict[str, Any]]]]:
        if not tasks or not self.external_channel:
            return None
        if not self._external_allowed_flag():
            return None
        return await self.external_channel.run(
            list(tasks),
            reasoning_trace=reasoning_trace,
            gap_result=gap_result,
        )

    def _resolve_quality_gate_config(self) -> QualityGateConfig:
        if not isinstance(self.config, dict) or not isinstance(self.config.get("quality_loop"), dict):
            raise ValueError("DeepSearchService missing required config.quality_loop")
        return QualityGateConfig.model_validate(self.config["quality_loop"])

    def _tool_names(self) -> Dict[str, str]:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config must be a dict")
        names = self.config.get("tool_names")
        if not isinstance(names, dict):
            raise ValueError("DeepSearchService config.tool_names is required")
        required = ("graph_channel_tool", "web_channel_tool", "text_channel_tool", "think_tool")
        missing = [key for key in required if not str(names.get(key) or "").strip()]
        if missing:
            raise ValueError(f"DeepSearchService config.tool_names missing: {missing}")
        return {k: str(names[k]).strip() for k in required}

    def _build_followup_plan_steps(self, *, actions: Sequence[Dict[str, Any]], round_idx: int) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        counter = 0
        graph_tool = self._tool_names()["graph_channel_tool"]
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action") != "graph_search":
                continue
            query = str(action.get("query") or "").strip()
            if not query:
                continue
            counter += 1
            step_id = f"quality_graph_r{round_idx + 1}_{counter:02d}"
            steps.append(
                {
                    "step_id": step_id,
                    "description": f"Quality follow-up graph search: {query}",
                    "channel": "graph",
                    "metadata": {"source": "quality_gate", "round": round_idx + 1},
                    "tool": graph_tool,
                    "tool_args": {"query": query},
                    "requires_external": False,
                    "enabled": True,
                }
            )
        return steps

    def _build_external_tasks_from_actions(
        self,
        *,
        actions: Sequence[Dict[str, Any]],
        round_idx: int,
        question: str,
    ) -> List[Dict[str, Any]]:
        if not actions:
            return []
        web_tool = self._tool_names()["web_channel_tool"]
        tasks: List[Dict[str, Any]] = []
        for idx, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                continue
            if action.get("action") != "external_search":
                continue
            query = str(action.get("query") or "").strip()
            if not query:
                continue
            step_id = f"quality_web_r{round_idx + 1}_{idx:02d}"
            tasks.append(
                {
                    "step_id": step_id,
                    "description": str(action.get("rationale") or "Quality follow-up external search"),
                    "channel": "web",
                    "tool": web_tool,
                    "metadata": {
                        "provider": getattr(self.external_channel, "default_provider", None) if self.external_channel else None,
                        "query": query,
                        "source": "quality_gate",
                        "round": round_idx + 1,
                        "question": question,
                    },
                    "tool_args": {"query": query, "reason": "quality_gate"},
                    "requires_external": True,
                }
            )
        return tasks

    @staticmethod
    def _merge_reasoning_traces(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(base or {})
        for key in ("graph_traversals", "reasoning_steps", "tool_results", "think_notes", "pending_external"):
            merged_list = list(merged.get(key) or [])
            add_list = list(incoming.get(key) or [])
            merged[key] = merged_list + add_list

        def _merge_by_chunk_id(items: Sequence[Any]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                out.append(item)
            return out

        merged["evidences"] = _merge_by_chunk_id(list(merged.get("evidences") or []) + list(incoming.get("evidences") or []))

        # Merge plan steps by step_id for observability (execution order preserved by append).
        plan_steps: List[Dict[str, Any]] = []
        seen_steps: set[str] = set()
        for item in list(merged.get("plan_steps") or []) + list(incoming.get("plan_steps") or []):
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("step_id") or "").strip()
            if not step_id or step_id in seen_steps:
                continue
            seen_steps.add(step_id)
            plan_steps.append(item)
        merged["plan_steps"] = plan_steps

        # Coverage metrics: keep base and overlay incoming fields (incoming wins).
        base_cov = merged.get("coverage_metrics") if isinstance(merged.get("coverage_metrics"), dict) else {}
        inc_cov = incoming.get("coverage_metrics") if isinstance(incoming.get("coverage_metrics"), dict) else {}
        cov = dict(base_cov or {})
        cov.update(inc_cov or {})
        cov.setdefault("evidence_count", len(merged.get("evidences") or []))
        merged["coverage_metrics"] = cov
        return merged

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
        artifact_dir: Optional[str] = None,
    ) -> GraphQueryContext:
        base = dict(context.metadata or {})
        base["run_id"] = run_id
        if artifact_dir:
            base["artifact_dir"] = str(artifact_dir)
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

    def _resolve_experiment_dir(self) -> Optional[Path]:
        candidate = None
        if isinstance(self.config, dict):
            candidate = self.config.get("experiment_output_dir")
        directory = candidate
        if not directory:
            return None
        return Path(str(directory)).expanduser()

    def _resolve_artifact_store(self) -> DeepSearchArtifactStore | None:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config is required to resolve artifact store")
        configured = self.config.get("artifact_dir")
        if not configured:
            raise ValueError("DeepSearchService config.artifact_dir is required (no implicit env fallback).")
        return DeepSearchArtifactStore.from_root_dir(str(configured))

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
            "quality_gates": snapshot.get("quality_gates") or [],
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
                json.dumps(json_safe(experiment_record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.warning("Failed to persist DeepSearch experiment snapshot: %s", exc)
