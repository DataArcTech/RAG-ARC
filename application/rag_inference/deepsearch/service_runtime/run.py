import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.deepsearch.report import DeepSearchQualityGate, QualityGateConfig
from core.deepsearch.state import DeepSearchState
from core.deepsearch.trace import emit_trace
from core.graph_adapter.base import GraphAccessScope
from core.utils.json_safe import json_safe

logger = logging.getLogger(__name__)


class DeepSearchServiceRunMixin:
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
        if getattr(self, "artifact_store", None) is not None:
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
        if getattr(self, "artifact_store", None) is not None:
            try:
                self.artifact_store.write_json(state.run_id, "plan_result.json", json_safe(plan_result))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist plan_result artifact: %s", exc, exc_info=True)
                state.append_error(f"artifact_store.write_json(plan_result.json) failed: {exc}", stage="persist")
        state.record_plan(plan_result)

        plan_payload = plan_result.get("plan") or {}
        plan_steps = plan_payload.get("steps") or []
        if not isinstance(plan_steps, list) or not plan_steps:
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
        reasoning_context = self._attach_file_scope_hints(
            reasoning_context,
            question=normalized_question,
        )

        routing_cfg = self._resolve_deterministic_routing_config()
        classification: Dict[str, Any] | None = None
        if isinstance(routing_cfg, dict) and routing_cfg.get("enabled"):
            classification = await self._classify_question(question=normalized_question, routing_cfg=routing_cfg)
            if classification:
                reasoning_context.metadata.setdefault("question_classification", dict(classification))
                await emit_trace(
                    "think",
                    "\n".join(
                        [
                            "Question classification (computable routing).",
                            json.dumps(json_safe(classification), ensure_ascii=False),
                        ]
                    ),
                    meta={"stage": "question_classification", "classification": json_safe(classification)},
                )

        await self._emit_initial_think(
            question=normalized_question,
            scope=scope,
            reasoning_context=reasoning_context,
            plan_steps=plan_steps,
        )

        quality_cfg = self._resolve_quality_gate_config()
        if isinstance(metadata, dict):
            raw_overrides = metadata.get("quality_loop")
        else:
            raw_overrides = None
        if isinstance(raw_overrides, dict) and raw_overrides:
            allowed = {
                "enabled",
                "max_rounds",
                "min_citation_sentence_coverage",
                "require_consistency",
                "max_uncited_sentences",
                "max_actions",
                "enable_llm_judge",
                "judge_temperature",
                "judge_max_retries",
                "judge_max_evidence_items",
                "judge_max_evidence_chars",
                "trigger_external_on_quality_failure",
            }
            merged = quality_cfg.model_dump()
            for key, value in raw_overrides.items():
                if key in allowed:
                    merged[key] = value
            quality_cfg = QualityGateConfig.model_validate(merged)
        quality_gate = DeepSearchQualityGate(
            getattr(getattr(self, "reporter", None), "llm_connector", None),
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

            gated_report = self._maybe_hard_gate_computable_question(
                question=normalized_question,
                reasoning_trace=cumulative_reasoning,
                classification=classification,
                routing_cfg=routing_cfg,
            )
            if gated_report is not None:
                final_report = gated_report
                break

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
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to emit external_channel trace: %s", exc, exc_info=True)
                    state.append_error(f"emit_trace(external_channel) failed: {exc}", stage="external_channel")

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
            if isinstance(quality_result, dict):
                if bool(quality_result.get("should_iterate")) and round_idx >= max_rounds:
                    quality_result["should_iterate"] = False
                    diagnostics = quality_result.get("diagnostics")
                    if not isinstance(diagnostics, dict):
                        diagnostics = {}
                        quality_result["diagnostics"] = diagnostics
                    diagnostics.setdefault("termination_reason", "max_rounds_reached")
                    diagnostics.setdefault("max_rounds", max_rounds)
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

            followup_steps = self._build_followup_plan_steps(actions=quality_result.get("actions") or [], round_idx=round_idx)
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
            state.reasoning_trace = cumulative_reasoning

        try:
            state.transition_stage(
                "done",
                metadata={
                    "rounds": len(quality_history) or 1,
                    "passed": bool((quality_history[-1].get("passed")) if quality_history else True),
                },
            )
        except Exception:
            pass

        snapshot = state.snapshot()
        snapshot.setdefault("plan_metadata", plan_result.get("plan"))
        if stage_timings:
            state.record_cost("stage_timings", stage_timings)
        if getattr(self, "artifact_store", None) is not None:
            try:
                self.artifact_store.write_json(state.run_id, "stage_timings.json", json_safe(stage_timings))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist stage_timings artifact: %s", exc, exc_info=True)
                state.append_error(f"artifact_store.write_json(stage_timings.json) failed: {exc}", stage="persist")
        self._persist_experiment_snapshot(
            question=normalized_question,
            plan=plan_result,
            reasoning=cumulative_reasoning or {},
            report=report,
            snapshot=snapshot,
            stage_timings=stage_timings,
        )
        if getattr(self, "artifact_store", None) is not None:
            try:
                self.artifact_store.write_json(state.run_id, "reasoning.json", json_safe(cumulative_reasoning or {}))
                self.artifact_store.write_json(state.run_id, "report.json", json_safe(report))
                if isinstance(report, dict) and isinstance(report.get("answer"), str):
                    self.artifact_store.write_text(state.run_id, "report.md", report.get("answer") or "")
                self.artifact_store.write_json(state.run_id, "state_snapshot.json", json_safe(snapshot))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist DeepSearch artifacts: %s", exc, exc_info=True)
                state.append_error(f"artifact_store persist failed: {exc}", stage="persist")
        logger.info(
            "DeepSearch run %s completed (owner=%s, timings=%s)",
            snapshot.get("run_id"),
            owner_id,
            stage_timings,
        )
        return {"plan": plan_result, "reasoning": cumulative_reasoning or {}, "report": report, "state": snapshot}

