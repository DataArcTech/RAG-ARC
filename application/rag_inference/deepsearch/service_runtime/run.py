import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext
from core.deepsearch.report import DeepSearchQualityGate, QualityGateConfig
from core.deepsearch.state import DeepSearchState
from core.deepsearch.trace import emit_trace, reset_trace_emitter
from core.graph_adapter.base import GraphAccessScope
from core.utils.json_safe import json_safe

from .artifact_dedupe_v2 import build_evidence_pool_v2, dedupe_reasoning_v2, dedupe_report_v2
from .artifact_views_v2 import build_v2_artifact_documents
from .trace_capture import attach_trace_capture

logger = logging.getLogger(__name__)


class DeepSearchServiceRunMixin:
    @staticmethod
    def _coerce_evidence_chunks(raw: Sequence[Dict[str, Any]] | None) -> List[EvidenceChunk]:
        items = raw or []
        out: List[EvidenceChunk] = []
        for item in items:
            if isinstance(item, EvidenceChunk):
                out.append(item)
                continue
            if isinstance(item, dict):
                try:
                    out.append(EvidenceChunk.model_validate(item))
                except Exception:
                    continue
        return out

    def _resolve_tool_budget_config(self) -> Dict[str, Any]:
        from config.application.deepsearch_config import ToolBudgetConfig

        raw = None
        try:
            raw = (getattr(self, "config", None) or {}).get("tool_budget")
        except Exception:
            raw = None
        payload: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        model = ToolBudgetConfig.model_validate(payload)
        return model.model_dump()

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
        """Plan → Graph reasoning → Report → Quality gate → Iterate."""

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
        plan_result_written = False
        stage_timings_written = False

        from core.deepsearch.tooling.budget import ToolBudget, reset_tool_budget, set_tool_budget

        budget_cfg = self._resolve_tool_budget_config()
        budget_token = None
        budget_obj: ToolBudget | None = None
        if budget_cfg.get("enabled"):
            budget_obj = ToolBudget(max_calls_total=int(budget_cfg["max_calls_total"]))
            budget_token = set_tool_budget(budget_obj)

        trace_capture_token = None
        captured_trace_events: list[dict[str, Any]] = []
        try:
            trace_capture_token, captured_trace_events = attach_trace_capture(sink=captured_trace_events)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to attach trace capture: %s", exc, exc_info=True)
            trace_capture_token = None

        try:
            bootstrap_context = graph_context or self._bootstrap_graph_context(
                question=normalized_question,
                scope=scope,
            )
            bootstrap_context = self._attach_run_metadata(
                bootstrap_context,
                run_id=state.run_id,
                metadata=metadata,
                artifact_dir=artifact_dir,
            )
            bootstrap_context = self._attach_file_scope_hints(
                bootstrap_context,
                question=normalized_question,
            )
            initial_think = await self._execute_stage(
                "initial_think",
                self._run_initial_think,
                state=state,
                stage_timings=stage_timings,
                question=normalized_question,
                scope=scope,
                reasoning_context=bootstrap_context,
                plan_steps=[],
            )
            report_needed = bool(initial_think.get("report_needed", True)) if isinstance(initial_think, dict) else True
            plan_state = initial_think.get("plan_state") if isinstance(initial_think, dict) else None
            if not report_needed:
                final_think = await self._execute_stage(
                    "final_think",
                    self._run_final_think,
                    state=state,
                    stage_timings=stage_timings,
                    question=normalized_question,
                    scope=scope,
                    reasoning_context=bootstrap_context,
                    evidences=[],
                    coverage_metrics={},
                    plan_items=getattr(plan_state, "items", None),
                    report_needed=False,
                    final_answer_mode="direct",
                )
                final_raw = final_think.get("raw") if isinstance(final_think, dict) else {}
                answer = ""
                if isinstance(final_raw, dict):
                    answer = str(final_raw.get("reasoning") or "")
                plan_result = {
                    "plan": {
                        "plan_id": None,
                        "question": normalized_question,
                        "mode": "initial_think",
                        "steps": [],
                    }
                }
                try:
                    state.record_plan(plan_result)
                except Exception:
                    pass
                runtime_plan = getattr(plan_state, "to_payload", lambda: {"items": [], "markdown": "", "version": 0})()
                reasoning_trace = {
                    "question": normalized_question,
                    "graph_context": bootstrap_context.model_dump(exclude_none=True),
                    "adapter_metadata": {},
                    "plan_steps": [],
                    "graph_traversals": [],
                    "reasoning_steps": [],
                    "evidences": [],
                    "tool_results": [],
                    "think_notes": (initial_think.get("think_notes") if isinstance(initial_think, dict) else [])
                    + (final_think.get("think_notes") if isinstance(final_think, dict) else []),
                    "runtime_plan": runtime_plan,
                    "final_think": final_raw,
                    "coverage_metrics": {
                        "evidence_count": 0,
                        "primary_evidence_count": 0,
                        "derived_evidence_count": 0,
                        "diagnostic_evidence_count": 0,
                        "total_evidence_count": 0,
                        "completed_steps": 0,
                        "total_steps": 0,
                        "coverage_ratio": 0.0,
                        "plan_progress_ratio": 0.0,
                        "expected_min_chunks": 0,
                        "coverage_score": 0.0,
                        "confidence_score": None,
                        "missing_topics": [],
                    },
                }
                state.record_reasoning(reasoning_trace)
                report = {
                    "question": normalized_question,
                    "answer": answer,
                    "evidences": [],
                    "metadata": {"report_generated": False, "source": "final_think"},
                }
                state.record_report(report)
                snapshot = state.snapshot()
                return {"plan": plan_result, "reasoning": reasoning_trace, "report": report, "state": snapshot}

            plan_result = {
                "plan": {
                    "plan_id": None,
                    "question": normalized_question,
                    "mode": "initial_think",
                    "steps": [],
                },
                "plan_items": list(getattr(plan_state, "items", []) or []),
            }
            if getattr(self, "artifact_store", None) is not None:
                try:
                    self.artifact_store.write_json(state.run_id, "plan_result.json", json_safe(plan_result))
                    plan_result_written = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist plan_result artifact: %s", exc, exc_info=True)
                    state.append_error(f"artifact_store.write_json(plan_result.json) failed: {exc}", stage="persist")
            state.record_plan(plan_result)

            reasoning_context = self._attach_run_metadata(
                bootstrap_context,
                run_id=state.run_id,
                metadata=metadata,
                artifact_dir=artifact_dir,
            )
            reasoning_context = self._attach_file_scope_hints(
                reasoning_context,
                question=normalized_question,
            )
            if budget_obj is not None and bool(budget_cfg.get("expose_to_llm")):
                reasoning_context.metadata.setdefault(
                    "tool_budget",
                    {
                        "max_calls_total": int(budget_cfg["max_calls_total"]),
                        "used_calls": 0,
                        "remaining_calls": int(budget_cfg["max_calls_total"]),
                    },
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

            if plan_state is not None:
                reasoning_context.metadata["runtime_plan"] = list(getattr(plan_state, "items", []) or [])

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
            final_report: Dict[str, Any] | None = None
            quality_history: List[Dict[str, Any]] = []

            followup_notes: List[Any] = []
            max_rounds = max(1, int(quality_cfg.max_rounds))
            for round_idx in range(1, max_rounds + 1):
                initial_notes = initial_think.get("think_notes_obj") if round_idx == 1 else followup_notes
                seed_evidences = None
                if round_idx > 1 and isinstance(cumulative_reasoning, dict):
                    seed_evidences = self._coerce_evidence_chunks(cumulative_reasoning.get("evidences") or [])
                round_trace = await self._execute_stage(
                    f"graph_reasoning_r{round_idx}",
                    self._think_loop_stage,
                    state=state,
                    stage_timings=stage_timings,
                    question=normalized_question,
                    reasoning_context=reasoning_context,
                    initial_think_notes=initial_notes,
                    seed_evidences=seed_evidences,
                    plan_items=list(getattr(plan_state, "items", []) or []),
                )
                if cumulative_reasoning is None:
                    cumulative_reasoning = round_trace
                    cumulative_reasoning["quality_loop"] = {"rounds": []}
                else:
                    cumulative_reasoning = self._merge_reasoning_traces(cumulative_reasoning, round_trace)
                cumulative_reasoning.setdefault("quality_loop", {}).setdefault("rounds", []).append(
                    {"round": round_idx, "plan_steps": []}
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

                final_think = await self._execute_stage(
                    f"final_think_r{round_idx}",
                    self._run_final_think,
                    state=state,
                    stage_timings=stage_timings,
                    question=normalized_question,
                    scope=scope,
                    reasoning_context=reasoning_context,
                    evidences=cumulative_reasoning.get("evidences") if isinstance(cumulative_reasoning, dict) else [],
                    coverage_metrics=cumulative_reasoning.get("coverage_metrics") if isinstance(cumulative_reasoning, dict) else {},
                    plan_items=(cumulative_reasoning.get("runtime_plan") or {}).get("items")
                    if isinstance(cumulative_reasoning, dict)
                    else None,
                    report_needed=True,
                    final_answer_mode="summary",
                )
                if isinstance(cumulative_reasoning, dict) and isinstance(final_think, dict):
                    cumulative_reasoning.setdefault("think_notes", []).extend(final_think.get("think_notes") or [])
                    cumulative_reasoning["final_think"] = final_think.get("raw") or {}

                report = await self._execute_stage(
                    f"report_r{round_idx}",
                    self._report_stage,
                    state=state,
                    stage_timings=stage_timings,
                    reasoning_trace=cumulative_reasoning,
                    external_logs=None,
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

                followup_notes = self._build_followup_think_notes(
                    actions=quality_result.get("actions") or [],
                    round_idx=round_idx,
                )
                if not followup_notes:
                    break
                await emit_trace(
                    "write_outline",
                    "\n".join(
                        [
                            f"Plan update from quality gate (round {round_idx + 1}).",
                            json.dumps(json_safe(quality_result.get("actions") or []), ensure_ascii=False, indent=2, default=str),
                        ]
                    ),
                    meta={"stage": "plan_update", "round": round_idx + 1, "actions": json_safe(quality_result.get("actions") or [])},
                )

            if final_report is None:
                raise RuntimeError("DeepSearch did not produce a report")
            report = final_report
            if cumulative_reasoning is not None:
                cumulative_reasoning.setdefault("quality_loop", {})["quality_gate_history"] = quality_history
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
                    stage_timings_written = True
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
                artifacts_present = {
                    "plan_result": plan_result_written,
                    "stage_timings": stage_timings_written,
                    "reasoning": False,
                    "report": False,
                    "report_md": False,
                }
                artifacts_cfg: Dict[str, Any] | None = None
                try:
                    artifacts_cfg = self._resolve_artifacts_config()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to resolve artifacts config: %s", exc, exc_info=True)
                    state.append_error(f"artifacts config invalid: {exc}", stage="persist")
                    artifacts_cfg = None

                refs_enabled = True
                dedupe_enabled = False
                evidence_pool_filename = ""
                artifacts_version = 0

                if isinstance(artifacts_cfg, dict):
                    try:
                        refs_enabled = bool((artifacts_cfg.get("refs") or {}).get("enabled", True))
                    except Exception:
                        refs_enabled = True

                    dedupe_cfg = artifacts_cfg.get("dedupe")
                    if not isinstance(dedupe_cfg, dict):
                        dedupe_cfg = {}
                    dedupe_enabled = bool(dedupe_cfg.get("enabled", True))
                    evidence_pool_filename = str(dedupe_cfg.get("evidence_pool_filename") or "evidence_pool.json").strip()
                    try:
                        artifacts_version = int(artifacts_cfg.get("version") or 2)
                    except Exception:
                        artifacts_version = 2

                reasoning_to_persist: Dict[str, Any] = json_safe(cumulative_reasoning or {})
                report_to_persist: Dict[str, Any] = json_safe(report or {})

                if artifacts_version >= 2 and dedupe_enabled and evidence_pool_filename:
                    try:
                        pool, reasoning_evidence_ids, report_evidence_ids = build_evidence_pool_v2(
                            reasoning=reasoning_to_persist,
                            report=report_to_persist,
                            artifact_version=artifacts_version,
                        )
                        self.artifact_store.write_json(state.run_id, evidence_pool_filename, json_safe(pool))
                        artifacts_present["evidence_pool"] = True

                        reasoning_to_persist = dedupe_reasoning_v2(
                            reasoning=reasoning_to_persist,
                            refs_enabled=refs_enabled,
                            evidence_pool_filename=evidence_pool_filename,
                            plan_filename="plan_result.json",
                            evidence_ids=reasoning_evidence_ids,
                        )
                        report_to_persist = dedupe_report_v2(
                            report=report_to_persist,
                            refs_enabled=refs_enabled,
                            report_markdown_filename="report.md",
                            evidence_pool_filename=evidence_pool_filename,
                            reasoning_filename="reasoning.json",
                            plan_filename="plan_result.json",
                            evidence_ids=report_evidence_ids,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to dedupe DeepSearch artifacts: %s", exc, exc_info=True)
                        state.append_error(f"artifact_store v2 dedupe failed: {exc}", stage="persist")

                try:
                    self.artifact_store.write_json(state.run_id, "reasoning.json", json_safe(reasoning_to_persist))
                    artifacts_present["reasoning"] = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist reasoning artifact: %s", exc, exc_info=True)
                    state.append_error(f"artifact_store.write_json(reasoning.json) failed: {exc}", stage="persist")

                try:
                    self.artifact_store.write_json(state.run_id, "report.json", json_safe(report_to_persist))
                    artifacts_present["report"] = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist report artifact: %s", exc, exc_info=True)
                    state.append_error(f"artifact_store.write_json(report.json) failed: {exc}", stage="persist")

                if isinstance(report, dict) and isinstance(report.get("answer"), str):
                    try:
                        self.artifact_store.write_text(state.run_id, "report.md", report.get("answer") or "")
                        artifacts_present["report_md"] = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to persist report.md artifact: %s", exc, exc_info=True)
                        state.append_error(f"artifact_store.write_text(report.md) failed: {exc}", stage="persist")

                if artifacts_cfg is None:
                    try:
                        self.artifact_store.write_json(state.run_id, "state_snapshot.json", json_safe(snapshot))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to persist fallback state_snapshot artifact: %s", exc, exc_info=True)
                        state.append_error(f"artifact_store.write_json(state_snapshot.json) failed: {exc}", stage="persist")
                else:
                    if artifacts_version >= 2:
                        try:
                            docs = build_v2_artifact_documents(
                                snapshot=snapshot,
                                stage_timings=stage_timings,
                                artifacts_config=artifacts_cfg,
                                artifacts_present=artifacts_present,
                                trace_events=captured_trace_events,
                            )
                            for filename, payload in docs.items():
                                self.artifact_store.write_json(state.run_id, filename, json_safe(payload))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to persist DeepSearch v2 artifacts: %s", exc, exc_info=True)
                            state.append_error(f"artifact_store v2 persist failed: {exc}", stage="persist")
                    else:
                        try:
                            self.artifact_store.write_json(state.run_id, "state_snapshot.json", json_safe(snapshot))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to persist legacy state_snapshot artifact: %s", exc, exc_info=True)
                            state.append_error(f"artifact_store.write_json(state_snapshot.json) failed: {exc}", stage="persist")
            logger.info(
                "DeepSearch run %s completed (owner=%s, timings=%s)",
                snapshot.get("run_id"),
                owner_id,
                stage_timings,
            )
            return {"plan": plan_result, "reasoning": cumulative_reasoning or {}, "report": report, "state": snapshot}
        finally:
            if trace_capture_token is not None:
                try:
                    reset_trace_emitter(trace_capture_token)
                except Exception:
                    pass
            if budget_token is not None:
                reset_tool_budget(budget_token)
