import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.deepsearch.state import DeepSearchState
from core.deepsearch.trace import emit_trace, reset_trace_emitter
from core.graph_adapter.base import GraphAccessScope
from core.utils.json_safe import json_safe
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_GRAPH_INFERENCE

from .artifact_dedupe_v2 import build_evidence_pool_v2, dedupe_reasoning_v2, dedupe_report_v2
from .artifact_views_v2 import build_v2_artifact_documents
from .trace_capture import attach_trace_capture
from .llm_dump_metrics import summarize_llm_dump_metrics

logger = logging.getLogger(__name__)


class DeepSearchServiceRunMixin:
    @staticmethod
    def _maybe_record_llm_dump_metrics(state: DeepSearchState) -> None:
        """Best-effort: attach aggregated LLM dump metrics to the state snapshot.

        This reads DEEPSEARCH_LLM_DUMP_PATH JSONL only when the env var is set.
        It must never raise or affect the main pipeline.
        """

        try:
            dump_path = str(os.getenv("DEEPSEARCH_LLM_DUMP_PATH", "") or "").strip()
        except Exception:
            dump_path = ""
        if not dump_path:
            return
        try:
            run_id = str(getattr(state, "run_id", "") or "").strip()
            metrics = summarize_llm_dump_metrics(path=dump_path, run_id=run_id)
            state.record_cost("llm_dump_metrics", json_safe(metrics))
        except Exception:
            return

    @staticmethod
    def _filter_evidences_for_report(raw: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        """Only keep citeable evidence for report writing (page + optional graph inference)."""

        items = list(raw or [])
        kept: List[Dict[str, Any]] = []
        for ev in items:
            if not isinstance(ev, dict):
                continue
            source = str(ev.get("source") or "").strip()
            if source == "read.pages":
                kept.append(ev)
                continue
            prov = ev.get("provenance") if isinstance(ev.get("provenance"), dict) else None
            if isinstance(prov, dict) and prov.get("evidence_class") == EVIDENCE_CLASS_GRAPH_INFERENCE:
                kept.append(ev)
        return kept

    @staticmethod
    def _build_structured_report_stub(text: str) -> Dict[str, Any]:
        """Build a minimal structured_report for schema consistency (Path A)."""
        return {
            "format_version": "2.0",
            "text": text,
            "citations": [],
            "evidence_index": [],
            "source_key_map": {},
        }

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

    def _finalize_and_persist_run(
        self,
        *,
        owner_id: Optional[str],
        state: DeepSearchState,
        plan_result: Dict[str, Any],
        reasoning_trace: Dict[str, Any],
        report: Dict[str, Any],
        snapshot: Dict[str, Any],
        stage_timings: Dict[str, Any],
        captured_trace_events: list[dict[str, Any]],
        plan_result_written: bool,
        stage_timings_written: bool,
    ) -> Dict[str, Any]:
        """Finalize state and persist artifacts (shared by report/no-report flows)."""

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
            question=str(reasoning_trace.get("question") or ""),
            plan=plan_result,
            reasoning=reasoning_trace or {},
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

            reasoning_to_persist: Dict[str, Any] = json_safe(reasoning_trace or {})
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
                    logger.warning("Failed to build evidence pool: %s", exc, exc_info=True)
                    state.append_error(f"build evidence pool failed: {exc}", stage="persist")

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
        return {"plan": plan_result, "reasoning": reasoning_trace or {}, "report": report, "state": snapshot}

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

        from framework.run_context import reset_run_id, set_run_id

        run_token = set_run_id(state.run_id)
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
                context_evidences=[],
            )
            report_needed = bool(initial_think.get("report_needed", True)) if isinstance(initial_think, dict) else True
            plan_state = initial_think.get("plan_state") if isinstance(initial_think, dict) else None
            # --- Build reasoning context (unified, no report_needed branching) ---
            report_style = (initial_think.get("report_style") if isinstance(initial_think, dict) else None) or "deepsearch"

            plan_result = {
                "plan": {
                    "plan_id": None,
                    "question": normalized_question,
                    "mode": "initial_think",
                    "steps": [],
                },
                "plan_items": list(getattr(plan_state, "items", []) or []) if plan_state is not None else [],
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
            if isinstance(reasoning_context.metadata, dict):
                reasoning_context.metadata.setdefault("report_style", report_style)
                reasoning_context.metadata["report_needed"] = report_needed
            if budget_obj is not None and bool(budget_cfg.get("expose_to_llm")):
                reasoning_context.metadata.setdefault(
                    "tool_budget",
                    {
                        "max_calls_total": int(budget_cfg["max_calls_total"]),
                        "used_calls": 0,
                        "remaining_calls": int(budget_cfg["max_calls_total"]),
                    },
                )

            if plan_state is not None:
                reasoning_context.metadata["runtime_plan"] = list(getattr(plan_state, "items", []) or [])

            # --- Single think loop call (no nav bootstrap, no outer retry) ---
            round_trace = await self._execute_stage(
                "graph_reasoning",
                self._think_loop_stage,
                state=state,
                stage_timings=stage_timings,
                question=normalized_question,
                reasoning_context=reasoning_context,
                initial_think_notes=list(initial_think.get("think_notes_obj") or []) if isinstance(initial_think, dict) else [],
                seed_evidences=[],
                plan_items=list(getattr(plan_state, "items", []) or []),
            )
            state.record_reasoning(round_trace)
            try:
                from application.rag_inference.deepsearch.service_runtime.reasoning_metrics import (
                    summarize_reasoning_trace,
                )

                state.record_cost("reasoning_metrics", summarize_reasoning_trace(round_trace or {}))
            except Exception as exc:  # noqa: BLE001
                state.append_error(f"reasoning_metrics failed: {exc}", stage="graph_reasoning")
            self._surface_worker_failures(state, round_trace)

            # --- Post-loop: branch based on report_needed ---
            all_evidences = round_trace.get("evidences") if isinstance(round_trace, dict) else []
            report_evidences = self._filter_evidences_for_report(all_evidences if isinstance(all_evidences, list) else [])
            evidence_bank = round_trace.get("evidence_bank") if isinstance(round_trace, dict) else None

            # Path A: direct answer (report_needed=false → no gates, no report)
            if not report_needed:
                answer = self._extract_answer_from_trace(round_trace if isinstance(round_trace, dict) else {})
                report = {
                    "question": normalized_question,
                    "answer": answer,
                    "evidences": [],
                    "metadata": {"report_generated": False, "source": "think_loop_direct"},
                    "structured_report": self._build_structured_report_stub(answer),
                }
                state.record_report(report)
                state.reasoning_trace = round_trace
                try:
                    state.transition_stage("done", metadata={"rounds": 1})
                except Exception:
                    pass
                self._maybe_record_llm_dump_metrics(state)
                snapshot = state.snapshot()
                return self._finalize_and_persist_run(
                    owner_id=owner_id,
                    state=state,
                    plan_result=plan_result,
                    reasoning_trace=round_trace or {},
                    report=report,
                    snapshot=snapshot,
                    stage_timings=stage_timings,
                    captured_trace_events=captured_trace_events,
                    plan_result_written=plan_result_written,
                    stage_timings_written=stage_timings_written,
                )

            # Path B: report stage (evidence collected)
            report = await self._execute_stage(
                "report",
                self._report_stage,
                state=state,
                stage_timings=stage_timings,
                reasoning_trace=round_trace,
                report_evidences=report_evidences,
                evidence_bank=evidence_bank,
            )
            state.record_report(report)
            await emit_trace(
                "progress",
                "\n".join(
                    [
                        "Draft report generated.",
                        f"answer_length={len((report.get('answer') or '') if isinstance(report, dict) else '')}",
                        f"evidence_count={len((report.get('evidences') or []) if isinstance(report, dict) else [])}",
                    ]
                ),
                meta={"stage": "report"},
            )

            state.reasoning_trace = round_trace

            try:
                state.transition_stage(
                    "done",
                    metadata={
                        "rounds": 1,
                    },
                )
            except Exception:
                pass

            self._maybe_record_llm_dump_metrics(state)
            snapshot = state.snapshot()
            return self._finalize_and_persist_run(
                owner_id=owner_id,
                state=state,
                plan_result=plan_result,
                reasoning_trace=round_trace or {},
                report=(report if isinstance(report, dict) else {}),
                snapshot=snapshot,
                stage_timings=stage_timings,
                captured_trace_events=captured_trace_events,
                plan_result_written=plan_result_written,
                stage_timings_written=stage_timings_written,
            )
        finally:
            try:
                reset_run_id(run_token)
            except Exception:
                pass
            if trace_capture_token is not None:
                try:
                    reset_trace_emitter(trace_capture_token)
                except Exception:
                    pass
            if budget_token is not None:
                reset_tool_budget(budget_token)

    @staticmethod
    def _surface_worker_failures(state: DeepSearchState, reasoning_trace: Dict[str, Any]) -> None:
        if not isinstance(reasoning_trace, dict):
            return
        coverage = reasoning_trace.get("coverage_metrics") or {}
        if not isinstance(coverage, dict):
            return
        errors = coverage.get("worker_errors") or []
        if isinstance(errors, list) and errors:
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

        probe_errors = coverage.get("probe_errors") or []
        if not isinstance(probe_errors, list) or not probe_errors:
            return
        previews = []
        for entry in probe_errors:
            if len(previews) >= 3:
                break
            if not isinstance(entry, dict):
                continue
            tool = entry.get("tool_name") or "probe"
            code = entry.get("error") or entry.get("error_type") or "error"
            previews.append(f"{tool}={code}")
        preview = ", ".join(previews)
        message = f"{len(probe_errors)} probe tool(s) failed"
        if preview:
            message = f"{message}: {preview}"
        state.append_error(message, stage="graph_reasoning", details={"probe_errors": probe_errors})

    @staticmethod
    def _extract_answer_from_trace(round_trace: Dict[str, Any]) -> str:
        """Extract reasoning from the last think note as direct answer."""
        think_notes = round_trace.get("think_notes") or []
        for note in reversed(think_notes):
            if not isinstance(note, dict):
                continue
            raw = (note.get("metadata") or {}).get("raw")
            if isinstance(raw, dict):
                answer = str(raw.get("reasoning") or "").strip()
                if answer:
                    return answer
        return ""
