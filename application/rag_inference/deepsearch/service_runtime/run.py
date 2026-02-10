import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ThinkNote
from core.deepsearch.state import DeepSearchState
from core.deepsearch.trace import emit_trace, reset_trace_emitter
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from core.utils.json_safe import json_safe
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_GRAPH_INFERENCE
from config.core.deepsearch.runtime_messages import MISSING_PRIMARY_EVIDENCE_HARD_GATE_MESSAGE
from config.core.deepsearch.reasoning_defaults import (
    REPORT_HARD_GATE_MAX_REASONING_RETRIES,
    REPORT_HARD_GATE_MIN_PRIMARY_PAGE_EVIDENCE,
)

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
    def _count_read_pages_evidence(raw: Sequence[Dict[str, Any]] | None) -> int:
        count = 0
        for ev in raw or []:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("source") or "").strip() == "read.pages":
                count += 1
        return count

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

    def _resolve_navigation_bootstrap_config(self) -> Dict[str, Any]:
        """Service-level knobs for deterministic long-doc navigation bootstrap."""

        from config.core.deepsearch import tool_defaults

        raw = None
        try:
            raw = (getattr(self, "config", None) or {}).get("navigation_bootstrap")
        except Exception:
            raw = None
        cfg: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        return {
            "enabled": bool(cfg.get("enabled", tool_defaults.NAV_BOOTSTRAP_ENABLED)),
            "file_top_k": int(cfg.get("file_top_k", tool_defaults.NAV_BOOTSTRAP_FILE_TOP_K)),
            "candidate_files": int(cfg.get("candidate_files", tool_defaults.NAV_BOOTSTRAP_CANDIDATE_FILES)),
            "toc_max_depth": int(cfg.get("toc_max_depth", tool_defaults.NAV_BOOTSTRAP_TOC_MAX_DEPTH)),
            "toc_max_nodes": int(cfg.get("toc_max_nodes", tool_defaults.NAV_BOOTSTRAP_TOC_MAX_NODES)),
            "section_top_k": int(cfg.get("section_top_k", tool_defaults.NAV_BOOTSTRAP_SECTION_TOP_K)),
        }

    async def _invoke_tool_for_bootstrap(
        self,
        *,
        tool_name: str,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        extra: Dict[str, Any] | None,
        plan_step: str,
    ):
        """Invoke a DeepSearch graph tool deterministically (no LLM decision)."""

        tool_manager = getattr(self, "tool_manager", None)
        if tool_manager is None:
            raise RuntimeError("DeepSearchService requires a tool_manager for navigation bootstrap")

        payload = {
            "question": question,
            "plan_step": plan_step,
            "context_evidences": [],
            "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
            "access_scope": scope,
            "extra": dict(extra or {}),
            "graph_context": reasoning_context.model_dump(exclude_none=True),
            "coverage_metrics": {},
        }
        invocation = tool_manager.invoke(tool_name, payload=payload)
        return await invocation

    async def _run_navigation_bootstrap(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
    ) -> List[EvidenceChunk]:
        """Run a minimal navigation backbone and expose it as diagnostic evidence.

        Why: models often skip `toc.tree` / `section.select` / `read.pages` unless they are visible and
        immediately available. This stage makes long-doc navigation the default path while staying
        observable and configurable.
        """

        cfg = self._resolve_navigation_bootstrap_config()
        if not cfg.get("enabled"):
            await emit_trace(
                "think",
                "Navigation bootstrap skipped (disabled).",
                meta={"stage": "nav_bootstrap", "reason": "disabled"},
            )
            return []

        # 1) relevant files (global routing)
        try:
            search_file = await self._invoke_tool_for_bootstrap(
                tool_name="search.file",
                question=question,
                scope=scope,
                reasoning_context=reasoning_context,
                extra={"top_k": max(0, int(cfg["file_top_k"]))},
                plan_step="nav_bootstrap.search_file",
            )
        except Exception as exc:  # noqa: BLE001
            await emit_trace(
                "think",
                f"Navigation bootstrap failed at search.file: {exc}",
                meta={"stage": "nav_bootstrap", "reason": "search_file_error"},
            )
            return []

        diagnostics = getattr(search_file, "diagnostics", None) if search_file is not None else None
        results = []
        if isinstance(diagnostics, dict):
            results = diagnostics.get("results") if isinstance(diagnostics.get("results"), list) else []

        candidate_files: List[Dict[str, Any]] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            file_id = str(row.get("file_id") or "").strip()
            if not file_id:
                continue
            candidate_files.append(
                {
                    "file_id": file_id,
                    "filename": row.get("filename"),
                    "score": row.get("score"),
                    "hit_count": row.get("hit_count"),
                    "per_channel_hits": row.get("per_channel_hits"),
                }
            )

        max_files = max(0, int(cfg["candidate_files"]))
        selected_files = candidate_files[:max_files] if max_files else []

        toc_summaries: Dict[str, str] = {}
        section_results: Dict[str, Any] = {}
        for row in selected_files:
            file_id = str(row.get("file_id") or "").strip()
            if not file_id:
                continue

            # 2) Within-file navigation: toc.tree + section.select.
            #
            # Note: toc.tree / read.pages are implemented as explore sub-tools and
            # are not registered as top-level tools in the tool manager. Invoke via explore to
            # keep bootstrap compatible with local-only (no MCP) deployments.
            try:
                explore = await self._invoke_tool_for_bootstrap(
                    tool_name="explore",
                    question=question,
                    scope=scope,
                    reasoning_context=reasoning_context,
                    extra={
                        "max_concurrency": 2,
                        "actions": [
                            {
                                "id": "toc",
                                "tool": "toc.tree",
                                "args": {
                                    "file_id": file_id,
                                    "max_depth": max(1, int(cfg["toc_max_depth"])),
                                    "max_nodes": max(1, int(cfg["toc_max_nodes"])),
                                },
                            },
                            {
                                "id": "sec",
                                "tool": "section.select",
                                "args": {"file_id": file_id, "top_k": max(1, int(cfg["section_top_k"]))},
                            },
                        ],
                    },
                    plan_step=f"nav_bootstrap.explore_nav.{file_id}",
                )
            except Exception as exc:  # noqa: BLE001
                toc_summaries[file_id] = f"toc.tree failed: {exc}"
                section_results[file_id] = {"error": str(exc)}
                continue

            explore_diag = getattr(explore, "diagnostics", None)
            actions = explore_diag.get("actions") if isinstance(explore_diag, dict) else None
            toc_action = None
            sec_action = None
            if isinstance(actions, list):
                for item in actions:
                    if not isinstance(item, dict):
                        continue
                    if item.get("id") == "toc" or item.get("tool") == "toc.tree":
                        toc_action = item
                    if item.get("id") == "sec" or item.get("tool") == "section.select":
                        sec_action = item

            if isinstance(toc_action, dict) and toc_action.get("status") == "ok":
                toc_summaries[file_id] = str(toc_action.get("summary") or "")
            elif isinstance(toc_action, dict):
                toc_summaries[file_id] = str(toc_action.get("error") or toc_action.get("summary") or "").strip() or "(toc.tree failed)"
            else:
                toc_summaries[file_id] = "(toc.tree missing)"

            if isinstance(sec_action, dict) and sec_action.get("status") == "ok":
                sec_diag = sec_action.get("diagnostics")
                section_results[file_id] = sec_diag if isinstance(sec_diag, dict) else {}
            elif isinstance(sec_action, dict):
                section_results[file_id] = {"error": str(sec_action.get("error") or sec_action.get("summary") or "").strip()}
            else:
                section_results[file_id] = {"error": "section.select missing"}

        # Produce a single diagnostic evidence chunk that the think tool can reference.
        lines: List[str] = []
        lines.append("Navigation bootstrap (deterministic).")
        lines.append("")
        lines.append("search.file candidates:")
        if not candidate_files:
            lines.append("- (none)")
        else:
            for idx, row in enumerate(candidate_files[: max(1, int(cfg["file_top_k"]))], start=1):
                lines.append(
                    f"- {idx}. file_id={row.get('file_id')} score={row.get('score')} hits={row.get('hit_count')} filename={row.get('filename') or ''}"
                )
        for row in selected_files:
            file_id = str(row.get("file_id") or "").strip()
            if not file_id:
                continue
            lines.append("")
            lines.append(f"[file_id={file_id}] toc.tree:")
            toc_text = str(toc_summaries.get(file_id) or "").strip()
            if toc_text:
                lines.append(toc_text)
            else:
                lines.append("(empty)")
            lines.append("")
            lines.append(f"[file_id={file_id}] section.select:")
            sec_diag = section_results.get(file_id)
            cand_rows = sec_diag.get("candidates") if isinstance(sec_diag, dict) else None
            if isinstance(cand_rows, list) and cand_rows:
                for idx, c in enumerate(cand_rows[: max(1, int(cfg["section_top_k"]))], start=1):
                    if not isinstance(c, dict):
                        continue
                    lines.append(
                        f"- {idx}. section_id={c.get('section_id')} score={c.get('score')} pages={c.get('page_start')}-{c.get('page_end')} "
                        f"path={c.get('path') or c.get('title') or ''} source={c.get('source') or ''}"
                    )
            elif isinstance(sec_diag, dict) and isinstance(sec_diag.get("selection"), dict):
                selection = sec_diag.get("selection") or {}
                lines.append(f"- primary={selection.get('primary_section_ids') or []}")
                lines.append(f"- supplementary={selection.get('supplementary_section_ids') or []}")
            elif isinstance(sec_diag, dict) and sec_diag.get("error"):
                lines.append(f"- (error) {sec_diag.get('error')}")
            else:
                lines.append("- (none)")

        # NOTE: No truncation. This is diagnostic-only and stored as an evidence chunk for LLM visibility.
        content = "\n".join(lines)
        evidence = EvidenceChunk(
            kind="diagnostic",
            chunk_id=derived_chunk_id(tool_name="nav.bootstrap", plan_step="nav_bootstrap", content=content, label="nav"),
            source="nav.bootstrap",
            content=content,
            score=None,
            provenance={
                "stage": "nav_bootstrap",
                "candidate_files": candidate_files,
                "selected_files": selected_files,
                "section_results": section_results,
            },
        )
        await emit_trace(
            "think",
            content,
            meta={"stage": "nav_bootstrap", "selected_file_count": len(selected_files)},
        )
        return [evidence]

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
            if not report_needed:
                # Direct-answer mode (no report) still needs to persist artifacts and traces so that
                # evaluation/debugging remains observable and reproducible.
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
                    },
                    "plan_items": list(getattr(plan_state, "items", []) or []) if plan_state is not None else [],
                }
                try:
                    if getattr(self, "artifact_store", None) is not None:
                        self.artifact_store.write_json(state.run_id, "plan_result.json", json_safe(plan_result))
                        plan_result_written = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist plan_result artifact: %s", exc, exc_info=True)
                    state.append_error(f"artifact_store.write_json(plan_result.json) failed: {exc}", stage="persist")
                try:
                    state.record_plan(plan_result)
                except Exception:
                    pass

                runtime_plan = getattr(plan_state, "to_payload", lambda: {"items": [], "markdown": "", "version": 0})()
                round_trace = {
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
                state.record_reasoning(round_trace)
                report = {
                    "question": normalized_question,
                    "answer": answer,
                    "evidences": [],
                    "metadata": {"report_generated": False, "source": "final_think"},
                }
                state.record_report(report)
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
                    reasoning_trace=round_trace,
                    report=report,
                    snapshot=snapshot,
                    stage_timings=stage_timings,
                    captured_trace_events=captured_trace_events,
                    plan_result_written=plan_result_written,
                    stage_timings_written=stage_timings_written,
                )
            bootstrap_evidences = await self._run_navigation_bootstrap(
                question=normalized_question,
                scope=scope,
                reasoning_context=bootstrap_context,
            )
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
            # Let downstream orchestration (think loop / report gate) know whether this run is expected
            # to produce a report-style answer.
            if isinstance(reasoning_context.metadata, dict):
                reasoning_context.metadata.setdefault("report_style", initial_think.get("report_style") if isinstance(initial_think, dict) else None)
                reasoning_context.metadata["report_needed"] = True
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
            # ------------------------------------------------------------------
            # Report hard gate (read.pages evidence required)
            #
            # Important: do NOT surface this gate as a user-visible "error report" by default.
            # Instead, feed the failure back into the next reasoning attempt so the LLM can
            # retry tool calls (e.g., routing -> section.select -> read.pages).
            # ------------------------------------------------------------------
            gate_cfg = {}
            try:
                gate_cfg = (getattr(self, "config", None) or {}).get("report_gate") or {}
            except Exception:
                gate_cfg = {}
            try:
                required_pages = int(gate_cfg.get("min_primary_page_evidence", REPORT_HARD_GATE_MIN_PRIMARY_PAGE_EVIDENCE))
            except Exception:
                required_pages = int(REPORT_HARD_GATE_MIN_PRIMARY_PAGE_EVIDENCE)
            try:
                max_gate_retries = int(gate_cfg.get("max_reasoning_retries", REPORT_HARD_GATE_MAX_REASONING_RETRIES))
            except Exception:
                max_gate_retries = int(REPORT_HARD_GATE_MAX_REASONING_RETRIES)
            required_pages = max(0, int(required_pages))
            max_gate_retries = max(0, int(max_gate_retries))

            # Carry-over for retries (so a second attempt can continue instead of forgetting the run).
            carried_initial_notes = list(initial_think.get("think_notes_obj") or [])
            carried_seed_evidences = list(bootstrap_evidences or [])

            round_trace: Dict[str, Any] | None = None
            report_evidences: List[Dict[str, Any]] = []
            page_evidence_count = 0
            attempt_idx = 0
            gate_note_added = 0
            while True:
                # If we have no remaining tool budget, we cannot keep the loop alive to satisfy the gate.
                # In that case we will stop gracefully after the loop with an actionable (but rare) message.
                try:
                    if required_pages > 0 and budget_obj is not None:
                        remaining = int(budget_obj.snapshot().get("remaining_calls") or 0)
                        if remaining <= 0:
                            break
                except Exception:
                    pass
                round_trace = await self._execute_stage(
                    "graph_reasoning",
                    self._think_loop_stage,
                    state=state,
                    stage_timings=stage_timings,
                    question=normalized_question,
                    reasoning_context=reasoning_context,
                    initial_think_notes=carried_initial_notes,
                    seed_evidences=carried_seed_evidences,
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
                try:
                    evidence_count = len(round_trace.get("evidences") or [])
                except Exception:
                    evidence_count = 0
                await emit_trace(
                    "progress",
                    "\n".join(
                        [
                            "Completed graph reasoning.",
                            f"evidence_count={evidence_count}",
                            f"attempt={attempt_idx + 1}",
                        ]
                    ),
                    meta={"stage": "graph_reasoning", "evidence_count": evidence_count, "attempt": attempt_idx + 1},
                )

                all_evidences = round_trace.get("evidences") if isinstance(round_trace, dict) else []
                page_evidence_count = self._count_read_pages_evidence(all_evidences if isinstance(all_evidences, list) else [])
                report_evidences = self._filter_evidences_for_report(all_evidences if isinstance(all_evidences, list) else [])
                if isinstance(round_trace, dict):
                    round_trace.setdefault("report_gate", {})
                    round_trace["report_gate"] = {
                        "required_primary_source": "read.pages",
                        "min_primary_page_evidence_required": required_pages,
                        "primary_page_evidence_count": page_evidence_count,
                        "report_evidence_count": len(report_evidences),
                        "total_evidence_count": len(all_evidences) if isinstance(all_evidences, list) else 0,
                        "attempt": attempt_idx + 1,
                        "soft_max_attempts": max_gate_retries + 1,
                    }

                # If no hard gate is required, proceed immediately.
                if required_pages <= 0 or page_evidence_count >= required_pages:
                    break

                # Give the LLM a clear, LLM-visible failure note and retry.
                #
                # IMPORTANT: do NOT stop the run just because the gate is unmet after a small number
                # of retries. In practice, models may need multiple rounds to route → select section
                # → read.pages, especially under flaky networks / JSON repair churn.
                #
                # We treat `max_gate_retries` as a *soft* cap on how many times we append the same
                # gate note (to avoid bloating think_notes), not as a reason to finalize a user-visible
                # "missing evidence" report.
                carried_seed_evidences = self._coerce_evidence_chunks(all_evidences if isinstance(all_evidences, list) else [])
                if gate_note_added < max_gate_retries + 1:
                    carried_initial_notes.append(
                        ThinkNote(
                            plan_step_id=None,
                            reasoning=(
                                "Report hard gate not satisfied: no citeable page evidence was collected yet. "
                                "Search snippets are navigation-only. You must call read.pages at least once on relevant pages "
                                "before stopping the reasoning loop."
                            ),
                            confidence_delta=None,
                            coverage_delta=None,
                            next_actions=[
                                "Use explore + search.file to pick a real file_id (UUID) if not already known.",
                                "Use toc.tree / tree.root / section.select to locate likely sections and page ranges.",
                                "Call read.pages on those pages (full-page evidence), then continue reasoning.",
                            ],
                            metadata={
                                "gate": "missing_primary_page_evidence",
                                "required_primary_source": "read.pages",
                                "attempt": attempt_idx + 1,
                                "soft_max_attempts": max_gate_retries + 1,
                            },
                        )
                    )
                    gate_note_added += 1
                await emit_trace(
                    "think",
                    "Evidence gate unmet (missing read.pages). Retrying reasoning with an explicit gate note.",
                    meta={"stage": "report_gate_retry", "attempt": attempt_idx + 1, "soft_max_attempts": max_gate_retries + 1},
                )
                attempt_idx += 1

            # If we still cannot collect citeable page evidence, stop gracefully with an actionable message
            # (rather than fabricating an evidence-grounded report).
            #
            # This should be rare in normal operation because the loop above keeps retrying until the tool budget
            # is exhausted. When it happens, it's usually due to:
            # - tool budget exhaustion
            # - persistent tool failures (invalid_file_id, repeated JSON parsing failures, network issues, etc.)
            if required_pages > 0 and page_evidence_count < required_pages:
                report = {
                    "question": normalized_question,
                    "answer": MISSING_PRIMARY_EVIDENCE_HARD_GATE_MESSAGE,
                    "evidences": [],
                    "metadata": {
                        "report_generated": False,
                        "reason": "missing_primary_page_evidence",
                        "required_primary_source": "read.pages",
                        "min_primary_page_evidence_required": required_pages,
                        "primary_page_evidence_count": page_evidence_count,
                        "soft_max_reasoning_retries": max_gate_retries,
                        "total_attempts": attempt_idx + 1,
                    },
                }
                state.record_report(report)
                await emit_trace(
                    "progress",
                    "Report blocked by evidence gate after retries: missing read.pages evidence.",
                    meta={
                        "stage": "report_gate",
                        "reason": "missing_primary_page_evidence",
                        "attempts": attempt_idx + 1,
                    },
                )
                state.reasoning_trace = round_trace or {}
                try:
                    state.transition_stage("done", metadata={"rounds": 1, "report_blocked": True})
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

            gated_report = self._maybe_hard_gate_computable_question(
                question=normalized_question,
                reasoning_trace=round_trace,
                classification=classification,
                routing_cfg=routing_cfg,
            )
            if gated_report is not None:
                report = gated_report
                state.record_report(report)
            else:
                report_trace = dict(round_trace) if isinstance(round_trace, dict) else {}
                report_trace["evidences"] = list(report_evidences)
                report = await self._execute_stage(
                    "report",
                    self._report_stage,
                    state=state,
                    stage_timings=stage_timings,
                    reasoning_trace=report_trace,
                    external_logs=None,
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
