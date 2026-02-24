import asyncio
from typing import Any, Dict, List, Sequence

from core.deepsearch.trace import emit_trace
from core.deepsearch.memory.plan_state import PlanState, update_plan_from_think_notes
from core.deepsearch.planning import (
    build_template_fingerprint,
    coerce_templates,
    instantiate_template_plan,
    select_plan_template,
)
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext
from encapsulation.data_model.deepsearch import ThinkNote

from application.rag_inference.deepsearch.runtime_cache import CachedInitialThink
from config.core.deepsearch import plan_template_defaults


class DeepSearchServiceInitialThinkMixin:
    @staticmethod
    def _normalize_question_kind(value: Any) -> str:
        token = str(value or "").strip().lower() or "file_qa"
        if token not in {"encyclopedia", "file_qa", "file_computable_qa", "exploratory_report"}:
            return "file_qa"
        return token

    @staticmethod
    def _coerce_decomposition(value: Any, *, question: str) -> Dict[str, Any]:
        """Coerce a 5W1H decomposition object into a stable shape (best-effort).

        The decomposition is a retrieval-oriented plan scaffold (NOT evidence).
        If the selector fails to provide it, return a minimal placeholder so the
        think loop can still see that decomposition is expected.
        """

        q = str(question or "").strip()
        default = {
            "fivew1h": {"who": "", "what": "", "when": "", "where": "", "why": "", "how": ""},
            "sub_queries": [f"The document contains the information needed to answer: {q} (<TO_EXTRACT>)."] if q else [],
            "answer_expectation": "",
        }
        if not isinstance(value, dict):
            return default
        out: Dict[str, Any] = {}
        five = value.get("fivew1h")
        if isinstance(five, dict):
            out["fivew1h"] = {
                "who": str(five.get("who") or ""),
                "what": str(five.get("what") or ""),
                "when": str(five.get("when") or ""),
                "where": str(five.get("where") or ""),
                "why": str(five.get("why") or ""),
                "how": str(five.get("how") or ""),
            }
        else:
            out["fivew1h"] = dict(default["fivew1h"])
        subs_raw = value.get("sub_queries")
        subs: List[str] = []
        if isinstance(subs_raw, list):
            for item in subs_raw:
                s = str(item or "").strip()
                if s and s not in subs:
                    subs.append(s)
        out["sub_queries"] = subs[:10] if subs else list(default["sub_queries"])
        out["answer_expectation"] = str(value.get("answer_expectation") or "")
        return out

    @staticmethod
    def _attach_initial_think_signals(
        reasoning_context: GraphQueryContext,
        *,
        is_computable: bool,
        question_kind: str,
        decomposition: Dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(reasoning_context.metadata, dict):
            return
        # Make key signals explicit and stable so downstream think loops do not miss them.
        initial_think = {
            "is_computable": bool(is_computable),
            "question_kind": str(question_kind),
            # PageIndex/MinerU use 0-based page indices; human-facing page numbers are typically 1-based.
            "page_indexing": {"tool_index_base": 0, "human_index_base": 1},
        }
        if isinstance(decomposition, dict) and decomposition:
            # Keep the decomposition in initial_think so the think loop can drive retrieval sub-queries explicitly.
            initial_think["decomposition"] = dict(decomposition)
            # Convenience alias (avoid deep nesting in prompts/tools).
            reasoning_context.metadata["decomposition"] = dict(decomposition)
        reasoning_context.metadata["initial_think"] = initial_think
        # Convenience aliases (avoid deep nesting in prompts/tools).
        reasoning_context.metadata["is_computable"] = bool(is_computable)
        reasoning_context.metadata["question_kind"] = str(question_kind)
    def _initial_think_template_cfg(self) -> Dict[str, Any]:
        cfg = None
        try:
            cfg = (getattr(self, "config", None) or {}).get("initial_think_template")
        except Exception:
            cfg = None
        return cfg if isinstance(cfg, dict) else {}

    def _resolve_llm_connector(self) -> Any | None:
        tool_manager = getattr(self, "tool_manager", None)
        if tool_manager is None:
            return None
        try:
            tool_cfgs = getattr(tool_manager, "tool_configs", None)
            if isinstance(tool_cfgs, dict):
                return tool_cfgs.get("llm_connector")
        except Exception:
            return None
        return None

    def _resolve_tool_timeout_seconds(self) -> float | None:
        cfg = None
        try:
            cfg = (getattr(self, "config", None) or {}).get("graph_reasoning")
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            raw = cfg.get("tool_timeout_seconds")
        else:
            raw = None
        if raw is None:
            return None
        value = float(raw)
        if value < 0:
            raise ValueError("graph_reasoning.tool_timeout_seconds must be >= 0")
        return value

    def _think_tool_name(self) -> str:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config must be a dict")
        tool_name = str(self.config.get("think_tool") or "").strip()
        if not tool_name:
            raise ValueError("DeepSearchService config.think_tool is required")
        return tool_name

    async def _run_initial_think(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        context_evidences: Sequence[Dict[str, Any]] | None = None,
        plan_steps: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        tool_manager = getattr(self, "tool_manager", None)
        if not tool_manager:
            raise RuntimeError("DeepSearchService requires a tool_manager for initial_think")

        steps = list(plan_steps or [])
        total_steps = len(steps)
        evidences = list(context_evidences or [])

        # ------------------------------------------------------------------
        # Initial-think cache (conservative): only cache "pure" initial think calls,
        # i.e., no injected evidences/plan steps. This avoids accidental dependence on
        # ephemeral context and keeps cache correctness high.
        # ------------------------------------------------------------------
        cache = getattr(self, "_initial_think_cache", None)
        cache_key = None
        if cache is not None and not steps and not evidences:
            try:
                owner_scope_id = str(getattr(scope, "scope_id", "") or "")
                service_fp = str((getattr(self, "config", None) or {}).get("fingerprint") or "")
                llm_fp = str(getattr(tool_manager, "llm_fingerprint", "") or "")
                prompt_fp = str(getattr(self, "_think_prompt_fingerprint", "") or "")
                if owner_scope_id and service_fp and prompt_fp:
                    cache_key = cache.make_key(
                        owner_scope_id=owner_scope_id,
                        question=question,
                        service_fingerprint=service_fp,
                        llm_fingerprint=llm_fp,
                        think_prompt_fingerprint=prompt_fp,
                    )
                    hit = cache.get(cache_key)
                else:
                    hit = None
            except Exception:
                hit = None

            if hit is not None:
                plan_state = PlanState()
                plan_state.update(hit.plan_items)
                note_objs: list[ThinkNote] = []
                for payload in hit.think_notes_payloads:
                    try:
                        note_objs.append(ThinkNote.model_validate(payload))
                    except Exception:
                        continue

                report_style = str(hit.report_style or "").strip().lower()
                if report_style not in {"deepsearch", "research"}:
                    report_style = "deepsearch"
                is_computable = bool(getattr(hit, "is_computable", False))
                question_kind = self._normalize_question_kind(getattr(hit, "question_kind", "file_qa"))
                decomposition = getattr(hit, "decomposition", None)
                decomposition = self._coerce_decomposition(decomposition, question=question)
                if isinstance(reasoning_context.metadata, dict):
                    reasoning_context.metadata["report_style"] = report_style
                    reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
                    self._attach_initial_think_signals(
                        reasoning_context,
                        is_computable=is_computable,
                        question_kind=question_kind,
                        decomposition=decomposition,
                    )

                await emit_trace(
                    "write_outline",
                    plan_state.markdown,
                    meta={
                        "stage": "think_init",
                        "plan_step_id": "think_init",
                        "plan_version": plan_state.version,
                        "plan_items": list(plan_state.items),
                        "cache": {"hit": True},
                    },
                )

                await emit_trace(
                    "think",
                    "Initial think checkpoint (cache hit).",
                    meta={"stage": "think_init", "plan_step": "think_init", "cache": {"hit": True}},
                )
                note_payloads = list(hit.think_notes_payloads)
                return {
                    "report_needed": bool(hit.report_needed),
                    "report_style": report_style,
                    "is_computable": is_computable,
                    "question_kind": question_kind,
                    "plan_state": plan_state,
                    "think_notes": note_payloads,
                    "think_notes_obj": note_objs,
                    "raw": dict(hit.raw),
                    "cache": {"hit": True},
                }

        # ------------------------------------------------------------------
        # Initial-think plan templates (fast path): use a light LLM to pick a
        # template + fill slots, then seed the think loop with a synthetic note.
        # This avoids paying the heavy think model just to create an initial plan.
        # ------------------------------------------------------------------
        template_cfg = self._initial_think_template_cfg()
        template_enabled = template_cfg.get("enabled")
        if template_enabled is None:
            template_enabled = plan_template_defaults.DEFAULT_INITIAL_THINK_TEMPLATE_ENABLED
        if (
            bool(template_enabled)
            and not steps
            and not evidences
            and getattr(self, "_think_prompt_fingerprint", None) is not None
        ):
            llm = self._resolve_llm_connector()
            templates = coerce_templates()
            model_name = template_cfg.get("model_name")
            if isinstance(model_name, str) and model_name.strip():
                model_name = model_name.strip()
            else:
                model_name = None
            # If caller did not specify a model, prefer the connector's low_cost_model_name when present.
            if model_name is None and llm is not None:
                cfg_obj = getattr(llm, "config", None)
                low_cost = getattr(cfg_obj, "low_cost_model_name", None) if cfg_obj is not None else None
                low_cost = str(low_cost or "").strip()
                if low_cost:
                    model_name = low_cost

            try:
                selection = await select_plan_template(
                    llm_connector=llm,
                    question=question,
                    templates=templates,
                    model_name=model_name,
                    temperature=template_cfg.get("temperature"),
                    max_tokens=template_cfg.get("max_tokens"),
                    attempts=template_cfg.get("json_attempts"),
                )
            except Exception:
                selection = None

            if selection is not None and selection.use_template and selection.template_id:
                try:
                    plan_items, tool_calls, signature = instantiate_template_plan(
                        templates=templates,
                        template_id=selection.template_id,
                        question=question,
                        slots=selection.slots,
                    )
                except Exception:
                    plan_items, tool_calls, signature = [], [], ""

                if plan_items or tool_calls:
                    plan_state = PlanState()
                    plan_state.update(plan_items)
                    report_style = str(selection.report_style or "deepsearch").strip().lower()
                    if report_style not in {"deepsearch", "research"}:
                        report_style = "deepsearch"
                    if isinstance(reasoning_context.metadata, dict):
                        reasoning_context.metadata["report_style"] = report_style
                        reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
                        reasoning_context.metadata["plan_template"] = {
                            "template_id": selection.template_id,
                            "signature": signature or None,
                            "slots": dict(selection.slots),
                        }
                        self._attach_initial_think_signals(
                            reasoning_context,
                            is_computable=bool(getattr(selection, "is_computable", False)),
                            question_kind=self._normalize_question_kind(getattr(selection, "question_kind", "file_qa")),
                            decomposition=(
                                self._coerce_decomposition(
                                    selection.raw.get("decomposition") if isinstance(selection.raw, dict) else None,
                                    question=question,
                                )
                            ),
                        )

                    raw = {
                        "reasoning": selection.reasoning or f"Using plan template: {selection.template_id}",
                        "tool_calls": list(tool_calls),
                        "plan": list(plan_state.items),
                        "report_needed": bool(selection.report_needed),
                        "report_style": report_style,
                        "is_computable": bool(getattr(selection, "is_computable", False)),
                        "question_kind": self._normalize_question_kind(getattr(selection, "question_kind", "file_qa")),
                        "decomposition": (
                            dict(selection.raw.get("decomposition"))
                            if isinstance(selection.raw, dict) and isinstance(selection.raw.get("decomposition"), dict)
                            else None
                        ),
                        "template": {
                            "template_id": selection.template_id,
                            "signature": signature or None,
                            "slots": dict(selection.slots),
                            "fingerprint": build_template_fingerprint(),
                        },
                    }
                    note = ThinkNote(
                        plan_step_id="think_init",
                        reasoning=str(raw["reasoning"]).strip() or "Template planning checkpoint.",
                        next_actions=[
                            f"Execute initial tool call: {tool_calls[0]['tool_name']}" if tool_calls else "Continue with tool-driven retrieval.",
                        ],
                        metadata={"raw": raw},
                    )

                    await emit_trace(
                        "write_outline",
                        plan_state.markdown,
                        meta={
                            "stage": "think_init",
                            "plan_step_id": "think_init",
                            "plan_version": plan_state.version,
                            "plan_items": list(plan_state.items),
                            "template": raw.get("template"),
                        },
                    )
                    await emit_trace(
                        "think",
                        "Initial think checkpoint (template mode).\n"
                        f"template_id={selection.template_id}\n"
                        f"reasoning={note.reasoning}\n"
                        f"initial_tool_calls={[c.get('tool_name') for c in tool_calls]}",
                        meta={
                            "stage": "think_init",
                            "plan_step": "think_init",
                            "template": raw.get("template"),
                        },
                    )

                    note_payloads = [note.model_dump(exclude_none=True)]
                    # Cache template result as an initial-think output (same eligibility as cache path).
                    if cache is not None and cache_key is not None:
                        try:
                            cache.set(
                                cache_key,
                                CachedInitialThink(
                                    report_needed=bool(selection.report_needed),
                                    report_style=report_style,
                                    is_computable=bool(raw.get("is_computable") or False),
                                    question_kind=str(raw.get("question_kind") or "file_qa"),
                                    decomposition=(dict(raw.get("decomposition")) if isinstance(raw.get("decomposition"), dict) else None),
                                    raw=dict(raw),
                                    plan_items=list(plan_state.items),
                                    think_notes_payloads=list(note_payloads),
                                ),
                            )
                        except Exception:
                            pass

                    return {
                        "report_needed": bool(selection.report_needed),
                        "report_style": report_style,
                        "is_computable": bool(raw.get("is_computable") or False),
                        "question_kind": str(raw.get("question_kind") or "file_qa"),
                        "plan_state": plan_state,
                        "think_notes": note_payloads,
                        "think_notes_obj": [note],
                        "raw": raw,
                        "cache": {"hit": False, "template": True} if cache is not None and cache_key is not None else {"template": True},
                    }

        plan_state = PlanState()
        if isinstance(reasoning_context.metadata, dict):
            plan_state.update(reasoning_context.metadata.get("runtime_plan"))
        payload = {
            "question": question,
            "plan_step": "think_init",
            "context_evidences": evidences,
            "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
            "access_scope": scope,
            "extra": {
                "trigger": "initial_think",
                "plan_steps": steps,
                "current_plan": list(plan_state.items),
                "think_mode": "initial",
            },
            "graph_context": reasoning_context.model_dump(exclude_none=True),
            "coverage_metrics": {
                "evidence_count": len(evidences),
                "unique_source_count": len({(ev.get("source") if isinstance(ev, dict) else None) for ev in evidences}),
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
        timeout = self._resolve_tool_timeout_seconds()
        invocation = tool_manager.invoke(self._think_tool_name(), payload=payload)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(invocation, timeout=timeout)
        else:
            result = await invocation

        notes = getattr(result, "think_notes", None)
        if not notes:
            raise RuntimeError("Initial think returned no think_notes")

        raw = None
        for note in reversed(list(notes)):
            raw = note.metadata.get("raw") if isinstance(note.metadata, dict) else None
            if isinstance(raw, dict):
                break
        if not isinstance(raw, dict):
            raise RuntimeError("Initial think returned no structured payload")
        report_needed = raw.get("report_needed")
        if report_needed is None:
            raise RuntimeError("Initial think missing report_needed")
        # Backward compatibility: older/custom think implementations may omit these signals.
        if "is_computable" not in raw:
            raw["is_computable"] = False
        if "question_kind" not in raw:
            raw["question_kind"] = "file_qa"
        report_style_raw = raw.get("report_style")
        report_style = str(report_style_raw or "").strip().lower() if report_style_raw is not None else ""
        if report_style not in {"deepsearch", "research"}:
            report_style = "deepsearch"
        if isinstance(reasoning_context.metadata, dict):
            reasoning_context.metadata["report_style"] = report_style
            self._attach_initial_think_signals(
                reasoning_context,
                is_computable=bool(raw.get("is_computable") or False),
                question_kind=self._normalize_question_kind(raw.get("question_kind")),
                decomposition=self._coerce_decomposition(raw.get("decomposition"), question=question),
            )

        if update_plan_from_think_notes(plan_state, think_notes=notes):
            reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
            await emit_trace(
                "write_outline",
                plan_state.markdown,
                meta={
                    "stage": "think_init",
                    "plan_step_id": "think_init",
                    "plan_version": plan_state.version,
                    "plan_items": list(plan_state.items),
                },
            )

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

        await emit_trace("think", "\n".join(lines), meta={"stage": "think_init", "plan_step": "think_init"})
        note_payloads = [note.model_dump(exclude_none=True) for note in notes]
        # Store cache entry after a successful run (same eligibility as above).
        if cache is not None and cache_key is not None:
            try:
                cache.set(
                    cache_key,
                    CachedInitialThink(
                        report_needed=bool(report_needed),
                        report_style=report_style,
                        is_computable=bool(raw.get("is_computable") or False),
                        question_kind=self._normalize_question_kind(raw.get("question_kind")),
                        decomposition=(dict(raw.get("decomposition")) if isinstance(raw.get("decomposition"), dict) else None),
                        raw=dict(raw),
                        plan_items=list(plan_state.items),
                        think_notes_payloads=list(note_payloads),
                    ),
                )
            except Exception:
                pass
        return {
            "report_needed": bool(report_needed),
            "report_style": report_style,
            "is_computable": bool(raw.get("is_computable") or False),
            "question_kind": self._normalize_question_kind(raw.get("question_kind")),
            "plan_state": plan_state,
            "think_notes": note_payloads,
            "think_notes_obj": list(notes),
            "raw": raw,
            "cache": {"hit": False} if cache is not None and cache_key is not None else None,
        }

    async def _run_final_think(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
        reasoning_context: GraphQueryContext,
        evidences: Sequence[Dict[str, Any]] | None,
        coverage_metrics: Dict[str, Any] | None,
        plan_items: Sequence[Dict[str, Any]] | None,
        report_needed: bool | None = None,
        final_answer_mode: str | None = None,
    ) -> Dict[str, Any]:
        tool_manager = getattr(self, "tool_manager", None)
        if not tool_manager:
            raise RuntimeError("DeepSearchService requires a tool_manager for final_think")

        plan_state = PlanState()
        plan_state.update(plan_items or [])
        extra = {
            "trigger": "final_think",
            "current_plan": list(plan_state.items),
            "think_mode": "final",
        }
        if report_needed is not None:
            extra["report_needed"] = bool(report_needed)
        if final_answer_mode:
            extra["final_answer_mode"] = str(final_answer_mode)
        payload = {
            "question": question,
            "plan_step": "think_final",
            "context_evidences": list(evidences or []),
            "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
            "access_scope": scope,
            "extra": extra,
            "graph_context": reasoning_context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_metrics or {},
        }
        timeout = self._resolve_tool_timeout_seconds()
        invocation = tool_manager.invoke(self._think_tool_name(), payload=payload)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(invocation, timeout=timeout)
        else:
            result = await invocation

        notes = getattr(result, "think_notes", None) or []
        raw = None
        for note in reversed(list(notes)):
            raw = note.metadata.get("raw") if isinstance(note.metadata, dict) else None
            if isinstance(raw, dict):
                break
        raw = raw if isinstance(raw, dict) else {}

        if notes and update_plan_from_think_notes(plan_state, think_notes=notes):
            await emit_trace(
                "write_outline",
                plan_state.markdown,
                meta={
                    "stage": "final_think",
                    "plan_step_id": "think_final",
                    "plan_version": plan_state.version,
                    "plan_items": list(plan_state.items),
                },
            )

        lines: List[str] = ["Final think checkpoint."]
        for idx, note in enumerate(notes, start=1):
            lines.append(f"note_{idx}. reasoning={note.reasoning}")
        await emit_trace("think", "\n".join(lines), meta={"stage": "final_think", "plan_step": "think_final"})
        note_payloads = [note.model_dump(exclude_none=True) for note in notes]
        return {
            "think_notes": note_payloads,
            "plan_state": plan_state,
            "raw": raw,
        }
