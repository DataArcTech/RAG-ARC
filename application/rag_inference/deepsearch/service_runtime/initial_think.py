import asyncio
from typing import Any, Dict, List, Sequence

from core.deepsearch.query_spec import generate_query_spec
from core.deepsearch.trace import emit_trace
from core.deepsearch.memory.plan_state import PlanState, update_plan_from_think_notes
from core.deepsearch.planning.template_planner import (
    coerce_templates,
    instantiate_template_plan,
    select_plan_template,
    PlanTemplateSelection,
)
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext
from encapsulation.data_model.deepsearch import ThinkNote

from application.rag_inference.deepsearch.runtime_cache import CachedInitialThink
from config.benchmark_mode import benchmark_mode_enabled
from config.core.deepsearch import plan_template_defaults


_DEFAULT_PLAN_ITEMS: List[Dict[str, Any]] = [
    {"text": "Locate relevant file(s) for the question.", "checked": False},
    {"text": "Locate pages containing the answer.", "checked": False},
    {"text": "Read full pages as citeable evidence.", "checked": False},
    {"text": "Submit final answer with citations.", "checked": False},
]


class DeepSearchServiceInitialThinkMixin:
    @staticmethod
    def _attach_initial_think_signals(
        reasoning_context: GraphQueryContext,
    ) -> None:
        if not isinstance(reasoning_context.metadata, dict):
            return
        initial_think = {
            "page_indexing": {"tool_index_base": 0, "human_index_base": 1},
        }
        reasoning_context.metadata["initial_think"] = initial_think

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

    def _resolve_query_spec_model(self) -> str | None:
        """Resolve model name for QuerySpec generation (prefer low-cost model)."""
        llm = self._resolve_llm_connector()
        if llm is None:
            return None
        cfg_obj = getattr(llm, "config", None)
        low_cost = getattr(cfg_obj, "low_cost_model_name", None) if cfg_obj is not None else None
        low_cost = str(low_cost or "").strip()
        return low_cost if low_cost else None

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
        evidences = list(context_evidences or [])

        # ------------------------------------------------------------------
        # Bench mode: skip query_spec but still run template planner for
        # question-specific plan items. Only QuerySpec classification is
        # skipped to avoid confounding algorithm comparisons.
        # ------------------------------------------------------------------
        if benchmark_mode_enabled():
            # Run template selection if enabled (lightweight LLM call).
            llm = self._resolve_llm_connector()
            template_enabled = bool(getattr(plan_template_defaults, "DEFAULT_INITIAL_THINK_TEMPLATE_ENABLED", True))
            initial_tool_calls: list[Dict[str, Any]] = []
            template_sig: str | None = None
            template_selection = None

            if llm is not None and template_enabled:
                model_name = self._resolve_query_spec_model()
                try:
                    template_selection = await select_plan_template(
                        llm_connector=llm, question=question, model_name=model_name,
                    )
                except Exception:
                    template_selection = None

            if (
                isinstance(template_selection, PlanTemplateSelection)
                and template_selection.use_template
                and template_selection.template_id
            ):
                try:
                    templates = coerce_templates()
                    plan_items_list, initial_tool_calls, template_sig = instantiate_template_plan(
                        templates=templates,
                        template_id=template_selection.template_id,
                        question=question,
                        slots=template_selection.slots,
                    )
                except Exception:
                    plan_items_list = list(_DEFAULT_PLAN_ITEMS)
            else:
                plan_items_list = list(_DEFAULT_PLAN_ITEMS)

            plan_state = PlanState()
            plan_state.update(plan_items_list)

            if isinstance(reasoning_context.metadata, dict):
                reasoning_context.metadata["report_style"] = "deepsearch"
                reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
                self._attach_initial_think_signals(reasoning_context)

            raw: Dict[str, Any] = {
                "reasoning": "Benchmark mode: query_spec skipped, template planner active.",
                "tool_calls": initial_tool_calls,
                "plan": list(plan_state.items),
                "report_needed": True,
                "report_style": "deepsearch",
            }
            note = ThinkNote(
                plan_step_id="think_init",
                reasoning=raw["reasoning"],
                next_actions=["Execute locate for file routing."],
                metadata={"raw": raw},
            )
            note_payloads = [note.model_dump(exclude_none=True)]

            await emit_trace(
                "write_outline",
                plan_state.markdown,
                meta={
                    "stage": "think_init",
                    "plan_step_id": "think_init",
                    "plan_version": plan_state.version,
                    "plan_items": list(plan_state.items),
                    "bench_mode": True,
                    "template_sig": template_sig,
                    "template_id": template_selection.template_id if isinstance(template_selection, PlanTemplateSelection) else None,
                },
            )
            await emit_trace(
                "think",
                "Initial think checkpoint (bench_mode: query_spec skipped, template planner active).",
                meta={
                    "stage": "think_init",
                    "plan_step": "think_init",
                    "bench_mode": True,
                    "template_sig": template_sig,
                    "template_id": template_selection.template_id if isinstance(template_selection, PlanTemplateSelection) else None,
                },
            )

            return {
                "report_needed": True,
                "report_style": "deepsearch",
                "plan_state": plan_state,
                "think_notes": note_payloads,
                "think_notes_obj": [note],
                "raw": raw,
                "query_spec": None,
                "cache": {"bench_mode": True},
            }

        # ------------------------------------------------------------------
        # Cache (conservative): only cache pure initial-think calls (no
        # injected evidences / plan steps).
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

                if isinstance(reasoning_context.metadata, dict):
                    reasoning_context.metadata["report_style"] = report_style
                    reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
                    self._attach_initial_think_signals(reasoning_context)
                    cached_query_spec = getattr(hit, "query_spec", None)
                    if isinstance(cached_query_spec, dict):
                        reasoning_context.metadata["query_spec"] = dict(cached_query_spec)

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

                return {
                    "report_needed": bool(hit.report_needed),
                    "report_style": report_style,
                    "plan_state": plan_state,
                    "think_notes": list(hit.think_notes_payloads),
                    "think_notes_obj": note_objs,
                    "raw": dict(hit.raw),
                    "cache": {"hit": True},
                    "query_spec": getattr(hit, "query_spec", None),
                }

        # ------------------------------------------------------------------
        # QuerySpec generation (lightweight 1x LLM call).
        # Input: ONLY question + target_langs.  No adapter, no graph_context.
        # ------------------------------------------------------------------
        llm = self._resolve_llm_connector()
        if llm is None:
            # Legacy fallback: if no LLM connector is configured for QuerySpec generation,
            # fall back to invoking the think tool for initial planning.
            extra = {
                "trigger": "initial_think",
                "think_mode": "initial",
            }
            payload = {
                "question": question,
                "plan_step": "think_init",
                "context_evidences": list(evidences or []),
                "adapter": getattr(getattr(self, "graph_loop", None), "adapter", None),
                "access_scope": scope,
                "extra": extra,
                "graph_context": reasoning_context.model_dump(exclude_none=True),
                "coverage_metrics": {},
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
            report_needed = bool(raw.get("report_needed", True))
            report_style = str(raw.get("report_style", "deepsearch") or "").strip().lower()
            if report_style not in {"deepsearch", "research"}:
                report_style = "deepsearch"
            plan_items_list = raw.get("plan") if isinstance(raw.get("plan"), list) else []

            plan_state = PlanState()
            plan_state.update(plan_items_list)
            if notes and update_plan_from_think_notes(plan_state, think_notes=notes):
                await emit_trace(
                    "write_outline",
                    plan_state.markdown,
                    meta={
                        "stage": "think_init",
                        "plan_step_id": "think_init",
                        "plan_version": plan_state.version,
                        "plan_items": list(plan_state.items),
                        "cache": {"hit": False, "fallback": "think_tool"},
                    },
                )

            fallback_query_spec = raw.get("query_spec")
            if isinstance(reasoning_context.metadata, dict):
                reasoning_context.metadata["report_style"] = report_style
                reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
                self._attach_initial_think_signals(reasoning_context)
                if isinstance(fallback_query_spec, dict):
                    reasoning_context.metadata["query_spec"] = dict(fallback_query_spec)

            await emit_trace(
                "think",
                "Initial think checkpoint (think-tool fallback; query_spec unavailable).",
                meta={"stage": "think_init", "plan_step": "think_init", "fallback": "think_tool"},
            )

            note_payloads = [note.model_dump(exclude_none=True) for note in notes]
            if cache is not None and cache_key is not None:
                try:
                    cache.set(
                        cache_key,
                        CachedInitialThink(
                            report_needed=report_needed,
                            report_style=report_style,
                            raw=dict(raw),
                            plan_items=list(plan_state.items),
                            think_notes_payloads=list(note_payloads),
                            query_spec=dict(fallback_query_spec) if isinstance(fallback_query_spec, dict) else None,
                        ),
                    )
                except Exception:
                    pass

            return {
                "report_needed": report_needed,
                "report_style": report_style,
                "plan_state": plan_state,
                "think_notes": note_payloads,
                "think_notes_obj": list(notes),
                "raw": raw,
                "cache": {"hit": False, "fallback": "think_tool"} if cache is not None and cache_key is not None else {"fallback": "think_tool"},
                "query_spec": dict(fallback_query_spec) if isinstance(fallback_query_spec, dict) else None,
            }

        model_name = self._resolve_query_spec_model()

        # Run QuerySpec and template selection in parallel (zero extra latency).
        template_enabled = bool(getattr(plan_template_defaults, "DEFAULT_INITIAL_THINK_TEMPLATE_ENABLED", True))
        query_spec_coro = generate_query_spec(
            llm_connector=llm,
            question=question,
            model_name=model_name,
        )
        if template_enabled:
            template_coro = select_plan_template(
                llm_connector=llm, question=question, model_name=model_name,
            )
            query_spec, template_selection = await asyncio.gather(
                query_spec_coro, template_coro, return_exceptions=True,
            )
            # Graceful fallback if either fails
            if isinstance(query_spec, BaseException):
                query_spec = {"report_needed": True, "report_style": "deepsearch", "bm25_terms": [], "regex_patterns": [], "reasoning": "QuerySpec failed."}
            if isinstance(template_selection, BaseException):
                template_selection = None
        else:
            query_spec = await query_spec_coro
            template_selection = None

        report_needed = bool(query_spec.get("report_needed", True))
        report_style = query_spec.get("report_style", "deepsearch")

        # Derive plan items: template > default
        initial_tool_calls: list[Dict[str, Any]] = []
        template_sig: str | None = None
        if (
            isinstance(template_selection, PlanTemplateSelection)
            and template_selection.use_template
            and template_selection.template_id
        ):
            try:
                templates = coerce_templates()
                plan_items_list, initial_tool_calls, template_sig = instantiate_template_plan(
                    templates=templates,
                    template_id=template_selection.template_id,
                    question=question,
                    slots=template_selection.slots,
                )
            except Exception:
                plan_items_list = list(_DEFAULT_PLAN_ITEMS)
        elif not report_needed:
            plan_items_list = [
                {"text": "Answer the general concept question directly.", "checked": False},
            ]
        else:
            plan_items_list = list(_DEFAULT_PLAN_ITEMS)

        plan_state = PlanState()
        plan_state.update(plan_items_list)

        if isinstance(reasoning_context.metadata, dict):
            reasoning_context.metadata["report_style"] = report_style
            reasoning_context.metadata["runtime_plan"] = list(plan_state.items)
            self._attach_initial_think_signals(reasoning_context)
            # Propagate query_spec to graph_context for downstream tools (locate)
            reasoning_context.metadata["query_spec"] = dict(query_spec)

        raw = {
            "reasoning": query_spec.get("reasoning", "QuerySpec generation."),
            "tool_calls": initial_tool_calls,
            "plan": list(plan_state.items),
            "report_needed": report_needed,
            "report_style": report_style,
            "query_spec": query_spec,
        }

        note = ThinkNote(
            plan_step_id="think_init",
            reasoning=str(raw["reasoning"]).strip() or "QuerySpec generation.",
            next_actions=["Execute locate for file routing."],
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
                "query_spec": True,
                "template_sig": template_sig,
                "template_id": template_selection.template_id if isinstance(template_selection, PlanTemplateSelection) else None,
            },
        )
        await emit_trace(
            "think",
            f"Initial think checkpoint (query_spec mode).\n"
            f"bm25_terms={query_spec.get('bm25_terms', [])}\n"
            f"regex_patterns={query_spec.get('regex_patterns', [])}",
            meta={
                "stage": "think_init",
                "plan_step": "think_init",
                "query_spec": True,
                "template_sig": template_sig,
                "template_id": template_selection.template_id if isinstance(template_selection, PlanTemplateSelection) else None,
            },
        )

        note_payloads = [note.model_dump(exclude_none=True)]

        if cache is not None and cache_key is not None:
            try:
                cache.set(
                    cache_key,
                    CachedInitialThink(
                        report_needed=report_needed,
                        report_style=report_style,
                        raw=dict(raw),
                        plan_items=list(plan_state.items),
                        think_notes_payloads=list(note_payloads),
                        query_spec=dict(query_spec),
                    ),
                )
            except Exception:
                pass

        return {
            "report_needed": report_needed,
            "report_style": report_style,
            "plan_state": plan_state,
            "think_notes": note_payloads,
            "think_notes_obj": [note],
            "raw": raw,
            "cache": {"hit": False, "query_spec": True} if cache is not None and cache_key is not None else {"query_spec": True},
            "query_spec": dict(query_spec),
        }
