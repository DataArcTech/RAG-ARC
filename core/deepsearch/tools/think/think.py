"""Think tool that offers structured reasoning pauses."""
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config.core.deepsearch import tool_defaults
from encapsulation.data_model.deepsearch import ThinkNote, GraphQueryContext, PlanItem
from core.prompts.deepsearch import (
    THINK_TOOL_SYSTEM_PROMPT_EN,
    THINK_TOOL_SYSTEM_PROMPT_FINAL_EN,
    THINK_TOOL_SYSTEM_PROMPT_GATE_EN,
    THINK_TOOL_SYSTEM_PROMPT_INITIAL_EN,
)
from core.deepsearch.utils.evidence_cards import evidence_cards
from core.deepsearch.utils.llm_envelope import build_llm_envelope
from core.utils.llm_json import repair_json_from_raw_with_retry

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, safe_json_loads
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER
from . import final as final_mode
from . import initial as initial_mode
from . import normal as normal_mode


class ThinkToolCall(BaseModel):
    tool_name: str = Field(..., min_length=1)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)
    parallelizable: bool = Field(default=False)


class ThinkToolResponse(BaseModel):
    reasoning: str = Field(..., min_length=1)
    tool_calls: List[ThinkToolCall] = Field(default_factory=list)
    plan: List[PlanItem] = Field(...)
    report_needed: Optional[bool] = None
    report_style: Optional[str] = None
    is_final: Optional[bool] = None


class ThinkTool(GraphTool):
    """Implements structured think windows for multi-tool reasoning."""

    descriptor = ToolDescriptor(
        name="think",
        channel="graph",
        description=(
            "Structured pause that digests current context before the next hop. "
            "Evidence: derived think notes (NOT citeable). "
            "Good: summarize gaps + propose next tool calls. Bad: invent facts or call think."
        ),
        speed="slow",
        cost="low",
        strategy_tags=("think", "control", "reflection", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_LLM),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.think",
        mcp_callable=True,
        example_args={
            "question": "What gaps remain?",
            "plan_step": "plan_02",
            "context_evidences": [],
            "coverage_metrics": {"coverage_score": 0.3, "confidence_score": 0.4},
        },
    )

    def __init__(
        self,
        llm_connector=None,
        *,
        temperature: float = 0.1,
        system_prompt: str = THINK_TOOL_SYSTEM_PROMPT_EN,
        max_tokens: int | None = tool_defaults.THINK_DEFAULT_MAX_TOKENS,
        include_extra_in_prompt: bool = tool_defaults.THINK_INCLUDE_EXTRA_IN_PROMPT,
        json_repair_attempts: int = tool_defaults.THINK_JSON_REPAIR_DEFAULT_ATTEMPTS,
        json_repair_temperature: float = tool_defaults.THINK_JSON_REPAIR_DEFAULT_TEMPERATURE,
        json_repair_max_raw_chars: int = tool_defaults.THINK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS,
        json_repair_model: str | None = None,
    ):
        self.llm_connector = llm_connector
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = None if max_tokens is None else max(0, int(max_tokens))
        self.include_extra_in_prompt = bool(include_extra_in_prompt)
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.json_repair_temperature = float(json_repair_temperature)
        self.json_repair_model = str(json_repair_model).strip() if isinstance(json_repair_model, str) and json_repair_model.strip() else None
        max_raw_chars = int(json_repair_max_raw_chars)
        self.json_repair_max_raw_chars = (
            max_raw_chars if max_raw_chars > 0 else tool_defaults.THINK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS
        )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        note = await self._build_note(request)
        context_snapshot = self._graph_context_snapshot(request.graph_context)
        coverage_snapshot = request.coverage_metrics or {}
        diagnostics = {
            "plan_step": request.plan_step,
            "temperature": self.temperature,
            "graph_context": context_snapshot,
            "coverage_metrics": coverage_snapshot,
        }
        diagnostics["thought_log"] = [
            self._build_thought_log_entry(
                note,
                coverage_snapshot=coverage_snapshot,
            )
        ]
        meta = note.metadata if isinstance(note.metadata, dict) else {}
        tool_calls = meta.get("tool_calls") or []
        next_steps: List[Dict[str, Any]] = []
        if isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("tool_name") or "").strip()
                if not name:
                    continue
                next_steps.append(
                    {
                        "tool_name": name,
                        "rationale": str(call.get("rationale") or "").strip() or None,
                    }
                )
                if len(next_steps) >= 4:
                    break
        summary = build_llm_envelope(
            thinking=note.reasoning,
            answer={
                "tool_calls": tool_calls,
                "plan": meta.get("plan") or [],
                "report_needed": meta.get("report_needed"),
                "report_style": meta.get("report_style"),
                "is_final": meta.get("is_final"),
            },
            extra={
                "plan_step": request.plan_step,
                "think_mode": str((request.extra or {}).get("think_mode") or "normal"),
                "next_steps": next_steps,
            },
        )
        return ToolResult(summary=summary, diagnostics=diagnostics, think_notes=[note])

    async def _build_note(self, request: ToolRunRequest) -> ThinkNote:
        context_snapshot = self._graph_context_snapshot(request.graph_context)
        coverage_snapshot = request.coverage_metrics or {}
        if not self.llm_connector:
            raise RuntimeError("ThinkTool requires an LLM connector")

        # Do not inline full evidence text in think prompts.
        # The EvidencePool/EvidenceBank stores full content; the think tool consumes metadata-only cards.
        cards = evidence_cards(request.context_evidences or [])
        extra = request.extra or {}
        available_tools = extra.get("available_tools")
        previous_tool_call_results = extra.get("previous_tool_call_results")
        recent_tool_runs = extra.get("recent_tool_runs")
        current_plan = extra.get("current_plan")
        tool_budget_snapshot = (
            (context_snapshot.get("metadata") or {}).get("tool_budget") if isinstance(context_snapshot, dict) else None
        )
        budget_status = self._build_budget_status(tool_budget_snapshot)
        prompt_payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_evidences": cards,
            "graph_context": context_snapshot,
            "tool_budget": tool_budget_snapshot,
            "budget_status": budget_status,
            "coverage_metrics": coverage_snapshot,
            "available_tools": available_tools,
            "previous_tool_call_results": previous_tool_call_results,
            "recent_tool_runs": recent_tool_runs,
            "current_plan": current_plan,
        }
        if self.include_extra_in_prompt:
            # Optional debug escape hatch (disabled by default) since `extra` can be large/redundant.
            prompt_payload["extra"] = extra
        else:
            # Keep only minimal runtime markers (helps debugging without bloating prompts).
            prompt_payload["runtime"] = {
                "trigger": extra.get("trigger"),
                "round": extra.get("round"),
                "think_mode": extra.get("think_mode"),
                "budget_phase": (budget_status or {}).get("phase") if isinstance(budget_status, dict) else None,
            }
        prompt_json = self._serialize_payload(prompt_payload)
        prompt_stats = {
            "prompt_chars": len(prompt_json),
            "evidence_cards": len(cards),
            "available_tools_count": len(available_tools) if isinstance(available_tools, list) else None,
            "previous_tool_call_results_count": len(previous_tool_call_results) if isinstance(previous_tool_call_results, list) else None,
            "recent_tool_runs_count": len(recent_tool_runs) if isinstance(recent_tool_runs, list) else None,
            "current_plan_count": len(current_plan) if isinstance(current_plan, list) else None,
            "budget_phase": (budget_status or {}).get("phase") if isinstance(budget_status, dict) else None,
            "budget_remaining_calls": (budget_status or {}).get("remaining_calls") if isinstance(budget_status, dict) else None,
            "max_tokens": self.max_tokens,
            "include_extra_in_prompt": self.include_extra_in_prompt,
        }
        system_prompt = self._select_system_prompt(
            extra=extra,
            previous_tool_call_results=previous_tool_call_results,
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_json}]
        mode = str((request.extra or {}).get("think_mode") or normal_mode.MODE).strip().lower()
        mode_defaults: Dict[str, Any] | None = None
        try:
            response = await call_llm_async(
                self.llm_connector,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                warn_context="deepsearch.think.init",
            )
            try:
                parsed = await self._parse_or_repair_json(messages=messages, raw=response)
            except ValueError as exc:
                # Robustness: some models occasionally ignore the JSON-only contract, especially in final mode.
                # In final mode, we can safely wrap plain text into a minimal valid payload (no tool calls).
                if mode == final_mode.MODE:
                    current_plan = (request.extra or {}).get("current_plan")
                    parsed = {
                        "reasoning": str(response or "").strip() or "（空输出）",
                        "tool_calls": [],
                        "plan": list(current_plan) if isinstance(current_plan, list) else [],
                        "is_final": True,
                    }
                    mode_defaults = {
                        **(mode_defaults or {}),
                        "filled_from_plain_text": True,
                        "parse_error": str(exc),
                    }
                else:
                    raise
            schema_repair = None
            try:
                normalized = self._normalize_payload(parsed)
                normalized, tool_call_norm = self._normalize_tool_calls_in_payload(normalized)
                if tool_call_norm:
                    mode_defaults = {**(mode_defaults or {}), "tool_call_normalization": tool_call_norm}
                # If the model omits the plan (common on weaker models), reuse the current plan
                # snapshot from the runtime. This keeps the loop robust without hardcoding new steps.
                if isinstance(normalized, dict) and not isinstance(normalized.get("plan"), list):
                    current_plan = (request.extra or {}).get("current_plan")
                    if isinstance(current_plan, list):
                        normalized = dict(normalized)
                        normalized["plan"] = list(current_plan)
                        mode_defaults = {**(mode_defaults or {}), "filled_plan_from_current_plan": True}
                    else:
                        normalized = dict(normalized)
                        normalized["plan"] = []
                        mode_defaults = {**(mode_defaults or {}), "filled_plan_empty": True}
                # Mode-aware coercions (minimal + observable):
                # - In think_mode=final, tool_calls are not executable; drop them to enforce the contract.
                # - In think_mode=final, some models omit `is_final`; when there are no tool calls, treat omission as
                #   `is_final=true` for robustness (still observable via mode_defaults).
                if mode == final_mode.MODE and isinstance(normalized, dict):
                    tool_calls = normalized.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        normalized = dict(normalized)
                        normalized["tool_calls"] = []
                        mode_defaults = {**(mode_defaults or {}), "dropped_tool_calls": len(tool_calls)}
                    if normalized.get("is_final") is not True and not normalized.get("tool_calls"):
                        normalized = dict(normalized)
                        normalized["is_final"] = True
                        mode_defaults = {**(mode_defaults or {}), "filled_is_final": True}
                payload = ThinkToolResponse.model_validate(normalized)
                self._validate_mode(payload, extra)
            except Exception as exc:
                repaired = await self._attempt_json_repair(
                    messages=messages,
                    raw=self._serialize_payload(parsed if isinstance(parsed, dict) else {"raw": parsed}),
                    expected="dict",
                    error=str(exc),
                )
                if not isinstance(repaired, dict):
                    raise
                schema_repair = {"error": str(exc)}
                parsed = repaired
                normalized = self._normalize_payload(parsed)
                normalized, tool_call_norm = self._normalize_tool_calls_in_payload(normalized)
                if tool_call_norm:
                    mode_defaults = {**(mode_defaults or {}), "tool_call_normalization": tool_call_norm, "after_repair": True}
                if isinstance(normalized, dict) and not isinstance(normalized.get("plan"), list):
                    current_plan = (request.extra or {}).get("current_plan")
                    if isinstance(current_plan, list):
                        normalized = dict(normalized)
                        normalized["plan"] = list(current_plan)
                        mode_defaults = {**(mode_defaults or {}), "filled_plan_from_current_plan": True, "after_repair": True}
                    else:
                        normalized = dict(normalized)
                        normalized["plan"] = []
                        mode_defaults = {**(mode_defaults or {}), "filled_plan_empty": True, "after_repair": True}
                if mode == final_mode.MODE and isinstance(normalized, dict):
                    tool_calls = normalized.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        normalized = dict(normalized)
                        normalized["tool_calls"] = []
                        mode_defaults = {**(mode_defaults or {}), "dropped_tool_calls": len(tool_calls), "after_repair": True}
                    # If the model still omits is_final after repair, fall back to is_final=true only when
                    # no tool calls exist (content is already final-answer-style). Keep this observable.
                    if normalized.get("is_final") is not True and not normalized.get("tool_calls"):
                        normalized = dict(normalized)
                        normalized["is_final"] = True
                        mode_defaults = {**(mode_defaults or {}), "filled_is_final": True, "after_repair": True}
                payload = ThinkToolResponse.model_validate(normalized)
                self._validate_mode(payload, extra)
            return ThinkNote(
                plan_step_id=request.plan_step,
                reasoning=payload.reasoning,
                confidence_delta=None,
                coverage_delta=None,
                next_actions=[],
                metadata={
                    "raw": parsed,
                    "graph_context": context_snapshot,
                    "coverage_metrics": coverage_snapshot,
                    "prompt_stats": prompt_stats,
                    "tool_calls": [call.model_dump() for call in payload.tool_calls],
                    "plan": [item.model_dump() for item in payload.plan],
                    "report_needed": payload.report_needed,
                    "report_style": payload.report_style,
                    "is_final": payload.is_final,
                    "schema_repair": schema_repair,
                    "mode_defaults": mode_defaults,
                },
            )
        except Exception as exc:
            raise RuntimeError(f"ThinkTool failed: {exc}") from exc

    @staticmethod
    def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        if not str(payload.get("reasoning") or "").strip():
            for key in ("thought", "summary", "analysis"):
                candidate = payload.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    updated = dict(payload)
                    updated["reasoning"] = candidate.strip()
                    return updated
            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list):
                rationales = []
                for call in tool_calls:
                    if isinstance(call, dict):
                        rationale = call.get("rationale")
                        if isinstance(rationale, str) and rationale.strip():
                            rationales.append(rationale.strip())
                if rationales:
                    updated = dict(payload)
                    updated["reasoning"] = " ".join(rationales)
                    return updated
            plan = payload.get("plan")
            if isinstance(plan, list):
                plan_texts = []
                for item in plan:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            plan_texts.append(text.strip())
                if plan_texts:
                    updated = dict(payload)
                    updated["reasoning"] = " ".join(plan_texts)
                    return updated
        return payload

    @staticmethod
    def _normalize_tool_calls_in_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Normalize common function-call style tool schemas into ThinkToolCall schema.

        Some models emit tool calls in OpenAI/Anthropic-style "function" formats such as:
        - {"function": "explore", "arguments": {...}}
        - {"type": "function", "function": {"name": "explore", "arguments": "{...json...}"}}
        We normalize these into {"tool_name","tool_args","rationale","parallelizable"} and drop irrecoverable entries.
        """

        if not isinstance(payload, dict):
            return payload, {}

        tool_calls = payload.get("tool_calls")
        if tool_calls is None:
            return payload, {}
        if not isinstance(tool_calls, list):
            # Keep schema errors observable (schema repair will handle if needed).
            return payload, {"skipped": "tool_calls_not_list"}

        normed: list[dict[str, Any]] = []
        stats: dict[str, Any] = {"input": len(tool_calls), "output": 0, "dropped": 0, "rewritten": 0}
        for call in tool_calls:
            if not isinstance(call, dict):
                stats["dropped"] += 1
                continue

            updated: dict[str, Any] = {}

            # Extract tool name.
            tool_name = call.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                updated["tool_name"] = tool_name.strip()
            else:
                fn = call.get("function")
                if isinstance(fn, str) and fn.strip():
                    updated["tool_name"] = fn.strip()
                elif isinstance(fn, dict):
                    name = fn.get("name")
                    if isinstance(name, str) and name.strip():
                        updated["tool_name"] = name.strip()

            # Extract tool args.
            tool_args = call.get("tool_args")
            if isinstance(tool_args, dict):
                updated["tool_args"] = tool_args
            else:
                # Common alternatives: "arguments" at top-level or under function dict.
                args = call.get("arguments")
                fn = call.get("function")
                if args is None and isinstance(fn, dict):
                    args = fn.get("arguments")
                if isinstance(args, dict):
                    updated["tool_args"] = args
                elif isinstance(args, str) and args.strip():
                    parsed = safe_json_loads(args, expected="dict")
                    updated["tool_args"] = parsed if isinstance(parsed, dict) else {}
                else:
                    updated["tool_args"] = {}

            # Extract rationale (required by schema).
            rationale = call.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                updated["rationale"] = rationale.strip()
            else:
                reason = call.get("reason") or call.get("thought") or call.get("why")
                if isinstance(reason, str) and reason.strip():
                    updated["rationale"] = reason.strip()
                else:
                    updated["rationale"] = "normalized_from_function_call"

            # Preserve optional fields.
            if isinstance(call.get("parallelizable"), bool):
                updated["parallelizable"] = bool(call["parallelizable"])

            if not str(updated.get("tool_name") or "").strip():
                stats["dropped"] += 1
                continue

            if updated.keys() != call.keys():
                stats["rewritten"] += 1
            normed.append(updated)

        stats["output"] = len(normed)
        updated_payload = dict(payload)
        updated_payload["tool_calls"] = normed
        # Only return stats if anything changed (keeps metadata clean).
        if stats["dropped"] or stats["rewritten"]:
            return updated_payload, stats
        return updated_payload, {}

    async def _parse_or_repair_json(self, *, messages: List[Dict[str, str]], raw: str) -> Dict[str, Any]:
        parsed = safe_json_loads(raw, expected="dict")
        if isinstance(parsed, dict):
            return parsed
        repaired = await self._attempt_json_repair(messages=messages, raw=raw, expected="dict", error="invalid_json")
        if isinstance(repaired, dict):
            return repaired
        snippet = (raw or "").strip().replace("\n", "\\n")
        if len(snippet) > self.json_repair_max_raw_chars:
            snippet = snippet[: self.json_repair_max_raw_chars] + "…"
        raise ValueError(f"Think tool returned non-JSON or non-dict payload. raw_snippet={snippet}")

    async def _attempt_json_repair(
        self,
        *,
        messages: List[Dict[str, str]],
        raw: str,
        expected: str,
        error: str,
    ) -> Any:
        if self.json_repair_attempts <= 0:
            return None
        _ = error  # kept for compatibility/logging context
        model = self.json_repair_model
        if model is None:
            cfg = getattr(self.llm_connector, "config", None) if self.llm_connector is not None else None
            candidate = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
            if isinstance(candidate, str) and candidate.strip():
                model = candidate.strip()
        kwargs: Dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        repaired = await repair_json_from_raw_with_retry(
            llm_connector=self.llm_connector,
            messages=messages,
            raw=str(raw or ""),
            expected=expected,
            temperature=self.json_repair_temperature,
            attempts=self.json_repair_attempts,
            llm_kwargs=kwargs or None,
            include_today_line=True,
            max_raw_chars=self.json_repair_max_raw_chars,
        )
        return repaired

    @staticmethod
    def _graph_context_snapshot(graph_context: GraphQueryContext | None) -> Dict[str, Any]:
        if not graph_context:
            return {}
        return graph_context.model_dump(exclude_none=True)

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> str:
        def _default(value):
            return str(value)

        # Stable, token-efficient JSON improves cache hit rates and reduces JSON-repair churn.
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)

    @staticmethod
    def _is_missing_primary_page_evidence(previous_tool_call_results: Any) -> bool:
        if not isinstance(previous_tool_call_results, list):
            return False
        for row in previous_tool_call_results:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("failure_reason") or "").strip()
            if reason == "missing_primary_page_evidence":
                return True
        return False

    @staticmethod
    def _budget_phase(snapshot: Dict[str, Any]) -> str:
        try:
            max_calls_total = int(snapshot.get("max_calls_total") or 0)
        except Exception:
            max_calls_total = 0
        try:
            remaining_calls = int(snapshot.get("remaining_calls") or 0)
        except Exception:
            remaining_calls = 0

        remaining_ratio = 0.0
        if max_calls_total > 0:
            remaining_ratio = remaining_calls / float(max_calls_total)

        critical_calls = max(0, int(getattr(tool_defaults, "THINK_BUDGET_CRITICAL_REMAINING_CALLS", 1)))
        low_calls = max(0, int(getattr(tool_defaults, "THINK_BUDGET_LOW_REMAINING_CALLS", 3)))
        raw_critical_ratio = getattr(tool_defaults, "THINK_BUDGET_CRITICAL_REMAINING_RATIO", 0.05)
        raw_low_ratio = getattr(tool_defaults, "THINK_BUDGET_LOW_REMAINING_RATIO", 0.15)
        try:
            critical_ratio = float(raw_critical_ratio) if raw_critical_ratio is not None else 0.05
        except Exception:
            critical_ratio = 0.05
        try:
            low_ratio = float(raw_low_ratio) if raw_low_ratio is not None else 0.15
        except Exception:
            low_ratio = 0.15

        if remaining_calls <= critical_calls or remaining_ratio <= critical_ratio:
            return "critical"
        if remaining_calls <= low_calls or remaining_ratio <= low_ratio:
            return "low"
        return "ok"

    def _build_budget_status(self, tool_budget_snapshot: Any) -> Dict[str, Any] | None:
        if not bool(getattr(tool_defaults, "THINK_BUDGET_STATUS_ENABLED", True)):
            return None
        if not isinstance(tool_budget_snapshot, dict):
            return None

        try:
            max_calls_total = int(tool_budget_snapshot.get("max_calls_total") or 0)
        except Exception:
            max_calls_total = 0
        try:
            used_calls = int(tool_budget_snapshot.get("used_calls") or 0)
        except Exception:
            used_calls = 0
        try:
            remaining_calls = int(tool_budget_snapshot.get("remaining_calls") or 0)
        except Exception:
            remaining_calls = 0

        remaining_ratio = 0.0
        if max_calls_total > 0:
            remaining_ratio = remaining_calls / float(max_calls_total)
        phase = self._budget_phase(
            {
                "max_calls_total": max_calls_total,
                "used_calls": used_calls,
                "remaining_calls": remaining_calls,
            }
        )
        return {
            "max_calls_total": max_calls_total,
            "used_calls": used_calls,
            "remaining_calls": remaining_calls,
            "remaining_ratio": round(float(remaining_ratio), 3),
            "phase": phase,
        }

    def _select_system_prompt(
        self,
        *,
        extra: Dict[str, Any],
        previous_tool_call_results: Any,
    ) -> str:
        if not bool(getattr(tool_defaults, "THINK_PROMPT_VARIANTS_ENABLED", True)):
            return self.system_prompt

        mode = str((extra or {}).get("think_mode") or normal_mode.MODE).strip().lower()
        if mode == initial_mode.MODE:
            return THINK_TOOL_SYSTEM_PROMPT_INITIAL_EN
        if mode == final_mode.MODE:
            return THINK_TOOL_SYSTEM_PROMPT_FINAL_EN
        if self._is_missing_primary_page_evidence(previous_tool_call_results):
            return THINK_TOOL_SYSTEM_PROMPT_GATE_EN
        return self.system_prompt

    @staticmethod
    def _build_thought_log_entry(note: ThinkNote, coverage_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        tags = ["think"]
        coverage_score = coverage_snapshot.get("coverage_score")
        if isinstance(coverage_score, (int, float)) and coverage_score < 0.4:
            tags.append("coverage_gap")
        entry = {
            "plan_step": note.plan_step_id,
            "reasoning": note.reasoning,
            "reasoning_tags": tags,
            "confidence_delta": note.confidence_delta,
            "coverage_delta": note.coverage_delta,
            "latency_ms": coverage_snapshot.get("latency_ms", 0),
        }
        return entry

    @staticmethod
    def _validate_mode(payload: ThinkToolResponse, extra: Dict[str, Any]) -> None:
        mode = str((extra or {}).get("think_mode") or normal_mode.MODE).strip().lower()
        if mode == initial_mode.MODE:
            if payload.report_needed is None:
                raise ValueError("think_mode=initial requires report_needed")
            if payload.report_style is not None:
                style = str(payload.report_style or "").strip().lower()
                if style not in {"deepsearch", "research"}:
                    raise ValueError("think_mode=initial report_style must be deepsearch or research")
            return
        if mode == final_mode.MODE:
            if payload.is_final is not True:
                raise ValueError("think_mode=final requires is_final=true")
            if payload.tool_calls:
                raise ValueError("think_mode=final must not include tool_calls")
