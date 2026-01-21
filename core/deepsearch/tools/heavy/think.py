"""Think tool that offers structured reasoning pauses."""
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config.core.deepsearch import tool_defaults
from encapsulation.data_model.deepsearch import ThinkNote, GraphQueryContext
from core.prompts.deepsearch.report import JSON_REPAIR_USER_PROMPT_EN
from core.prompts.deepsearch import THINK_TOOL_SYSTEM_PROMPT_EN
from core.deepsearch.utils.compression import compact_evidences, resolve_compaction_config

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, safe_json_loads
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER


class ThinkToolCall(BaseModel):
    tool_name: str = Field(..., min_length=1)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)
    parallelizable: bool = Field(...)


class ThinkToolResponse(BaseModel):
    reasoning: str = Field(..., min_length=1)
    tool_calls: List[ThinkToolCall] = Field(default_factory=list)


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
        namespace="rag-arc.deepsearch.tools.heavy.think",
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
        json_repair_attempts: int = tool_defaults.THINK_JSON_REPAIR_DEFAULT_ATTEMPTS,
        json_repair_temperature: float = tool_defaults.THINK_JSON_REPAIR_DEFAULT_TEMPERATURE,
        json_repair_max_raw_chars: int = tool_defaults.THINK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS,
    ):
        self.llm_connector = llm_connector
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.json_repair_temperature = float(json_repair_temperature)
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
        if isinstance(note.metadata, dict) and isinstance(note.metadata.get("compression"), dict):
            diagnostics["compression"] = note.metadata.get("compression")
        diagnostics["thought_log"] = [
            self._build_thought_log_entry(
                note,
                coverage_snapshot=coverage_snapshot,
            )
        ]
        summary = note.reasoning
        return ToolResult(summary=summary, diagnostics=diagnostics, think_notes=[note])

    async def _build_note(self, request: ToolRunRequest) -> ThinkNote:
        context_snapshot = self._graph_context_snapshot(request.graph_context)
        coverage_snapshot = request.coverage_metrics or {}
        if not self.llm_connector:
            raise RuntimeError("ThinkTool requires an LLM connector")

        cfg = resolve_compaction_config(
            branch="think",
            graph_context=request.graph_context,
            extra=(request.extra or {}),
            default_max_items=8,
            default_max_chars=1600,
            default_mode="truncate",
            default_excerpt_chars=900,
            default_retention="head",
            env_max_items="DEEPSEARCH_THINK_MAX_EVIDENCES",
            env_max_chars="DEEPSEARCH_THINK_EVIDENCE_MAX_CHARS",
            env_excerpt_chars="DEEPSEARCH_THINK_EVIDENCE_EXCERPT_CHARS",
        )
        compacted, compaction_meta = compact_evidences(
            request.context_evidences or [],
            cfg=cfg,
            question=request.question,
            extra=(request.extra or {}),
            include_triple_count=True,
        )
        extra = request.extra or {}
        prompt_payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_evidences": compacted,
            "graph_context": context_snapshot,
            "tool_budget": (context_snapshot.get("metadata") or {}).get("tool_budget") if isinstance(context_snapshot, dict) else None,
            "coverage_metrics": coverage_snapshot,
            "available_tools": extra.get("available_tools"),
            "previous_tool_call_results": extra.get("previous_tool_call_results"),
            "recent_tool_runs": extra.get("recent_tool_runs"),
            "extra": extra,
            "compression": compaction_meta,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._serialize_payload(prompt_payload),
            },
        ]
        try:
            response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
            parsed = await self._parse_or_repair_json(messages=messages, raw=response)
            payload = ThinkToolResponse.model_validate(parsed)
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
                    "tool_calls": [call.model_dump() for call in payload.tool_calls],
                    "compression": compaction_meta,
                },
            )
        except Exception as exc:
            raise RuntimeError(f"ThinkTool failed: {exc}") from exc

    async def _parse_or_repair_json(self, *, messages: List[Dict[str, str]], raw: str) -> Dict[str, Any]:
        parsed = safe_json_loads(raw, expected="dict")
        if isinstance(parsed, dict):
            return parsed
        repaired = await self._attempt_json_repair(messages=messages, raw=raw, expected="dict")
        if isinstance(repaired, dict):
            return repaired
        snippet = (raw or "").strip().replace("\n", "\\n")
        if len(snippet) > self.json_repair_max_raw_chars:
            snippet = snippet[: self.json_repair_max_raw_chars] + "…"
        raise ValueError(f"Think tool returned non-JSON or non-dict payload. raw_snippet={snippet}")

    async def _attempt_json_repair(self, *, messages: List[Dict[str, str]], raw: str, expected: str) -> Any:
        if self.json_repair_attempts <= 0:
            return None
        snippet = (raw or "").strip()
        if len(snippet) > self.json_repair_max_raw_chars:
            snippet = snippet[: self.json_repair_max_raw_chars] + "…"
        expected_label = "object" if expected == "dict" else ("array" if expected == "list" else expected)
        repair_prompt = JSON_REPAIR_USER_PROMPT_EN.format(
            expected_top_level=expected_label,
            error="invalid_json",
            raw_snippet=snippet,
        )
        thread = messages + [{"role": "assistant", "content": str(raw or "")}, {"role": "user", "content": repair_prompt}]
        last_raw = raw
        for _attempt in range(self.json_repair_attempts):
            last_raw = await call_llm_async(self.llm_connector, thread, temperature=self.json_repair_temperature)
            parsed = safe_json_loads(last_raw, expected=expected)
            if parsed is not None:
                return parsed
            thread = messages + [{"role": "assistant", "content": str(last_raw or "")}, {"role": "user", "content": repair_prompt}]
        return None

    @staticmethod
    def _graph_context_snapshot(graph_context: GraphQueryContext | None) -> Dict[str, Any]:
        if not graph_context:
            return {}
        return graph_context.model_dump(exclude_none=True)

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> str:
        def _default(value):
            return str(value)

        return json.dumps(payload, ensure_ascii=False, default=_default)

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
