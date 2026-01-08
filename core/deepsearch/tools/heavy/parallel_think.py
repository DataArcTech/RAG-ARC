"""Parallel thinking tool that explores multiple reasoning branches."""
import json
from typing import Any, Dict, List, Optional

from config.core.deepsearch.tool_defaults import (
    PARALLEL_THINK_DEFAULT_BRANCHES,
    PARALLEL_THINK_DEFAULT_CONFIDENCE_DELTA_PER_BRANCH,
    PARALLEL_THINK_DEFAULT_CONTEXT_PREVIEW_CHARS,
    PARALLEL_THINK_DEFAULT_CONTEXT_PREVIEW_ITEMS,
    PARALLEL_THINK_DEFAULT_COVERAGE_DELTA_PER_BRANCH,
    PARALLEL_THINK_DEFAULT_TEMPERATURE,
)
from encapsulation.data_model.deepsearch import ThinkNote

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, safe_json_loads
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER
from core.prompts.deepsearch import PARALLEL_THINK_SYSTEM_PROMPT
from core.deepsearch.utils.compression import compact_evidences, resolve_compaction_config, truncate_text


class ParallelThinkTool(GraphTool):
    """Runs multiple lightweight think passes to stabilise long reasoning chains."""

    descriptor = ToolDescriptor(
        name="graph.parallel_think",
        channel="graph",
        description="Spins up multiple `thought/action` branches to stress-test the current plan before running "
        "expensive graph traversals.",
        speed="slow",
        cost="high",
        strategy_tags=("parallel", "reflection", "llm", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_LLM),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.heavy.parallel_think",
        mcp_callable=True,
        example_args={
            "question": "Need more options",
            "plan_step": "plan_03",
            "context_evidences": [],
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        branches: int = PARALLEL_THINK_DEFAULT_BRANCHES,
        temperature: float = PARALLEL_THINK_DEFAULT_TEMPERATURE,
    ):
        self.llm_connector = llm_connector
        self.branches = branches
        self.temperature = temperature

    async def run(self, request: ToolRunRequest) -> ToolResult:
        reflections = await self._generate_reflections(request)
        summary = "Parallel think produced the following ideas:\n" + "\n".join(
            f"{idx+1}. {item['thought']}" for idx, item in enumerate(reflections)
        )
        note = ThinkNote(
            plan_step_id=request.plan_step,
            reasoning="Parallel think suggested next actions.",
            confidence_delta=float(PARALLEL_THINK_DEFAULT_CONFIDENCE_DELTA_PER_BRANCH) * len(reflections),
            coverage_delta=float(PARALLEL_THINK_DEFAULT_COVERAGE_DELTA_PER_BRANCH) * len(reflections),
            next_actions=[item["action"] for item in reflections if item.get("action")],
            metadata={"branches": reflections},
        )
        diagnostics = {
            "branches": len(reflections),
            "temperature": self.temperature,
        }
        diagnostics["thought_log"] = self._build_thought_log(reflections, request.plan_step)
        return ToolResult(summary=summary, diagnostics=diagnostics, think_notes=[note])

    async def _generate_reflections(self, request: ToolRunRequest) -> List[Dict[str, str]]:
        preview_items = int(PARALLEL_THINK_DEFAULT_CONTEXT_PREVIEW_ITEMS)
        preview_chars = int(PARALLEL_THINK_DEFAULT_CONTEXT_PREVIEW_CHARS)
        cfg = resolve_compaction_config(
            branch="tool_context",
            graph_context=request.graph_context,
            extra=(request.extra or {}),
            default_max_items=preview_items,
            default_max_chars=preview_chars,
            default_mode="truncate",
            default_retention="tail",
            env_max_items="DEEPSEARCH_TOOL_CONTEXT_MAX_EVIDENCES",
            env_max_chars="DEEPSEARCH_TOOL_CONTEXT_MAX_CHARS",
        )
        compacted, _meta = compact_evidences(
            request.context_evidences or [],
            cfg=cfg,
            question=request.question,
            extra=(request.extra or {}),
            include_triple_count=False,
        )
        payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_preview": [str(item.get("content") or "") for item in compacted],
            "branches": self.branches,
        }
        messages = [
            {
                "role": "system",
                "content": PARALLEL_THINK_SYSTEM_PROMPT,
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)

        data = safe_json_loads(response)
        if isinstance(data, dict):
            for key in ("branches", "reflections", "items", "results"):
                maybe = data.get(key)
                if isinstance(maybe, list):
                    data = maybe
                    break

        if isinstance(data, list):
            branches = [self._coerce_branch(item) for item in data if isinstance(item, dict)]
            if branches:
                return branches

        # Fallback: keep the tool contract stable (always return at least one branch).
        raw = str(response or "").strip()
        thought = raw.replace("\n", " ")
        thought = truncate_text(thought, max_chars=360)
        return [
            {
                "thought": thought or "Consider reviewing missing evidence and rerunning a focused probe.",
                "action": "tighten_query_and_probe",
            }
        ]

    @staticmethod
    def _coerce_branch(item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "thought": str(item.get("thought") or item.get("reasoning") or "consider previous evidence"),
            "action": str(item.get("action") or item.get("next_action") or "follow_up"),
        }

    @staticmethod
    def _build_thought_log(reflections: List[Dict[str, str]], plan_step: Optional[str]) -> List[Dict[str, Any]]:
        log: List[Dict[str, Any]] = []
        for idx, reflection in enumerate(reflections):
            log.append(
                {
                    "plan_step": plan_step,
                    "branch": idx,
                    "reasoning": reflection.get("thought"),
                    "next_action": reflection.get("action"),
                    "reasoning_tags": ["parallel_think"],
                }
            )
        return log
