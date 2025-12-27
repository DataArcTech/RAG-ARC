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
from core.prompts.deepsearch import PARALLEL_THINK_SYSTEM_PROMPT


class ParallelThinkTool(GraphTool):
    """Runs multiple lightweight think passes to stabilise long reasoning chains."""

    descriptor = ToolDescriptor(
        name="graph.parallel_think",
        channel="graph",
        description="Spins up multiple `thought/action` branches to stress-test the current plan before running "
        "expensive graph traversals.",
        speed="slow",
        cost="high",
        strategy_tags=("parallel", "reflection", "llm"),
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
        payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_preview": [ev.content[:preview_chars] for ev in request.context_evidences[-preview_items:]],
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
        data = safe_json_loads(response, expected="list")
        if not isinstance(data, list):
            raise ValueError("ParallelThinkTool returned non-JSON or non-list payload")
        branches = [self._coerce_branch(item) for item in data if isinstance(item, dict)]
        if not branches:
            raise ValueError("ParallelThinkTool returned an empty list of branches")
        return branches

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
