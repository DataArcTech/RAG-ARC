"""Parallel thinking tool that explores multiple reasoning branches."""
import json
from typing import Any, Dict, List, Optional

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

    def __init__(self, llm_connector, *, branches: int = 3, temperature: float = 0.4):
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
            confidence_delta=0.1 * len(reflections),
            coverage_delta=0.05 * len(reflections),
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
        payload = {
            "question": request.question,
            "plan_step": request.plan_step,
            "context_preview": [ev.content[:200] for ev in request.context_evidences[-3:]],
            "branches": self.branches,
        }
        messages = [
            {
                "role": "system",
                "content": PARALLEL_THINK_SYSTEM_PROMPT,
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
            data = safe_json_loads(response, expected="list")
            if isinstance(data, list):
                return [self._coerce_branch(item) for item in data if isinstance(item, dict)]
        except Exception:
            pass
        return [
            {"thought": f"Review context snippet {idx+1}", "action": "rerun_fast_probe"}
            for idx in range(self.branches)
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
