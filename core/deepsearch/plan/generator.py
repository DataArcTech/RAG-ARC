"""Plan generation utilities for DeepSearch pipelines."""
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from encapsulation.data_model.deepsearch import GraphQueryContext, PlanSpec
from core.prompts.deepsearch import GRAPH_PLANNER_SYSTEM_PROMPT, GRAPH_PLANNER_USER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PlannerSettings:
    """Runtime knobs that influence plan generation."""

    mode: str = "react"
    max_steps: int = 6
    enable_sub_question: bool = True
    system_prompt: str = GRAPH_PLANNER_SYSTEM_PROMPT
    user_prompt_template: str = GRAPH_PLANNER_USER_PROMPT
    available_tools_hint: str = ""


class PlanGenerator:
    """Generate PlanSpec sequences using either an LLM or heuristics."""

    def __init__(self, llm=None, settings: Optional[PlannerSettings] = None):
        self.llm = llm
        self.settings = settings or PlannerSettings()

    def generate_plan(
        self,
        question: str,
        *,
        context: Optional[GraphQueryContext] = None,
    ) -> List[PlanSpec]:
        """Produce a list of PlanSpec objects ranked by execution order."""

        if not question:
            raise ValueError("question must be a non-empty string")

        raw_steps = self._call_llm(question, context=context)
        if not raw_steps:
            raw_steps = self._fallback_plan(question)

        return self._build_plan_specs(raw_steps)

    async def agenerate_plan(
        self,
        question: str,
        *,
        context: Optional[GraphQueryContext] = None,
    ) -> List[PlanSpec]:
        """Async wrapper to keep FastAPI/CLI event loops responsive."""

        if not question:
            raise ValueError("question must be a non-empty string")

        raw_steps = await self._call_llm_async(question, context=context)
        if not raw_steps:
            raw_steps = self._fallback_plan(question)

        return self._build_plan_specs(raw_steps)

    # internal helpers -----------------------------------------------------

    def _build_plan_specs(self, raw_steps: List[dict]) -> List[PlanSpec]:
        plan_specs: List[PlanSpec] = []
        for idx, step in enumerate(raw_steps[: self.settings.max_steps]):
            description = step.get("description") or step.get("step") or "Graph inspect"
            channel = self._normalize_channel(step.get("channel"))
            plan_specs.append(
                PlanSpec(
                    step_id=f"plan_{idx+1:02d}",
                    description=description.strip(),
                    channel=channel,
                    metadata={
                        "mode": self.settings.mode,
                        "source": step.get("source", "llm" if self.llm else "rule"),
                    },
                )
            )
        return plan_specs

    def _build_messages(
        self,
        question: str,
        context: Optional[GraphQueryContext],
    ) -> List[Dict[str, str]]:
        context_hint = ""
        if context and context.seed_entities:
            context_hint = "\nKnown seed entities: " + ", ".join(context.seed_entities) + "."
        question_payload = f"{question}{context_hint}"
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {
                "role": "user",
                "content": self.settings.user_prompt_template.format(
                    question=question_payload,
                    mode=self.settings.mode,
                    max_steps=self.settings.max_steps,
                    available_tools=self.settings.available_tools_hint,
                ),
            },
        ]

    def _call_llm(
        self,
        question: str,
        *,
        context: Optional[GraphQueryContext],
    ) -> List[dict]:
        if self.llm is None:
            return []
        messages = self._build_messages(question, context)
        try:
            response = self.llm.chat(messages, temperature=0.1)
            return self._parse_llm_response(response)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("LLM planning failed: %s", exc)
            return []

    async def _call_llm_async(
        self,
        question: str,
        *,
        context: Optional[GraphQueryContext],
    ) -> List[dict]:
        if self.llm is None:
            return []

        messages = self._build_messages(question, context)
        async_chat = getattr(self.llm, "achat", None)
        if callable(async_chat):
            try:
                response = await async_chat(messages, temperature=0.1)
                return self._parse_llm_response(response)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Async LLM planning failed: %s", exc)
                return []

        return await asyncio.to_thread(self._call_llm, question, context=context)

    def _parse_llm_response(self, response: str) -> List[dict]:
        response = self._extract_json_payload(response)
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [step for step in data if isinstance(step, dict)]
        except json.JSONDecodeError:
            logger.warning("Planner LLM returned non-JSON response, switching to rule-based plan")
        return []

    def _extract_json_payload(self, payload: str) -> str:
        """Handle ```json fenced blocks or extra commentary around JSON."""
        text = (payload or "").strip()
        if not text:
            return ""

        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()

        start = text.find("[")
        end = text.rfind("]")
        if 0 <= start < end:
            return text[start : end + 1]

        return text

    def _fallback_plan(self, question: str) -> List[dict]:
        clauses = re.split(r"[?。.!]", question)
        sub_questions = [cl for cl in clauses if cl.strip()]
        if not sub_questions:
            sub_questions = [question]
        steps: List[dict] = []
        for clause in sub_questions:
            description = clause.strip()
            if not description:
                continue
            channel = "graph"
            if "web" in description.lower() or "search" in description.lower():
                channel = "web"
            steps.append({"description": description, "channel": channel, "source": "rule"})
        if self.settings.enable_sub_question and len(steps) == 1:
            steps.append({
                "description": "Validate graph findings and propose next-hop queries",
                "channel": "graph",
                "source": "rule",
            })
        return steps

    @staticmethod
    def _normalize_channel(channel: Optional[str]) -> str:
        value = (channel or "graph").strip().lower()
        if value not in {"graph", "web", "text"}:
            return "graph"
        return value
