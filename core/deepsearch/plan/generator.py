"""Plan generation utilities for DeepSearch pipelines."""
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from encapsulation.data_model.deepsearch import GraphQueryContext, PlanSpec
from core.utils.json_extract import extract_last_json_array_from_text

logger = logging.getLogger(__name__)


@dataclass
class PlannerSettings:
    """Runtime knobs that influence plan generation."""

    mode: str
    max_steps: int
    enable_sub_question: bool
    system_prompt: str
    user_prompt_template: str
    available_tools_hint: str


class PlanGenerator:
    """Generate PlanSpec sequences using an LLM."""

    def __init__(self, llm, settings: PlannerSettings):
        self.llm = llm
        self.settings = settings

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
            raise RuntimeError("Planner did not return a usable JSON plan")

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
            raise RuntimeError("Planner did not return a usable JSON plan")

        return self._build_plan_specs(raw_steps)

    # internal helpers -----------------------------------------------------

    def _build_plan_specs(self, raw_steps: List[dict]) -> List[PlanSpec]:
        plan_specs: List[PlanSpec] = []
        for idx, step in enumerate(raw_steps):
            if idx >= self.settings.max_steps:
                break
            description = str(step.get("description") or step.get("step") or "").strip()
            if not description:
                raise ValueError("Planner step is missing required 'description'")
            channel = self._normalize_channel(step.get("channel"))
            metadata = {
                "mode": self.settings.mode,
                "source": step.get("source", "llm"),
            }
            selected_tool = (step.get("tool") or "").strip()
            if selected_tool:
                metadata["tool"] = selected_tool
            tool_args = step.get("tool_args")
            if isinstance(tool_args, dict) and tool_args:
                metadata["tool_args"] = tool_args
            tool_profile = (step.get("tool_profile") or step.get("profile") or "").strip()
            if tool_profile:
                metadata["tool_profile"] = tool_profile.upper()
            determinism = (step.get("determinism") or "").strip()
            if determinism:
                metadata["tool_determinism"] = determinism
            step_metadata = step.get("metadata")
            if isinstance(step_metadata, dict) and step_metadata:
                # Allow LLM to pass through additional executor hints (e.g. parallelizable flags).
                for key, value in step_metadata.items():
                    if key in {"tool", "tool_args"}:
                        continue
                    metadata.setdefault(key, value)
            plan_specs.append(
                PlanSpec(
                    step_id=f"plan_{idx+1:02d}",
                    description=description.strip(),
                    channel=channel,
                    metadata=metadata,
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
            raise RuntimeError("PlanGenerator requires an LLM connector")
        messages = self._build_messages(question, context)
        try:
            response = self.llm.chat(messages, temperature=0.1)
            return self._parse_llm_response(response)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"LLM planning failed: {exc}") from exc

    async def _call_llm_async(
        self,
        question: str,
        *,
        context: Optional[GraphQueryContext],
    ) -> List[dict]:
        if self.llm is None:
            raise RuntimeError("PlanGenerator requires an LLM connector")

        messages = self._build_messages(question, context)
        async_chat = getattr(self.llm, "achat", None)
        if callable(async_chat):
            response = await async_chat(messages, temperature=0.1)
            return self._parse_llm_response(response)

        raise RuntimeError("PlanGenerator requires an async-capable LLM connector (missing `achat`).")

    def _parse_llm_response(self, response: str) -> List[dict]:
        extracted = extract_last_json_array_from_text(response)
        response = extracted if extracted is not None else (response or "").strip()
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [step for step in data if isinstance(step, dict)]
        except json.JSONDecodeError:
            raise ValueError("Planner LLM returned non-JSON output")
        raise ValueError("Planner LLM returned an unsupported JSON payload")

    @staticmethod
    def _normalize_channel(channel: Optional[str]) -> str:
        if channel is None:
            raise ValueError("Planner step is missing required 'channel'")
        value = str(channel).strip().lower()
        if value not in {"graph", "web", "text"}:
            raise ValueError(f"Planner step has unsupported channel: {value}")
        return value
