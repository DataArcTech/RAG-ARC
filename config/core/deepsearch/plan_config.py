"""Config for DeepSearch planner component."""
from typing import Literal, Optional

from pydantic import Field

from core.deepsearch.plan import PlanGenerator, PlannerSettings
from core.prompts.deepsearch import GRAPH_PLANNER_SYSTEM_PROMPT, GRAPH_PLANNER_USER_PROMPT
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from framework.config import AbstractConfig


class DeepSearchPlannerConfig(AbstractConfig):
    """Builds PlanGenerator with graph-first defaults."""

    type: Literal["deepsearch_planner"] = "deepsearch_planner"
    llm_config: Optional[OpenAIChatConfig] = Field(
        default_factory=OpenAIChatConfig,
        description="Optional LLM config used for plan generation (defaults to chat provider)",
    )
    mode: Literal["react", "iter_research", "parallel_thinking"] = Field(
        "react", description="Planner runtime mode"
    )
    max_steps: int = Field(6, description="Maximum number of plan steps to emit")
    enable_sub_question: bool = Field(True, description="Enable heuristic sub-question expansion")

    def build(self) -> PlanGenerator:
        if self.llm_config is None:
            raise ValueError("DeepSearchPlannerConfig.llm_config is required")
        llm = self.llm_config.build()
        settings = PlannerSettings(
            mode=self.mode,
            max_steps=int(self.max_steps),
            enable_sub_question=bool(self.enable_sub_question),
            system_prompt=GRAPH_PLANNER_SYSTEM_PROMPT,
            user_prompt_template=GRAPH_PLANNER_USER_PROMPT,
            available_tools_hint="",
        )
        return PlanGenerator(llm, settings)
