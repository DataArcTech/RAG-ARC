"""Config for DeepSearch planner component."""
import os
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from core.deepsearch.plan import PlanGenerator, PlannerSettings
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

    _VALID_MODES = {"react", "iter_research", "parallel_thinking"}

    def build(self) -> PlanGenerator:
        llm_config = self._build_llm_config_from_env()
        llm = llm_config.build() if llm_config else None

        mode = self._resolve_mode()
        max_steps = self._env_int("DEEPSEARCH_PLANNER_MAX_STEPS", self.max_steps)
        enable_sub_question = self._env_bool(
            "DEEPSEARCH_PLANNER_ENABLE_SUBQUESTION", self.enable_sub_question
        )

        settings = PlannerSettings(
            mode=mode,
            max_steps=max_steps,
            enable_sub_question=enable_sub_question,
        )
        return PlanGenerator(llm=llm, settings=settings)

    def _resolve_mode(self) -> str:
        raw = os.getenv("DEEPSEARCH_PLANNER_MODE")
        if raw:
            candidate = raw.strip().lower()
            if candidate in self._VALID_MODES:
                return candidate
        return self.mode

    def _build_llm_config_from_env(self) -> Optional[OpenAIChatConfig]:
        if self._env_bool("DEEPSEARCH_PLANNER_DISABLE_LLM", False):
            return None

        overrides: Dict[str, Any] = {}
        provider = os.getenv("DEEPSEARCH_PLANNER_LLM_PROVIDER")
        if provider:
            candidate = provider.strip().lower()
            if candidate in {"openai", "huggingface"}:
                overrides["loading_method"] = candidate

        model_name = self._env_str("DEEPSEARCH_PLANNER_MODEL_NAME")
        if model_name:
            overrides["model_name"] = model_name

        max_tokens = self._env_int("DEEPSEARCH_PLANNER_MAX_TOKENS", None)
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens

        temperature = self._env_float("DEEPSEARCH_PLANNER_TEMPERATURE", None)
        if temperature is not None:
            overrides["temperature"] = temperature

        api_key = os.getenv("DEEPSEARCH_PLANNER_API_KEY")
        if api_key:
            overrides["openai_api_key"] = api_key.strip()

        base_url = self._env_str("DEEPSEARCH_PLANNER_BASE_URL")
        if base_url:
            overrides["openai_base_url"] = base_url

        organization = self._env_str("DEEPSEARCH_PLANNER_ORGANIZATION")
        if organization:
            overrides["organization"] = organization

        timeout = self._env_float("DEEPSEARCH_PLANNER_TIMEOUT", None)
        if timeout is not None:
            overrides["timeout"] = timeout

        max_retries = self._env_int("DEEPSEARCH_PLANNER_MAX_RETRIES", None)
        if max_retries is not None:
            overrides["max_retries"] = max_retries

        if not overrides:
            return self.llm_config

        base_config = self.llm_config or OpenAIChatConfig()
        return base_config.model_copy(update=overrides)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: Optional[int]) -> Optional[int]:
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: Optional[float]) -> Optional[float]:
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _env_str(name: str) -> Optional[str]:
        value = os.getenv(name)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
