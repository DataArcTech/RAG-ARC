"""Plan runtime for DeepSearch: generate graph-centric task lists and emit JSON artifacts for execution."""
import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from encapsulation.data_model.deepsearch import PlanSpec

from .generator import PlanGenerator, PlannerSettings
from core.prompts.deepsearch import GRAPH_PLANNER_SYSTEM_PROMPT, GRAPH_PLANNER_USER_PROMPT
from core.deepsearch.tooling import describe_available_tools

logger = logging.getLogger(__name__)


class DeepSearchPlanner:
    """Orchestrate problem analysis, task decomposition, and channel selection for DeepSearch plans.

    Goals for this stage:
    1. Pull prompt templates from the shared prompt_store rather than hard-coding strings.
    2. Honor configuration/.env flags that gate web/external channels.
    3. Emit JSON artifacts annotated with MCP tool metadata so downstream loops can replay steps.
    """

    PLAN_SYSTEM_PROMPT_KEY = "deepsearch.plan.system"
    PLAN_USER_PROMPT_KEY = "deepsearch.plan.user"

    def __init__(self, prompt_store, llm_connector, config, *, plan_generator: PlanGenerator | None = None):
        # prompt_store: repository for plan/question-decomposition/query-expansion templates
        self.prompt_store = prompt_store
        # llm_connector: encapsulation.llm-compatible client reused across core modules
        self.llm_connector = llm_connector
        # config: contains runtime knobs such as mode, max steps, output directories
        self.config = config
        self._config_dict = self._as_dict(config)
        self.available_tools = describe_available_tools()

        settings = self._build_planner_settings()
        self.plan_generator = plan_generator or PlanGenerator(llm=self.llm_connector, settings=settings)

        self.persist_plan = self._bool_config(
            "persist_plan",
            default=True,
            env_var="DEEPSEARCH_PERSIST_PLAN",
        )
        default_output_dir = "./local/deepsearch_runs"
        self.plan_output_dir = Path(
            self._config_value("plan_output_dir", os.getenv("DEEPSEARCH_PLAN_OUTPUT_DIR") or default_output_dir)
        )
        self.allow_external = self._bool_config(
            "allow_external_channel",
            default=False,
            env_var="DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL",
        )

        self.graph_channel_tool = self._config_value("graph_channel_tool", "graph_adapter.query")
        self.text_channel_tool = self._config_value("text_channel_tool", "llm.summarize")
        self.web_channel_tool = self._config_value("web_channel_tool", "web.search")
        self.default_web_provider = self._config_value("default_web_provider", os.getenv("DEEPSEARCH_WEB_PROVIDER"))
        self.graph_adapter_name = self._config_value(
            "graph_adapter_name", os.getenv("DEEPSEARCH_DEFAULT_ADAPTER") or "hipporag"
        )

        self.tool_arg_templates: Mapping[str, Mapping[str, str]] = self._config_value("tool_arg_templates", {})

    async def build_plan(self, question: str, *, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Produce a structured plan containing graph-first steps plus optional external channels."""

        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must be a non-empty string")

        plan_specs = await self._generate_plan_async(normalized_question)
        steps_payload = [
            self._build_step_payload(spec, owner_id=owner_id)
            for spec in plan_specs
        ]

        plan_id = uuid.uuid4().hex
        artifact = {
            "plan_id": plan_id,
            "question": normalized_question,
            "owner_id": str(owner_id) if owner_id else None,
            "mode": self.plan_generator.settings.mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "max_steps": self.plan_generator.settings.max_steps,
                "allow_external_channel": self.allow_external,
                "persist_plan": self.persist_plan,
            },
            "steps": steps_payload,
            "available_tools": self.available_tools,
        }

        artifact_path = None
        if self.persist_plan:
            artifact_path = self._persist_plan(artifact, plan_id)

        logger.info(
            "DeepSearch plan generated (plan_id=%s, steps=%d, artifact=%s)",
            plan_id,
            len(steps_payload),
            artifact_path,
        )

        return {
            "plan_id": plan_id,
            "plan": artifact,
            "artifact_path": str(artifact_path) if artifact_path else None,
        }

    # internal helpers -------------------------------------------------

    async def _generate_plan_async(
        self,
        question: str,
        *,
        context: Optional[Any] = None,
    ) -> List[PlanSpec]:
        agen = getattr(self.plan_generator, "agenerate_plan", None)
        if callable(agen):
            return await agen(question, context=context)
        return await asyncio.to_thread(self.plan_generator.generate_plan, question, context=context)

    def _build_planner_settings(self) -> PlannerSettings:
        mode = self._config_value("mode", "react")
        max_steps = int(self._config_value("max_steps", 6))
        enable_sub_question = self._bool_config("enable_sub_question", True)
        system_prompt = self._prompt(self.PLAN_SYSTEM_PROMPT_KEY, GRAPH_PLANNER_SYSTEM_PROMPT)
        user_prompt = self._prompt(self.PLAN_USER_PROMPT_KEY, GRAPH_PLANNER_USER_PROMPT)
        tool_hint = self._tool_hint_text()
        return PlannerSettings(
            mode=mode,
            max_steps=max_steps,
            enable_sub_question=enable_sub_question,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            available_tools_hint=tool_hint,
        )

    def _tool_hint_text(self) -> str:
        if not self.available_tools:
            return "No registered tools."
        lines = []
        for spec in self.available_tools:
            name = spec.get("name", "unknown")
            channel = spec.get("channel", "")
            description = spec.get("description", "")
            label = f"- {name}"
            if channel:
                label += f" [{channel}]"
            if description:
                label += f": {description}"
            lines.append(label)
        return "\n".join(lines)

    def _build_step_payload(self, spec: PlanSpec, *, owner_id: Optional[str]) -> Dict[str, Any]:
        metadata = dict(spec.metadata or {})
        channel = (spec.channel or "graph").lower()
        channel = channel if channel in {"graph", "web", "text"} else "graph"

        requested_tool = metadata.get("tool")
        tool_name = requested_tool or self._resolve_tool(channel)
        requires_external = channel == "web"
        tool_enabled = not requires_external or self.allow_external

        tool_args = self._build_tool_args(
            channel=channel,
            description=spec.description,
            owner_id=owner_id,
            extra_metadata=metadata,
        )

        if requires_external:
            metadata.setdefault("requires_external_channel", True)
            if not self.allow_external:
                metadata.setdefault("disabled_reason", "external_channel_disabled")

        metadata["tool"] = tool_name

        return {
            "step_id": spec.step_id,
            "description": spec.description,
            "channel": channel,
            "metadata": metadata,
            "tool": tool_name,
            "tool_args": tool_args,
            "requires_external": requires_external,
            "enabled": tool_enabled,
        }

    def _resolve_tool(self, channel: str) -> str:
        if channel == "web":
            return self.web_channel_tool
        if channel == "text":
            return self.text_channel_tool
        return self.graph_channel_tool

    def _build_tool_args(
        self,
        *,
        channel: str,
        description: str,
        owner_id: Optional[str],
        extra_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_args = {"query": description.strip(), "channel": channel}
        if owner_id:
            base_args["owner_id"] = str(owner_id)

        if channel == "graph":
            base_args.setdefault("adapter_name", extra_metadata.get("adapter") or self.graph_adapter_name)
        elif channel == "web":
            provider = extra_metadata.get("provider") or self.default_web_provider
            if provider:
                base_args.setdefault("provider", provider)

        template_args = self.tool_arg_templates.get(channel)
        if template_args:
            formatted = {}
            for key, template in template_args.items():
                formatted[key] = template.format(
                    description=description,
                    owner_id=str(owner_id) if owner_id else "",
                    channel=channel,
                )
            base_args.update(formatted)
        return base_args

    def _persist_plan(self, artifact: Dict[str, Any], plan_id: str) -> Optional[Path]:
        plan_dir = self.plan_output_dir
        try:
            plan_dir.mkdir(parents=True, exist_ok=True)
            path = plan_dir / f"{plan_id}_plan.json"
            with path.open("w", encoding="utf-8") as fp:
                json.dump(artifact, fp, indent=2)
            return path
        except OSError as exc:
            logger.warning("Failed to persist DeepSearch plan %s: %s", plan_id, exc)
            return None

    def _prompt(self, key: str, default: str) -> str:
        if not self.prompt_store:
            return default
        getter = getattr(self.prompt_store, "get", None)
        if callable(getter):
            try:
                value = getter(key)
            except TypeError:
                value = getter(key, None)
            if value:
                return value
        prompt_getter = getattr(self.prompt_store, "get_prompt", None)
        if callable(prompt_getter):
            value = prompt_getter(key)
            if value:
                return value
        if isinstance(self.prompt_store, Mapping):
            value = self.prompt_store.get(key)
            if value:
                return value
        return default

    def _bool_config(self, key: str, default: bool, env_var: Optional[str] = None) -> bool:
        env_override = self._resolve_env_bool(env_var) if env_var else None
        if env_override is not None:
            return env_override
        raw = self._config_value(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    @staticmethod
    def _resolve_env_bool(env_var: Optional[str]) -> Optional[bool]:
        if not env_var:
            return None
        value = os.getenv(env_var)
        if value is None:
            return None
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _config_value(self, key: str, default: Any = None) -> Any:
        if isinstance(self._config_dict, Mapping):
            return self._config_dict.get(key, default)
        return getattr(self.config, key, default) if self.config else default

    @staticmethod
    def _as_dict(config: Any) -> Dict[str, Any]:
        if not config:
            return {}
        if hasattr(config, "model_dump"):
            return config.model_dump()
        if isinstance(config, Mapping):
            return dict(config)
        if hasattr(config, "__dict__"):
            return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        try:
            return asdict(config)
        except Exception:
            return {}


class PlanStep:
    """Lightweight container representing a single plan node for execution tracking."""

    def __init__(self, description: str, channel: str, metadata: Optional[Dict[str, Any]] = None):
        self.description = description
        self.channel = channel
        self.metadata = metadata or {}
