"""Plan runtime for DeepSearch: generate graph-centric task lists and emit JSON artifacts for execution."""
import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from encapsulation.data_model.deepsearch import GraphQueryContext, PlanSpec
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.scope_provider import require_scope, scope_to_dict

from .generator import PlanGenerator, PlannerSettings
from core.prompts.deepsearch import GRAPH_PLANNER_SYSTEM_PROMPT_EN, GRAPH_PLANNER_USER_PROMPT_EN
from core.deepsearch.tooling import describe_available_tools
from core.deepsearch.tooling.registry import ToolHintRegistry
from core.deepsearch.trace import emit_trace, with_trace_protocol
from core.deepsearch.utils.language_policy import infer_user_language

from config.core.deepsearch.planner_web_policy_defaults import (
    DEFAULT_REALTIME_WEB_INTENT_KEYWORDS,
    DEFAULT_REALTIME_WEB_STRONG_KEYWORDS,
    DEFAULT_REALTIME_WEB_STEP_DESCRIPTION_EN,
    DEFAULT_REALTIME_WEB_STEP_DESCRIPTION_ZH,
    DEFAULT_REALTIME_WEB_TOPIC_KEYWORDS,
)
from config.benchmark_mode import benchmark_mode_enabled

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

    def __init__(
        self,
        prompt_store,
        llm_connector,
        config,
        *,
        plan_generator: PlanGenerator | None = None,
        tool_hint_registry: ToolHintRegistry | None = None,
    ):
        # prompt_store: repository for plan/question-decomposition/query-expansion templates
        self.prompt_store = prompt_store
        # llm_connector: encapsulation.llm-compatible client reused across core modules
        self.llm_connector = llm_connector
        # config: contains runtime knobs such as mode, max steps, output directories
        self.config = config
        self._config_dict = self._as_dict(config)
        self._tool_hint_registry = tool_hint_registry or ToolHintRegistry()
        self._available_tools: List[Dict[str, str]] = []
        self._tool_hint_revision: int = -1

        if "include_llm_tools_in_catalog" not in self._config_dict:
            raise ValueError("planner.include_llm_tools_in_catalog is required (no implicit default).")
        self.include_llm_tools_in_catalog = bool(self._config_dict["include_llm_tools_in_catalog"])

        settings = self._build_planner_settings()
        self.plan_generator = plan_generator or PlanGenerator(self.llm_connector, settings)
        self.plan_generator.settings.available_tools_hint = self._tool_hint_text()

        self.persist_plan = bool(self._config_dict["persist_plan"])
        plan_output_dir = self._config_dict.get("plan_output_dir")
        if not plan_output_dir or not str(plan_output_dir).strip():
            raise ValueError("planner.plan_output_dir is required (no implicit default).")
        self.plan_output_dir = Path(str(plan_output_dir))

        self.allow_external = bool(self._config_dict["allow_external_channel"])
        self.web_step_policy = str(self._config_dict.get("web_step_policy") or "off").strip().lower()
        def _coerce_str_list(value: Any) -> list[str]:
            raw = value or []
            if not isinstance(raw, (list, tuple, set)):
                raw = []
            return [str(item).strip() for item in raw if str(item).strip()]

        self.realtime_web_keywords = _coerce_str_list(self._config_dict.get("realtime_web_keywords"))
        self.realtime_web_strong_keywords = _coerce_str_list(self._config_dict.get("realtime_web_strong_keywords"))
        self.realtime_web_intent_keywords = _coerce_str_list(self._config_dict.get("realtime_web_intent_keywords"))
        self.realtime_web_topic_keywords = _coerce_str_list(self._config_dict.get("realtime_web_topic_keywords"))
        if not self.realtime_web_strong_keywords:
            self.realtime_web_strong_keywords = list(DEFAULT_REALTIME_WEB_STRONG_KEYWORDS)
        if not self.realtime_web_intent_keywords:
            self.realtime_web_intent_keywords = list(DEFAULT_REALTIME_WEB_INTENT_KEYWORDS)
        if not self.realtime_web_topic_keywords:
            self.realtime_web_topic_keywords = list(DEFAULT_REALTIME_WEB_TOPIC_KEYWORDS)
        self.realtime_web_force_external = bool(self._config_dict.get("realtime_web_force_external", True))

        self.graph_channel_tool = str(self._config_dict["graph_channel_tool"]).strip()
        self.text_channel_tool = str(self._config_dict["text_channel_tool"]).strip()
        self.web_channel_tool = str(self._config_dict["web_channel_tool"]).strip()
        self.default_web_provider = self._config_dict.get("default_web_provider")
        self.graph_adapter_name = str(self._config_dict["graph_adapter_name"]).strip()
        self.tool_arg_templates: Mapping[str, Mapping[str, str]] = self._config_dict.get("tool_arg_templates") or {}
        self.honor_planner_tool_selection = bool(self._config_dict["honor_planner_tool_selection"])

        self._refresh_available_tools(update_generator=True)

    async def build_plan(
        self,
        question: str,
        *,
        access_scope: GraphAccessScope | None = None,
    ) -> Dict[str, Any]:
        """Produce a structured plan containing graph-first steps plus optional external channels."""

        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must be a non-empty string")
        scope = require_scope(access_scope)

        await emit_trace(
            "think",
            "\n".join(
                [
                    "Planning the research workflow.",
                    f"mode={self.plan_generator.settings.mode}",
                    f"max_steps={self.plan_generator.settings.max_steps}",
                    f"external_channel_allowed={bool(self.allow_external)}",
                ]
            ),
            meta={
                "stage": "plan",
                "mode": self.plan_generator.settings.mode,
                "max_steps": self.plan_generator.settings.max_steps,
                "external_channel_allowed": bool(self.allow_external),
            },
        )

        self._refresh_available_tools()
        plan_specs = await self._generate_plan_async(normalized_question)
        plan_specs = self._apply_web_step_policy(question=normalized_question, plan_specs=plan_specs)
        steps_payload = [
            self._build_step_payload(spec)
            for spec in plan_specs
        ]

        plan_id = uuid.uuid4().hex
        graph_context_payload = self._build_graph_context_payload(
            access_scope=scope,
            question=normalized_question,
            plan_id=plan_id,
            steps=steps_payload,
        )

        artifact = with_trace_protocol(
            {
            "plan_id": plan_id,
            "question": normalized_question,
            "owner_scope": scope_to_dict(scope),
            "mode": self.plan_generator.settings.mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "max_steps": self.plan_generator.settings.max_steps,
                "allow_external_channel": self.allow_external,
                "persist_plan": self.persist_plan,
            },
            "steps": steps_payload,
            "available_tools": self.available_tools,
            "graph_context": graph_context_payload,
            }
        )

        artifact_path = None
        if self.persist_plan:
            artifact_path = self._persist_plan(artifact, plan_id)

        plan_lines: List[str] = []
        plan_lines.append(f"Plan ID: {plan_id}")
        plan_lines.append(f"Question: {normalized_question}")
        plan_lines.append(f"Mode: {self.plan_generator.settings.mode}")
        plan_lines.append(f"External allowed: {bool(self.allow_external)}")
        plan_lines.append("Note: coarse macro plan; tool selection happens during execution.")
        plan_lines.append("Steps:")
        for idx, step in enumerate(steps_payload, start=1):
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("step_id") or f"plan_{idx:02d}")
            channel = str(step.get("channel") or "graph")
            description = str(step.get("description") or "")
            enabled = bool(step.get("enabled", True))
            requires_external = bool(step.get("requires_external", False))
            line = f"{idx}. {step_id} [{channel}] enabled={enabled} requires_external={requires_external}"
            if description:
                line += f"\n   {description}"
            plan_lines.append(line)

        await emit_trace(
            "write_outline",
            "\n".join(plan_lines),
            meta={
                "stage": "plan",
                "plan_id": plan_id,
                "step_count": len(steps_payload),
                "artifact_path": str(artifact_path) if artifact_path else None,
            },
        )

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
        return self.plan_generator.generate_plan(question, context=context)

    def _build_planner_settings(self) -> PlannerSettings:
        mode = str(self._config_dict["mode"]).strip()
        max_steps = int(self._config_dict["max_steps"])
        enable_sub_question = bool(self._config_dict["enable_sub_question"])
        system_prompt = self._prompt(self.PLAN_SYSTEM_PROMPT_KEY, GRAPH_PLANNER_SYSTEM_PROMPT_EN)
        user_prompt = self._prompt(self.PLAN_USER_PROMPT_KEY, GRAPH_PLANNER_USER_PROMPT_EN)
        tool_hint = self._tool_hint_text()
        return PlannerSettings(
            mode=mode,
            max_steps=max_steps,
            enable_sub_question=enable_sub_question,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            available_tools_hint=tool_hint,
        )

    def _tool_hint_text(self, hints: Optional[List[Dict[str, str]]] = None) -> str:
        catalog = hints if hints is not None else self.available_tools
        if not catalog:
            return "No registered tools."
        profile_labels = {
            "F": "F-tools (fast deterministic probes; run first for cheap coverage)",
            "X": "X-tools (hybrid tools balancing deterministic + LLM summarisation)",
            "H": "H-tools (heavy LLM planners/think modules; expensive, run only when needed)",
        }
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for spec in catalog:
            profile = str(spec.get("profile") or "F").upper()
            grouped.setdefault(profile, []).append(spec)
        lines = [
            "Tool catalog (used during execution, not required in the initial plan).",
            "The initial plan should stay coarse; avoid selecting tools unless absolutely necessary.",
            "Profile legend: F=fast deterministic, X=hybrid, H=heavy LLM.",
        ]
        for profile_code in ("F", "X", "H"):
            specs = grouped.get(profile_code)
            if not specs:
                continue
            lines.append(f"{profile_labels.get(profile_code, profile_code + ' tools')}:")
            for spec in specs:
                name = spec.get("name", "unknown")
                channel = spec.get("channel", "")
                description = spec.get("description", "")
                determinism = spec.get("determinism")
                label = f"  - {name}"
                if channel:
                    label += f" [{channel}]"
                if determinism:
                    label += f" ({determinism})"
                if description:
                    label += f": {description}"
                lines.append(label)
        return "\n".join(lines)

    def _build_step_payload(self, spec: PlanSpec) -> Dict[str, Any]:
        metadata = dict(spec.metadata or {})
        if not spec.channel:
            raise ValueError("PlanSpec.channel is required")
        channel = str(spec.channel).lower()
        if channel not in {"graph", "web", "text"}:
            raise ValueError(f"Unsupported plan channel: {channel}")

        requested_tool = metadata.get("tool") if self.honor_planner_tool_selection else None
        if isinstance(requested_tool, str):
            requested_tool = requested_tool.strip()

        # Guardrails: keep macro plans coarse; prefer graph_adapter.query for graph steps unless the
        # planner explicitly labels a step as a probe. This avoids the LLM overusing scan tools as
        # primary execution steps.
        # Deprecated probe/scan tools removed; no special casing required here.
        if not self.honor_planner_tool_selection:
            metadata.pop("tool", None)
        tool_name = requested_tool or self._resolve_tool(channel)
        requires_external = channel == "web"
        tool_enabled = not requires_external or self.allow_external

        requested_tool_args = metadata.get("tool_args") if isinstance(metadata.get("tool_args"), dict) else None
        if requested_tool_args is not None:
            metadata.pop("tool_args", None)

        tool_args: Dict[str, Any]
        if tool_name == self.graph_channel_tool:
            tool_args = self._build_tool_args(
                channel=channel,
                description=spec.description,
                extra_metadata=metadata,
            )
            if requested_tool_args:
                tool_args.update(dict(requested_tool_args))
        else:
            # For graph/text tools, tool_args is passed as ToolRunRequest.extra.
            tool_args = dict(requested_tool_args or {})
            tool_args.setdefault("focus_query", spec.description.strip())

        if requires_external:
            metadata.setdefault("requires_external_channel", True)
            if not self.allow_external:
                metadata.setdefault("disabled_reason", "external_channel_disabled")
            # Ensure web.search always receives tool_args.query; ExternalSearchChannel requires it.
            query_value = tool_args.get("query")
            if not isinstance(query_value, str) or not query_value.strip():
                fallback = tool_args.get("focus_query") or spec.description
                tool_args["query"] = str(fallback or "").strip()
            if self.default_web_provider and not tool_args.get("provider"):
                tool_args["provider"] = self.default_web_provider

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

    def _apply_web_step_policy(self, *, question: str, plan_specs: List[PlanSpec]) -> List[PlanSpec]:
        if benchmark_mode_enabled():
            return plan_specs
        if not self.allow_external:
            return plan_specs
        policy = (self.web_step_policy or "off").strip().lower()
        if policy != "realtime_required":
            return plan_specs
        if not self._is_realtime_question(question):
            return plan_specs

        normalized = list(plan_specs or [])
        web_steps = [idx for idx, spec in enumerate(normalized) if str(getattr(spec, "channel", "")).strip().lower() == "web"]
        if web_steps:
            if self.realtime_web_force_external:
                idx = web_steps[0]
                spec = normalized[idx]
                meta = dict(spec.metadata or {})
                meta.setdefault("force_external", True)
                meta.setdefault("requires_external_reason", "realtime")
                tool_args = meta.get("tool_args") if isinstance(meta.get("tool_args"), dict) else {}
                tool_args = dict(tool_args or {})
                tool_args.setdefault("query", question.strip())
                meta["tool_args"] = tool_args
                normalized[idx] = spec.model_copy(update={"metadata": meta})
            return normalized

        lang = infer_user_language(question)
        desc = DEFAULT_REALTIME_WEB_STEP_DESCRIPTION_ZH if lang == "zh" else DEFAULT_REALTIME_WEB_STEP_DESCRIPTION_EN
        injected_meta: Dict[str, Any] = {"source": "policy_injected", "tool": self.web_channel_tool, "tool_args": {"query": question.strip()}}
        if self.realtime_web_force_external:
            injected_meta["force_external"] = True
            injected_meta["requires_external_reason"] = "realtime"
        injected = PlanSpec(
            step_id="policy_web_01",
            description=desc,
            channel="web",
            metadata=injected_meta,
        )

        # Prefer inserting before the final text synthesis step when present.
        insert_at = len(normalized)
        for idx, spec in enumerate(normalized):
            if str(getattr(spec, "channel", "")).strip().lower() == "text":
                insert_at = idx
                break
        normalized.insert(insert_at, injected)
        if len(normalized) > int(self.plan_generator.settings.max_steps):
            normalized = normalized[: int(self.plan_generator.settings.max_steps)]
        return normalized

    def _is_realtime_question(self, question: str) -> bool:
        text = (question or "").strip().lower()
        if not text:
            return False
        strong = self.realtime_web_strong_keywords or []
        for kw in strong:
            token = (kw or "").strip().lower()
            if token and token in text:
                return True

        intent = self.realtime_web_intent_keywords or []
        topic = self.realtime_web_topic_keywords or []
        if not intent or not topic:
            # Backward-compatible fallback: treat any keyword match as realtime intent.
            for kw in self.realtime_web_keywords:
                token = (kw or "").strip().lower()
                if token and token in text:
                    return True
            return False

        has_intent = any(((kw or "").strip().lower() in text) for kw in intent if (kw or "").strip())
        if not has_intent:
            return False
        return any(((kw or "").strip().lower() in text) for kw in topic if (kw or "").strip())

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
        extra_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_args = {"query": description.strip(), "channel": channel}

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
                json.dump(artifact, fp, indent=2, ensure_ascii=False)
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

    def _refresh_available_tools(self, update_generator: bool = True) -> None:
        """Refresh cached tool descriptors so planner sees MCP/runtime additions."""

        graph_channel_tool = str(getattr(self, "graph_channel_tool", None) or "graph_adapter.query").strip() or "graph_adapter.query"
        adapter_hint = {
            "name": graph_channel_tool,
            "channel": "graph",
            "description": "Primary graph traversal via the configured graph adapter (prepare→query→filter→summarize→chain_traverse).",
            "profile": "X",
            "determinism": "adapter",
            "strategy_tags": ["graph", "adapter", "traversal"],
        }
        hints = describe_available_tools(
            extra_hints=[],
            registry=self._tool_hint_registry,
            include_llm_tools=self.include_llm_tools_in_catalog,
        )

        # Optional: allowlist planner catalog to reduce cognitive load.
        allowlist = self._config_dict.get("tool_catalog_allowlist")
        allowed: set[str] | None = None
        if isinstance(allowlist, (list, tuple, set)):
            allowed = {str(name).strip() for name in allowlist if str(name).strip()}
            hints = [hint for hint in hints if str(hint.get("name") or "").strip() in allowed]

        # Ensure graph_adapter.query is present only when not explicitly excluded.
        if allowed is None or graph_channel_tool in allowed:
            hints = [hint for hint in hints if isinstance(hint, dict) and hint.get("name") != graph_channel_tool]
            hints.insert(0, adapter_hint)

        raw_limit = self._config_dict.get("tool_catalog_max_items")
        try:
            limit = int(raw_limit) if raw_limit is not None else 0
        except (TypeError, ValueError):
            limit = 0
        if limit > 0 and len(hints) > limit:
            hints = hints[:limit]

        self._available_tools = hints
        self._tool_hint_revision = self._tool_hint_registry.get_revision()
        if update_generator and getattr(self, "plan_generator", None):
            self.plan_generator.settings.available_tools_hint = self._tool_hint_text(self._available_tools)

    @property
    def available_tools(self) -> List[Dict[str, str]]:
        """Expose cached tool descriptors; refresh lazily when hints change."""

        current_revision = self._tool_hint_registry.get_revision()
        if current_revision != self._tool_hint_revision:
            self._refresh_available_tools()
        return self._available_tools

    def _build_graph_context_payload(
        self,
        *,
        access_scope: GraphAccessScope,
        question: str,
        plan_id: str,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        seed_entities = self._collect_seed_entities(steps)
        metadata = {
            "plan_id": plan_id,
            "planner_mode": self.plan_generator.settings.mode,
        }
        context = GraphQueryContext(
            adapter_name=self.graph_adapter_name,
            question=question,
            seed_entities=seed_entities,
            metadata=metadata,
            access_scope=access_scope,
        )
        return context.model_dump(exclude_none=True)

    @staticmethod
    def _collect_seed_entities(steps: List[Dict[str, Any]]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []

        def _add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _add(item)
                return
            token = str(value).strip()
            if not token or token in seen:
                return
            seen.add(token)
            ordered.append(token)

        for step in steps:
            metadata = step.get("metadata") or {}
            tool_args = step.get("tool_args") or {}
            for candidate in (
                metadata.get("seed_entities"),
                tool_args.get("seed_entities"),
                metadata.get("seed_nodes"),
                tool_args.get("seed_nodes"),
            ):
                _add(candidate)
        return ordered


class PlanStep:
    """Lightweight container representing a single plan node for execution tracking."""

    def __init__(self, description: str, channel: str, metadata: Optional[Dict[str, Any]] = None):
        self.description = description
        self.channel = channel
        self.metadata = metadata or {}
