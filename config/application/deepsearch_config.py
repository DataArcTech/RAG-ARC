"""Configuration entry point that wires planner, graph reasoning, gap detection, and external channels."""
import hashlib
import json
from typing import Any, Dict, Literal, Optional, List

from pydantic import BaseModel, Field

from application.rag_inference.deepsearch.service import DeepSearchService
from config.core.deepsearch.graph_adapter_config import GraphAdapterConfig
from config.core.deepsearch.gap_config import GapDetectionEvaluatorConfig
from config.encapsulation.mcp.client_config import MCPClientConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.deepsearch.plan import DeepSearchPlanner
from core.deepsearch.reasoning import GraphReasoningLoop
from core.deepsearch.gap import GapDetectionEngine
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.tooling import DeepSearchToolManager
from core.deepsearch.external import ExternalSearchChannel
from framework.config import AbstractConfig


class PlannerRuntimeConfig(BaseModel):
    """Planner runtime knobs controlling ReAct/IterResearch/Parallel-Thinking."""

    mode: Literal["react", "iter_research", "parallel_thinking"] = Field(
        "react", description="Default lightweight mode; switch to iter_research/parallel_thinking when needed."
    )
    max_steps: int = Field(8, description="Maximum planner steps per question.")
    enable_sub_question: bool = Field(True, description="Allow heuristic sub-question expansion.")
    persist_plan: bool = Field(True, description="Persist plan JSON artifacts for replay/debugging.")
    plan_output_dir: str = Field("./local/deepsearch_runs", description="Directory for persisted plan artifacts.")
    allow_external_channel: bool = Field(False, description="Enable optional web/external channel steps from planner.")
    graph_channel_tool: str = Field("graph_adapter.query", description="Default tool name for graph channel steps.")
    text_channel_tool: str = Field(
        "graph.context_rollup",
        description="Default text-channel summariser (GraphSearch-style chunk rollup).",
    )
    web_channel_tool: str = Field("web.search", description="Default tool name for web channel steps.")
    default_web_provider: Optional[str] = Field(None, description="Fallback provider for web/search tools.")
    tool_arg_templates: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Channel-specific tool argument templates (string.Template supported).",
    )


class GraphReasoningThinkConfig(BaseModel):
    """Controls think window cadence / Parallel Thinking behaviour."""

    tool_name: str = Field("graph.think", description="Tool used for think checkpoints.")
    every_n_steps: int = Field(0, description="Trigger periodic think after N completed graph steps (0 disables).")
    min_coverage: float = Field(0.75, description="Only run think when coverage ratio falls below this threshold.")


class GraphReasoningStrategyConfig(BaseModel):
    """Chain-of-exploration parameters, semantic channel flags, and think settings."""

    strategy_name: str = Field("ppr_chain", description="Primary traversal strategy label.")
    allow_semantic_channel: bool = Field(True, description="Allow semantic-only channel as fallback.")
    chain_depth: int = Field(4, description="Maximum traversal chain depth.")
    enable_custom_hooks: bool = Field(False, description="Reserved flag for custom traversal hooks.")
    think: GraphReasoningThinkConfig = Field(
        default_factory=GraphReasoningThinkConfig,
        description="Think window configuration consumed by GraphReasoningLoop.",
    )
    parallel_branches: int = Field(1, description="Number of branches used for parallel thinking (reporter hint).")


class GapDetectionConfig(BaseModel):
    """Threshold tuning for the gap evaluator."""

    coverage_threshold: float = Field(0.7, description="Minimum coverage score before treating the answer as complete.")
    confidence_threshold: float = Field(0.6, description="Minimum model-reported confidence score.")
    expected_min_chunks: int = Field(3, description="Target evidence chunk count for coverage normalization.")
    enable_external_on_gap: bool = Field(True, description="Trigger external search when a gap is detected.")


class ReporterConfig(BaseModel):
    """Report generation options."""

    include_graph_viz: bool = Field(True, description="Include traversal metadata in the report payload.")
    enable_custom_summary: bool = Field(False, description="Enable custom domain summaries.")
    parallel_thinking_runs: int = Field(1, description="Number of combined parallel-thinking passes.")


class ToolManagerConfig(BaseModel):
    """Declarative knobs for tool registration and auditing."""

    enable_builtin_tools: bool = Field(True, description="Load the built-in tool set when true.")
    llm_connector: Optional[Any] = Field(
        None, description="LLM connector instance injected by the application layer."
    )
    default_mcp_server: Optional[str] = Field(None, description="Default MCP server routing hint.")
    remote_argument_templates: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Per-tool argument templates evaluated via string.Template."
    )
    enabled_tools: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Extra flags per tool (enablement, params, MCP overrides).",
    )
    audit_label: Optional[str] = Field(None, description="Optional label propagated to telemetry.")
    remote_tools: Dict[str, "RemoteToolDescriptorConfig"] = Field(
        default_factory=dict, description="Descriptors for remote-only tools."
    )


class RemoteToolDescriptorConfig(BaseModel):
    """Descriptor for remote-only tools exposed via MCP."""

    description: str = Field(..., description="Human-readable tool summary.")
    channel: str = Field("graph", description="Channel label used by the planner.")
    namespace: str = Field(..., description="MCP namespace for remote invocation.")
    profile: str = Field("F", description="Planner profile tag (F/H/X).")
    determinism: str = Field("deterministic", description="Determinism label shown to the planner.")
    speed: str = Field("medium", description="Speed hint (fast/medium/slow).")
    cost: str = Field("medium", description="Cost hint (low/medium/high).")
    strategy_tags: List[str] = Field(default_factory=list, description="Optional strategy hints for the planner.")


class ExternalChannelConfig(BaseModel):
    """Runtime constraints for the optional external search channel."""

    default_provider: Optional[str] = Field(None, description="Preferred provider identifier, e.g. tavily.")
    max_rounds: int = Field(2, description="Maximum number of tasks executed per request.")
    enabled: bool = Field(False, description="Force-enable the channel even when env flags disable it.")
    context_window_limit: int = Field(12, description="Evidence window size forwarded to external tools.")
    http_timeout: float = Field(20.0, description="Timeout applied to HTTP-based providers.")
    max_results: int = Field(5, description="Maximum documents returned by HTTP providers.")


class DeepSearchServiceConfig(AbstractConfig):
    """Application-layer builder that assembles all DeepSearch components."""

    type: Literal["deepsearch_service"] = "deepsearch_service"
    planner: PlannerRuntimeConfig
    graph_adapter: GraphAdapterConfig
    graph_reasoning: GraphReasoningStrategyConfig
    gap_detection: GapDetectionConfig
    reporter: ReporterConfig
    tool_manager: ToolManagerConfig
    external_channel: ExternalChannelConfig
    mcp_client: Optional[MCPClientConfig] = Field(
        default=None, description="Optional MCP client config used for remote tools."
    )
    llm: Optional[OpenAIChatConfig] = Field(
        default=None,
        description="Optional LLM config shared across planner reasoning, graph tools, and external channels.",
    )

    def build(self) -> DeepSearchService:
        llm_connector = self._build_llm_connector()
        adapter = self.graph_adapter.build()
        mcp_client = self._build_mcp_client()
        tool_manager = self._build_tool_manager(llm_connector=llm_connector, mcp_client=mcp_client)
        planner = DeepSearchPlanner(
            prompt_store=None,
            llm_connector=llm_connector,
            config=self.planner,
        )
        graph_loop = GraphReasoningLoop(
            adapter=adapter,
            llm_connector=llm_connector,
            strategy_config=self.graph_reasoning,
            tool_manager=tool_manager,
        )
        gap_detector = self._build_gap_detector()
        reporter = DeepSearchReporter(
            template_store=None,
            config=self.reporter.model_dump(),
        )
        external_channel = ExternalSearchChannel(
            tool_manager=tool_manager,
            config=self.external_channel,
        )
        return DeepSearchService(
            planner=planner,
            graph_loop=graph_loop,
            gap_detector=gap_detector,
            reporter=reporter,
            tool_manager=tool_manager,
            external_channel=external_channel,
            config={"name": "deepsearch-service", "fingerprint": self._fingerprint()},
        )

    # ------------------------------------------------------------------
    def _build_llm_connector(self):
        if self.llm:
            return self.llm.build()
        try:
            return OpenAIChatConfig().build()
        except Exception:
            return None

    def _build_mcp_client(self):
        if not self.mcp_client:
            return None
        return self.mcp_client.build()

    def _build_tool_manager(self, *, llm_connector, mcp_client):
        payload = self.tool_manager.model_dump()
        payload["llm_connector"] = payload.get("llm_connector") or llm_connector
        return DeepSearchToolManager(
            tool_configs=payload,
            telemetry_client=None,
            mcp_client=mcp_client,
        )

    def _build_gap_detector(self) -> GapDetectionEngine:
        evaluator_config = GapDetectionEvaluatorConfig(
            coverage_threshold=self.gap_detection.coverage_threshold,
            confidence_threshold=self.gap_detection.confidence_threshold,
            expected_min_chunks=self.gap_detection.expected_min_chunks,
        )
        evaluator = evaluator_config.build()
        return GapDetectionEngine(
            evaluator,
            telemetry_client=None,
            config=self.gap_detection.model_dump(),
        )

    def _fingerprint(self) -> str:
        payload = self.model_dump()
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
