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
from core.deepsearch.reasoning import MultiAgentGraphReasoningLoop
from core.deepsearch.gap import GapDetectionEngine
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.tooling.registry import ToolHintRegistry
from encapsulation.deepsearch.tooling import DeepSearchToolManager
from encapsulation.deepsearch.external import ExternalSearchChannel
from encapsulation.deepsearch.telemetry import LoggingTelemetryClient
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
        description="Default text-channel summariser (chunk rollup).",
    )
    web_channel_tool: str = Field("web.search", description="Default tool name for web channel steps.")
    include_llm_tools_in_catalog: bool = Field(
        ...,
        description="Whether to include LLM-dependent tools in the planner tool catalog (must be explicit; no env heuristics).",
    )
    default_web_provider: Optional[str] = Field(None, description="Fallback provider for web/search tools.")
    tool_arg_templates: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Channel-specific tool argument templates (string.Template supported).",
    )
    honor_planner_tool_selection: bool = Field(
        ...,
        description="When true, DeepSearchPlanner will honor explicit tool selections in plan steps (must be explicit).",
    )
    graph_adapter_name: str = Field(
        "hipporag",
        description="Default graph adapter name used in plan artifacts (must match a registered adapter).",
    )


class GraphReasoningThinkConfig(BaseModel):
    """Controls think window cadence / Parallel Thinking behaviour."""

    tool_name: str = Field("graph.think", description="Tool used for think checkpoints.")
    every_n_steps: int = Field(0, description="Trigger periodic think after N completed graph steps (0 disables).")
    min_coverage: float = Field(0.75, description="Only run think when coverage ratio falls below this threshold.")
    enable_tool_calls: bool = Field(
        False,
        description="Allow think checkpoints to propose additional tool calls (executed immediately by the reasoning loop).",
    )
    max_tool_calls: int = Field(0, description="Maximum tool calls accepted from a single think response.")
    tool_call_concurrency: int = Field(0, description="Max concurrent think-proposed tool invocations (0 = sequential).")
    tool_catalog_max_items: int = Field(
        0,
        description="Include up to N tool descriptors in think context (0 disables the tool catalog).",
    )
    include_llm_tools: bool = Field(
        ...,
        description="Whether to include LLM-dependent tools in the think tool catalog (must be explicit; no env heuristics).",
    )
    max_rounds_per_checkpoint: int = Field(
        1,
        description="Maximum think→tool_calls→think iterations per periodic checkpoint (>=1).",
    )


class CompressionBranchConfig(BaseModel):
    """Shared compaction schema used across tool contexts and think windows."""

    mode: Literal["truncate", "excerpt"] = Field(
        "truncate",
        description="truncate keeps prefixes; excerpt extracts windows around key terms.",
    )
    max_items: int = Field(0, description="Maximum number of evidence items kept (0 disables item limit).")
    max_chars: int = Field(0, description="Maximum characters kept per evidence item (0 disables truncation).")
    excerpt_chars: int = Field(900, description="Excerpt window size (only used when mode == 'excerpt').")
    retention: Literal["head", "tail"] = Field("tail", description="Retention policy when max_items applies.")


class GraphReasoningCompressionConfig(BaseModel):
    """Unified `compression` schema: tool_context vs think."""

    tool_context: CompressionBranchConfig = Field(
        default_factory=lambda: CompressionBranchConfig(
            mode="truncate",
            max_items=5,
            max_chars=800,
            excerpt_chars=900,
            retention="tail",
        ),
        description="Compaction settings applied to tool payload `context_evidences`.",
    )
    think: CompressionBranchConfig = Field(
        default_factory=lambda: CompressionBranchConfig(
            mode="truncate",
            max_items=8,
            max_chars=1600,
            excerpt_chars=900,
            retention="head",
        ),
        description="Compaction settings applied to `graph.think` context window.",
    )


class GraphReasoningStrategyConfig(BaseModel):
    """Chain-of-exploration parameters, semantic channel flags, and think settings."""

    strategy_name: str = Field("ppr_chain", description="Primary traversal strategy label.")
    allow_semantic_channel: bool = Field(True, description="Allow semantic-only channel as fallback.")
    chain_depth: int = Field(4, description="Maximum traversal chain depth.")
    enable_custom_hooks: bool = Field(False, description="Reserved flag for custom traversal hooks.")
    tool_context_max_evidences: int = Field(
        5,
        description="Deprecated: prefer compression.tool_context.max_items (kept for backward compatibility).",
    )
    tool_context_max_chars: int = Field(
        800,
        description="Deprecated: prefer compression.tool_context.max_chars (kept for backward compatibility).",
    )
    compression: GraphReasoningCompressionConfig = Field(
        default_factory=GraphReasoningCompressionConfig,
        description="Unified evidence compaction schema shared by tool payload windows and think checkpoints.",
    )
    coverage_expected_min_chunks: int = Field(
        3,
        description="Expected minimum evidence chunks for coverage normalization inside the reasoning loop.",
    )
    trace_reflection_enabled: bool = Field(
        True,
        description="Emit short user-visible reflections after each step (Trace-first UX).",
    )
    trace_reflection_max: int = Field(
        24,
        description="Maximum reflection messages emitted per run when trace_reflection_enabled is true.",
    )
    step_summary_max_chars: int = Field(
        2000,
        description="Maximum characters kept for adapter step summaries in traces/reports.",
    )
    max_parallel_branches: int = Field(
        4,
        description="Upper bound for auto parallel scheduling when parallel_branches <= 0.",
    )
    think: GraphReasoningThinkConfig = Field(
        default_factory=GraphReasoningThinkConfig,
        description="Think window configuration consumed by GraphReasoningLoop.",
    )
    parallel_branches: int = Field(
        1,
        description="Number of parallel branches; set <=0 to enable auto scheduling up to max_parallel_branches.",
    )
    tool_timeout_seconds: float = Field(
        45.0,
        description="Safety timeout applied to each tool/MCP invocation triggered by the reasoning loop.",
    )


class MultiAgentConfig(BaseModel):
    """Lead/worker orchestration knobs for DeepSearch reasoning."""

    enabled: bool = Field(..., description="Enable the lead/worker orchestrator for graph reasoning.")
    max_subagents: int = Field(..., description="Maximum number of worker agents spawned per request.")
    subagent_concurrency: int = Field(..., description="Max concurrent worker agents.")
    enable_parallel_tool_probes: bool = Field(..., description="Run fast probe tools in parallel inside each worker.")
    probe_tool_names: List[str] = Field(
        ...,
        description="Fast probe tools executed by each worker (invoked concurrently).",
    )
    probe_concurrency: int = Field(..., description="Max concurrent probe tool invocations per worker.")
    lead_tool_names: List[str] = Field(
        ...,
        description="Optional tools invoked by the lead agent after merging worker evidence.",
    )
    lead_tool_concurrency: int = Field(..., description="Max concurrent tool invocations for lead post-processing.")
    max_merge_evidences: int = Field(
        ...,
        description="Cap the merged evidence list size (after dedupe) to prevent prompt blow-ups.",
    )
    worker_timeout_seconds: Optional[float] = Field(
        ...,
        description="Optional per-worker timeout (seconds). When null, rely on underlying tool timeouts.",
    )
    worker_retry_attempts: int = Field(..., description="Retry attempts per worker session (0 disables retries).")
    fail_fast: bool = Field(
        ...,
        description="Fail the whole request when any worker/probe/lead tool fails.",
    )
    incremental_parallelism: bool = Field(
        ...,
        description="When true, launch a small worker batch first and expand if evidence coverage is insufficient.",
    )
    initial_worker_count: int = Field(
        ...,
        description="Initial number of workers to run when incremental_parallelism is enabled (>=1).",
    )
    stop_min_evidence_count: int = Field(
        ...,
        description="Stop expanding incremental workers once evidence count reaches this threshold (<=0 disables).",
    )
    stop_min_coverage_ratio: float = Field(
        ...,
        description="Stop expanding incremental workers once coverage_ratio reaches this threshold (<=0 disables).",
    )


class GapDetectionConfig(BaseModel):
    """Threshold tuning for the gap evaluator."""

    coverage_threshold: float = Field(0.7, description="Minimum coverage score before treating the answer as complete.")
    confidence_threshold: float = Field(0.6, description="Minimum model-reported confidence score.")
    expected_min_chunks: int = Field(3, description="Target evidence chunk count for coverage normalization.")
    enable_external_on_gap: bool = Field(True, description="Trigger external search when a gap is detected.")


class ReporterConfig(BaseModel):
    """Report generation options."""

    max_highlights: int = Field(6, description="Maximum number of reasoning step highlights included in the report context.")
    include_graph_viz: bool = Field(True, description="Include traversal metadata in the report payload.")
    enable_custom_summary: bool = Field(False, description="Enable custom domain summaries.")
    parallel_thinking_runs: int = Field(1, description="Number of combined parallel-thinking passes.")
    enable_llm_report: bool = Field(True, description="Generate the final report via the LLM.")
    report_temperature: float = Field(0.2, description="Sampling temperature for the report writer.")
    report_max_evidence_chars: int = Field(900, description="Maximum characters per evidence snippet forwarded to the report writer.")
    max_evidence_items: int = Field(
        10,
        description="Maximum number of authoritative evidence chunks included in report prompts/appendices.",
    )
    report_max_graph_chain_items: int = Field(
        200,
        description="Maximum number of graph chain edges listed in the report appendix.",
    )
    report_max_seed_entities: int = Field(
        15,
        description="Maximum number of seed entities listed in the report appendix.",
    )
    enable_consistency_check: bool = Field(
        True, description="Run a consistency check against the evidence after report generation."
    )
    consistency_temperature: float = Field(0.0, description="Sampling temperature for the consistency checker.")
    consistency_max_retries: int = Field(2, description="Max retry attempts for the consistency checker LLM call.")
    consistency_max_claims: int = Field(
        40,
        description="Maximum number of cited claim sentences checked for supportiveness/contradictions.",
    )
    enable_citation_agent: bool = Field(
        True, description="Post-process inline citations and build a structured evidence index."
    )
    parallel_sections: bool = Field(
        False, description="Generate report sections in parallel for faster report generation."
    )
    max_parallel_sections: int = Field(
        4, description="Maximum number of sections to generate concurrently when parallel_sections is enabled."
    )
    sectionwise_writer: bool = Field(
        True,
        description="Write reports section-by-section with bounded evidence windows (recommended for long contexts).",
    )
    sectionwise_retain_k: int = Field(
        5,
        description="Recency retention: keep the last K cited evidence snippets available across section writes.",
    )
    citation_aliases: bool = Field(
        True,
        description="Alias long chunk IDs into short stable tokens for LLM prompting.",
    )
    outline_evidence_summary_chars: int = Field(
        240,
        description="Characters per evidence index summary forwarded to the outline planner.",
    )
    methodology_summary_chars: int = Field(
        1200,
        description="Maximum characters kept per reasoning/tool summary in report prompts.",
    )
    keep_tool_results: int = Field(
        8,
        description="Recency retention budget for tool results in report prompts (-1 keep all, 0 keep none).",
    )
    synthesis_section_max_chars: int = Field(
        1200,
        description="Maximum characters per section body forwarded to the parallel synthesis step.",
    )


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
    artifact_dir: Optional[str] = Field(
        None,
        description="Directory for persisted tool artifacts.",
    )
    max_remote_evidences: int = Field(
        32,
        description="Maximum number of evidences forwarded to a remote (MCP) tool invocation.",
    )
    max_remote_context_chars: int = Field(
        4096,
        description="Maximum characters forwarded to a remote (MCP) tool invocation context window.",
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
    tool_timeout_seconds: float = Field(45.0, description="Timeout applied to tool_manager invocations.")
    cache_mode: str = Field("auto", description="External cache mode: off/record/replay/auto.")
    cache_dir: Optional[str] = Field(None, description="Optional directory for external cache files.")
    tavily_api_key: Optional[str] = Field(None, description="Optional Tavily API key used by the provider implementation.")


class QualityLoopConfig(BaseModel):
    """Quality gate settings powering research→verify→iterate loops."""

    enabled: bool = Field(
        False,
        description="Enable report quality gate and allow iterative follow-up retrieval when gates fail.",
    )
    max_rounds: int = Field(2, description="Maximum total rounds (initial + follow-ups).")
    min_citation_sentence_coverage: float = Field(
        0.6, description="Minimum fraction of report sentences that include valid citations."
    )
    require_consistency: bool = Field(
        True,
        description="Fail the quality gate when the consistency checker reports issues.",
    )
    max_uncited_sentences: int = Field(
        6,
        description="Maximum uncited sentences to surface as repair targets.",
    )
    max_actions: int = Field(6, description="Maximum follow-up actions returned by the quality gate.")
    enable_llm_judge: bool = Field(True, description="Use an LLM rubric judge for gate scoring + actions.")
    judge_temperature: float = Field(0.0, description="Sampling temperature for the quality judge.")
    judge_max_retries: int = Field(1, description="Retry attempts for the quality judge call.")
    trigger_external_on_quality_failure: bool = Field(
        True,
        description="Allow the quality gate to request external search when enabled.",
    )


class DeepSearchServiceConfig(AbstractConfig):
    """Application-layer builder that assembles all DeepSearch components."""

    type: Literal["deepsearch_service"] = "deepsearch_service"
    planner: PlannerRuntimeConfig
    graph_adapter: GraphAdapterConfig
    graph_reasoning: GraphReasoningStrategyConfig
    multi_agent: MultiAgentConfig
    gap_detection: GapDetectionConfig
    reporter: ReporterConfig
    tool_manager: ToolManagerConfig
    external_channel: ExternalChannelConfig
    quality_loop: QualityLoopConfig = Field(default_factory=QualityLoopConfig)
    mcp_client: Optional[MCPClientConfig] = Field(
        default=None, description="Optional MCP client config used for remote tools."
    )
    llm: Optional[OpenAIChatConfig] = Field(
        default=None,
        description="Optional LLM config shared across planner reasoning, graph tools, and external channels.",
    )
    telemetry_enabled: bool = Field(
        True,
        description="Enable the built-in telemetry logger to surface tool/gap/external events.",
    )

    def build(self) -> DeepSearchService:
        llm_connector = self._build_llm_connector()
        adapter = self.graph_adapter.build()
        mcp_client = self._build_mcp_client()
        telemetry_client = self._build_telemetry_client() if self._resolve_telemetry_flag() else None
        tool_hint_registry = ToolHintRegistry()
        tool_manager = self._build_tool_manager(
            llm_connector=llm_connector,
            mcp_client=mcp_client,
            telemetry_client=telemetry_client,
            tool_hint_registry=tool_hint_registry,
        )
        planner = DeepSearchPlanner(
            prompt_store=None,
            llm_connector=llm_connector,
            config=self.planner,
            tool_hint_registry=tool_hint_registry,
        )
        
        graph_loop = MultiAgentGraphReasoningLoop(
            adapter=adapter,
            llm_connector=llm_connector,
            strategy_config=self.graph_reasoning,
            tool_manager=tool_manager,
            settings=self.multi_agent.model_dump(),
            graph_channel_tool=self.planner.graph_channel_tool,
        )
        gap_detector = self._build_gap_detector(telemetry_client=telemetry_client)
        graph_store = self._resolve_graph_store(adapter)
        reporter = DeepSearchReporter(
            template_store=None,
            config=self.reporter.model_dump(),
            llm_connector=llm_connector,
            graph_store=graph_store,
        )
        external_channel = ExternalSearchChannel(
            tool_manager=tool_manager,
            config=self.external_channel,
            telemetry_client=telemetry_client,
        )
        return DeepSearchService(
            planner=planner,
            graph_loop=graph_loop,
            gap_detector=gap_detector,
            reporter=reporter,
            tool_manager=tool_manager,
            external_channel=external_channel,
            config={
                "name": "deepsearch-service",
                "fingerprint": self._fingerprint(),
                "artifact_dir": self.tool_manager.artifact_dir,
                "quality_loop": self.quality_loop.model_dump(),
                "tool_names": {
                    "graph_channel_tool": self.planner.graph_channel_tool,
                    "text_channel_tool": self.planner.text_channel_tool,
                    "web_channel_tool": self.planner.web_channel_tool,
                    "think_tool": self.graph_reasoning.think.tool_name,
                },
                "coverage_expected_min_chunks": self.graph_reasoning.coverage_expected_min_chunks,
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_graph_store(adapter) -> Any | None:
        retriever = getattr(adapter, "retriever", None)
        if retriever is None:
            return None
        return getattr(retriever, "graph_store", None)

    def _build_llm_connector(self):
        if not self.llm:
            raise ValueError("DeepSearchServiceConfig.llm is required (no implicit LLM fallback).")
        return self.llm.build()

    def _build_mcp_client(self):
        if not self.mcp_client:
            return None
        return self.mcp_client.build()

    def _build_tool_manager(self, *, llm_connector, mcp_client, telemetry_client, tool_hint_registry: ToolHintRegistry):
        payload = self.tool_manager.model_dump()
        payload["llm_connector"] = payload.get("llm_connector") or llm_connector
        if not payload.get("artifact_dir"):
            raise ValueError("tool_manager.artifact_dir is required (no implicit default).")
        return DeepSearchToolManager(
            tool_configs=payload,
            telemetry_client=telemetry_client,
            mcp_client=mcp_client,
            tool_hint_registry=tool_hint_registry,
        )

    def _build_gap_detector(self, *, telemetry_client) -> GapDetectionEngine:
        evaluator_config = GapDetectionEvaluatorConfig(
            coverage_threshold=self.gap_detection.coverage_threshold,
            confidence_threshold=self.gap_detection.confidence_threshold,
            expected_min_chunks=self.gap_detection.expected_min_chunks,
        )
        evaluator = evaluator_config.build()
        gap_config = self.gap_detection.model_dump()
        gap_config["external_channel_enabled"] = bool(self.external_channel.enabled)
        return GapDetectionEngine(
            evaluator,
            telemetry_client=telemetry_client,
            config=gap_config,
        )

    @staticmethod
    def _build_telemetry_client():
        return LoggingTelemetryClient()

    def _resolve_telemetry_flag(self) -> bool:
        return bool(self.telemetry_enabled)

    def _fingerprint(self) -> str:
        payload = self.model_dump()
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
