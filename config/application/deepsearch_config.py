"""Configuration entry point that wires planner, graph reasoning, and reporting."""
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, List

from pydantic import BaseModel, Field

from config.core.deepsearch import bench_answer_defaults
from config.core.deepsearch.computable_gate_defaults import DEFAULT_COMPUTABLE_POLICY
from config.core.deepsearch.planner_web_policy_defaults import (
    DEFAULT_REALTIME_WEB_INTENT_KEYWORDS,
    DEFAULT_REALTIME_WEB_KEYWORDS,
    DEFAULT_REALTIME_WEB_STRONG_KEYWORDS,
    DEFAULT_REALTIME_WEB_TOPIC_KEYWORDS,
)
from application.rag_inference.deepsearch.service import DeepSearchService
from config.core.deepsearch.graph_adapter_config import GraphAdapterConfig
from config.encapsulation.mcp.client_config import MCPClientConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.deepsearch.plan import DeepSearchPlanner
from core.deepsearch.reasoning import MultiAgentGraphReasoningLoop
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.tooling.registry import ToolHintRegistry
from core.deepsearch.tooling.adapter_capability_gate import disabled_tools_for_adapter, merge_disabled_tools
from encapsulation.deepsearch.tooling import DeepSearchToolManager
from encapsulation.deepsearch.telemetry import LoggingTelemetryClient
from framework.config import AbstractConfig

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
_UNRESOLVED_ENV_PLACEHOLDER = object()

try:  # Keep substitution rules consistent with framework.Register
    from config.env_placeholder_policy import ENV_DEFAULTS as _ENV_DEFAULTS
    from config.env_placeholder_policy import SILENT_MISSING_ENV_VARS as _SILENT_MISSING_ENV_VARS
except Exception:  # pragma: no cover - defensive for minimal runtimes
    _ENV_DEFAULTS = {}
    _SILENT_MISSING_ENV_VARS = set()


def _substitute_env(obj: Any):
    """Apply ${VAR} substitutions similar to framework.Register."""

    if isinstance(obj, dict):
        resolved: Dict[str, Any] = {}
        for key, value in obj.items():
            substituted = _substitute_env(value)
            if substituted is _UNRESOLVED_ENV_PLACEHOLDER:
                continue
            resolved[key] = substituted
        return resolved
    if isinstance(obj, list):
        items = [_substitute_env(item) for item in obj]
        return [None if item is _UNRESOLVED_ENV_PLACEHOLDER else item for item in items]
    if isinstance(obj, str):
        whole = re.fullmatch(r"\$\{([^}]+)\}", obj.strip())
        if whole:
            var_name = whole.group(1)
            env_value = os.getenv(var_name)
            if not env_value:
                default_value = _ENV_DEFAULTS.get(var_name)
                if default_value is not None:
                    return default_value
                if var_name in _SILENT_MISSING_ENV_VARS:
                    return _UNRESOLVED_ENV_PLACEHOLDER
                return _UNRESOLVED_ENV_PLACEHOLDER
            return env_value
        return _ENV_PATTERN.sub(_replace_env, obj)
    return obj


def _replace_env(match: re.Match[str]) -> str:
    var = match.group(1)
    value = os.getenv(var)
    if not value:
        default_value = _ENV_DEFAULTS.get(var)
        if default_value is not None:
            return default_value
        if var in _SILENT_MISSING_ENV_VARS:
            return match.group(0)
        return match.group(0)
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path


def _load_search_retriever_payload(path: str | Path) -> Dict[str, Any]:
    payload_path = _resolve_config_path(path)
    raw_text = payload_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    substituted = _substitute_env(data)
    if not isinstance(substituted, dict):
        raise ValueError("search retriever config must be a JSON object")
    if isinstance(substituted.get("retrieval_config"), dict):
        return substituted["retrieval_config"]
    return substituted


def _build_search_retrievers(path: str | Path) -> tuple[Any | None, Any | None]:
    from config.core.retrieval.dense_config import DenseRetrieverConfig
    from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig

    payload = _load_search_retriever_payload(path)
    cfg_type = str(payload.get("type") or "").strip()
    dense = None
    bm25 = None
    if cfg_type == "multipath":
        retrievers = payload.get("retrievers")
        if not isinstance(retrievers, list):
            raise ValueError("multipath retriever config requires a list of retrievers")
        for entry in retrievers:
            if not isinstance(entry, dict):
                continue
            entry_type = str(entry.get("type") or "").strip()
            if entry_type == "dense" and dense is None:
                dense = DenseRetrieverConfig.model_validate(entry).build()
            elif entry_type == "tantivy_bm25" and bm25 is None:
                bm25 = TantivyBM25RetrieverConfig.model_validate(entry).build()
    elif cfg_type == "dense":
        dense = DenseRetrieverConfig.model_validate(payload).build()
    elif cfg_type == "tantivy_bm25":
        bm25 = TantivyBM25RetrieverConfig.model_validate(payload).build()
    else:
        raise ValueError(f"Unsupported search retriever config type: {cfg_type or 'unknown'}")

    if dense is None and bm25 is None:
        raise ValueError("search retriever config did not build dense or bm25 retrievers")
    if dense is None:
        logger.warning("search retriever config missing dense retriever (FAISS will be unavailable)")
    if bm25 is None:
        logger.warning("search retriever config missing bm25 retriever (BM25 will be unavailable)")
    return dense, bm25


def _inject_search_retrievers(payload: Dict[str, Any], *, dense: Any | None, bm25: Any | None) -> None:
    if dense is None and bm25 is None:
        return
    enabled_tools = payload.setdefault("enabled_tools", {})
    if not isinstance(enabled_tools, dict):
        return

    def _ensure_params(tool_name: str) -> Dict[str, Any] | None:
        cfg = enabled_tools.get(tool_name)
        if cfg is None:
            cfg = {}
            enabled_tools[tool_name] = cfg
        if not isinstance(cfg, dict):
            return None
        params = cfg.get("params")
        if params is None:
            params = {}
            cfg["params"] = params
        if not isinstance(params, dict):
            return None
        return params

    for tool_name in ("search", "search.faiss", "search.bm25", "knowledge_base.explore"):
        params = _ensure_params(tool_name)
        if params is None:
            continue
        if dense is not None and "dense_retriever" not in params:
            params["dense_retriever"] = dense
        if bm25 is not None and "bm25_retriever" not in params:
            params["bm25_retriever"] = bm25


class PlannerRuntimeConfig(BaseModel):
    """Planner runtime knobs controlling ReAct/IterResearch/Parallel-Thinking."""

    mode: Literal["react", "iter_research", "parallel_thinking"] = Field(
        "react", description="Default lightweight mode; switch to iter_research/parallel_thinking when needed."
    )
    max_steps: int = Field(8, description="Maximum planner steps per question.")
    enable_sub_question: bool = Field(True, description="Allow heuristic sub-question expansion.")
    persist_plan: bool = Field(True, description="Persist plan JSON artifacts for replay/debugging.")
    plan_output_dir: str = Field("./local/deepsearch_runs", description="Directory for persisted plan artifacts.")
    web_step_policy: Literal["off", "realtime_required"] = Field(
        "realtime_required",
        description=(
            "Policy for including at least one web search step in the plan. "
            "'realtime_required' injects/forces a web step when the question asks for realtime/latest/current info."
        ),
    )
    realtime_web_keywords: List[str] = Field(
        default_factory=lambda: list(DEFAULT_REALTIME_WEB_KEYWORDS),
        description=(
            "Keyword cues (substring match) that indicate realtime/latest/current requirements. "
            "Used only when web_step_policy='realtime_required'."
        ),
    )
    realtime_web_strong_keywords: List[str] = Field(
        default_factory=lambda: list(DEFAULT_REALTIME_WEB_STRONG_KEYWORDS),
        description="Strong keyword cues that force a web step even without topic matching (e.g. '引用网络来源').",
    )
    realtime_web_intent_keywords: List[str] = Field(
        default_factory=lambda: list(DEFAULT_REALTIME_WEB_INTENT_KEYWORDS),
        description="Recency intent keywords; used together with realtime_web_topic_keywords.",
    )
    realtime_web_topic_keywords: List[str] = Field(
        default_factory=lambda: list(DEFAULT_REALTIME_WEB_TOPIC_KEYWORDS),
        description="Time-sensitive topic keywords; used together with realtime_web_intent_keywords.",
    )
    graph_channel_tool: str = Field("graph_adapter.query", description="Default tool name for graph channel steps.")
    text_channel_tool: str = Field(
        "search",
        description="Default text-channel tool (fallback to search when text-only steps are needed).",
    )
    web_channel_tool: str = Field("web.search", description="Default tool name for web channel steps.")
    include_llm_tools_in_catalog: bool = Field(
        ...,
        description="Whether to include LLM-dependent tools in the planner tool catalog (must be explicit; no env heuristics).",
    )
    tool_arg_templates: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Channel-specific tool argument templates (string.Template supported).",
    )
    tool_catalog_allowlist: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional allowlist of tool names shown to the planner (reduces prompt size/cognitive load). "
            "When set, the planner catalog will include only these tools plus the graph adapter traversal primitive."
        ),
    )
    tool_catalog_max_items: int = Field(
        0,
        ge=0,
        le=200,
        description="Optional hard cap on the number of tool descriptors shown to the planner (0 disables).",
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

    tool_name: str = Field("think", description="Tool used for think checkpoints.")
    every_n_steps: int = Field(0, description="Trigger periodic think after N completed graph steps (0 disables).")
    min_coverage: float = Field(0.75, description="Only run think when coverage ratio falls below this threshold.")
    always_run: bool = Field(
        False,
        description="When true, run periodic think on cadence regardless of the coverage ratio.",
    )
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
    tool_catalog_allowlist: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional allowlist of tool names shown to the think tool (reduces prompt size/cognitive load). "
            "When set, only these tools are exposed in the think tool catalog."
        ),
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
        description="Compaction settings applied to `think` context window.",
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
        description="Expected minimum PRIMARY evidence chunks for coverage normalization inside the reasoning loop.",
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
        description="Stop expanding incremental workers once PRIMARY evidence count reaches this threshold (<=0 disables).",
    )
    stop_min_coverage_ratio: float = Field(
        ...,
        description="Stop expanding incremental workers once coverage_ratio reaches this threshold (<=0 disables).",
    )


class BenchAnswerTypeConfig(BaseModel):
    """Per-question-type settings for benchmark-mode answer synthesis."""

    mode: bench_answer_defaults.BenchAnswerMode = Field(
        "single_stage",
        description="Answer synthesis mode: single_stage (1 call) or two_stage (extract -> answer).",
    )
    preference: bench_answer_defaults.BenchAnswerPreference = Field(
        "balanced",
        description="Optimization preference: correctness (reduce FP), coverage (reduce FN), balanced.",
    )
    max_evidence_items: int = Field(
        bench_answer_defaults.DEFAULT_BENCH_MAX_EVIDENCE_ITEMS,
        description="Max evidence snippets included in bench evidence block.",
    )
    max_evidence_chars: int = Field(
        bench_answer_defaults.DEFAULT_BENCH_MAX_EVIDENCE_CHARS,
        description="Max total characters for the bench evidence block.",
    )
    snippet_chars: int = Field(
        bench_answer_defaults.DEFAULT_BENCH_SNIPPET_CHARS,
        description="Max characters per extracted snippet before block concatenation.",
    )


class BenchAnswerConfig(BaseModel):
    """Benchmark-mode answer synthesis knobs.

    This config is consumed by `application.rag_inference.deepsearch.service_bench` and
    `core.deepsearch.report.bench_answer` only (no impact on product report generation).
    """

    enabled: bool = Field(True, description="Enable benchmark-mode answer synthesis settings when bench_mode=1.")
    allowed_evidence_kinds: List[str] = Field(
        default_factory=lambda: list(bench_answer_defaults.DEFAULT_BENCH_ALLOWED_EVIDENCE_KINDS),
        description="Evidence kinds allowed to enter the bench evidence block (default: primary only).",
    )
    heading_window_max_lines: int = Field(
        bench_answer_defaults.DEFAULT_BENCH_HEADING_WINDOW_MAX_LINES,
        description="When a heading line matches the question, include up to N following lines to preserve bullets.",
    )
    default_policy: BenchAnswerTypeConfig = Field(
        default_factory=BenchAnswerTypeConfig,
        description="Fallback policy when question_type is missing or unmapped.",
    )
    policies_by_question_type: Dict[str, BenchAnswerTypeConfig] = Field(
        default_factory=lambda: {
            key: BenchAnswerTypeConfig.model_validate(value)
            for key, value in bench_answer_defaults.DEFAULT_BENCH_POLICIES_BY_QUESTION_TYPE.items()
        },
        description="Overrides keyed by dataset-provided question_type labels (e.g., GraphRAG-Benchmark).",
    )


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
    bench_answer: BenchAnswerConfig = Field(
        default_factory=BenchAnswerConfig,
        description="Settings for benchmark-mode answer synthesis (only used when bench_mode=1).",
    )


class ToolManagerConfig(BaseModel):
    """Declarative knobs for tool registration and auditing."""

    enable_builtin_tools: bool = Field(True, description="Load the built-in tool set when true.")
    search_retriever_config_path: Optional[str] = Field(
        None,
        description="Optional path to a retriever config (or rag_inference.json) used to build FAISS/BM25 for search.",
    )
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
        "./local/deepsearch_artifacts",
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


class ToolBudgetConfig(BaseModel):
    """Global tool-call budget per DeepSearch run (not counting graph adapter traversals)."""

    enabled: bool = Field(True, description="Enable tool-call budget enforcement across tool_manager tools.")
    max_calls_total: int = Field(
        60,
        ge=0,
        description="Maximum total tool invocations allowed for one DeepSearch run (0 disables tool calls).",
    )
    expose_to_llm: bool = Field(
        True,
        description="When true, attach remaining tool-call budget to graph_context metadata for LLM visibility.",
    )


class ArtifactRefsConfig(BaseModel):
    """Reference/pointer emission settings for persisted DeepSearch artifacts."""

    enabled: bool = Field(
        True,
        description="When true, emit structured $ref objects instead of duplicating large payloads across files.",
    )


class PublicArtifactsProfileConfig(BaseModel):
    """Public (frontend-facing) artifact trimming settings."""

    include_final_report_in_json: bool = Field(
        False,
        description="When false, exclude the final report text from public.json (prefer SSE write stream).",
    )
    max_plan_steps: int = Field(12, ge=0, description="Maximum number of plan steps retained in public.json.")
    max_stage_history: int = Field(128, ge=0, description="Maximum stage_history entries retained in public.json.")
    max_errors: int = Field(64, ge=0, description="Maximum error entries retained in public.json.")


class ArtifactDedupeConfig(BaseModel):
    """Run-artifact de-duplication settings (avoid cross-file repeats via refs)."""

    enabled: bool = Field(True, description="When true, persist de-duplicated reasoning/report JSON (v2 artifacts).")
    evidence_pool_filename: str = Field(
        "evidence_pool.json",
        description="Filename for pooled evidences referenced by reasoning/report artifacts (relative to run dir).",
    )


class ArtifactsConfig(BaseModel):
    """DeepSearch run-artifact settings (manifest + dev/public views)."""

    enabled: bool = Field(True, description="When true, persist DeepSearch run artifacts to artifact_dir.")
    version: int = Field(2, description="Artifact schema version (v2 = manifest/dev/public).")
    profiles: List[Literal["dev", "public"]] = Field(
        default_factory=lambda: ["dev", "public"],
        description="Artifact views to generate for each run.",
    )
    state_snapshot_mode: Literal["manifest", "legacy"] = Field(
        "manifest",
        description="state_snapshot.json content mode: 'manifest' (v2) or 'legacy' (includes report+reasoning).",
    )
    refs: ArtifactRefsConfig = Field(default_factory=ArtifactRefsConfig)
    public: PublicArtifactsProfileConfig = Field(default_factory=PublicArtifactsProfileConfig)
    dedupe: ArtifactDedupeConfig = Field(default_factory=ArtifactDedupeConfig)


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


class DeterministicRoutingConfig(BaseModel):
    """Application-layer routing + gates for 'computable' questions."""

    enabled: bool = Field(
        False,
        description="Enable computable-question classification + deterministic evidence gating.",
    )
    classifier: Literal["llm", "heuristic", "hybrid"] = Field(
        "llm",
        description=(
            "Classifier mode: "
            "`llm` uses a low-cost LLM to decide computability; "
            "`heuristic` uses configured keyword/operator cues; "
            "`hybrid` tries LLM then falls back to heuristic."
        ),
    )
    fail_on_classifier_error: bool = Field(
        False,
        description=(
            "When true, abort the request if the configured classifier fails "
            "(avoids silently bypassing computable gating)."
        ),
    )
    hard_gate_missing_deterministic_tools: bool = Field(
        False,
        description=(
            "When true and the question is classified as computable, fail-fast if no deterministic tool runs occurred "
            "(prevents returning non-verifiable numeric/time answers)."
        ),
    )
    # Heuristic-only tuning (empty by default; set explicitly in json_configs for domain deployments).
    computable_keywords: List[str] = Field(
        default_factory=list,
        description="Keyword cues used by the heuristic computable-question classifier (lowercased substring matching).",
    )
    computable_operators: List[str] = Field(
        default_factory=list,
        description="Operator/constraint cues used by the heuristic computable-question classifier.",
    )
    computable_policy: Dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_COMPUTABLE_POLICY),
        description="Policy knobs for the heuristic classifier (min hits / require weak cue).",
    )
    llm_model_name: Optional[str] = Field(
        None,
        description=(
            "Optional model override for LLM classification (e.g. from env LOW_COST_MODEL). "
            "When empty, the system uses the LLM connector default model."
        ),
    )
    llm_temperature: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for LLM classification (prefer 0 for stable routing).",
    )


class DeepSearchServiceConfig(AbstractConfig):
    """Application-layer builder that assembles all DeepSearch components."""

    type: Literal["deepsearch_service"] = "deepsearch_service"
    planner: PlannerRuntimeConfig
    graph_adapter: GraphAdapterConfig
    graph_reasoning: GraphReasoningStrategyConfig
    multi_agent: MultiAgentConfig
    reporter: ReporterConfig
    tool_manager: ToolManagerConfig
    tool_budget: ToolBudgetConfig
    quality_loop: QualityLoopConfig = Field(default_factory=QualityLoopConfig)
    deterministic_routing: DeterministicRoutingConfig = Field(default_factory=DeterministicRoutingConfig)
    mcp_client: Optional[MCPClientConfig] = Field(
        default=None, description="Optional MCP client config used for remote tools."
    )
    llm: Optional[OpenAIChatConfig] = Field(
        default=None,
        description="Optional LLM config shared across planner reasoning and graph/tools.",
    )
    artifacts: ArtifactsConfig = Field(
        default_factory=ArtifactsConfig,
        description="Artifact persistence settings (manifest/dev/public).",
    )
    telemetry_enabled: bool = Field(
        True,
        description="Enable the built-in telemetry logger to surface tool events.",
    )

    def build(self) -> DeepSearchService:
        llm_connector = self._build_llm_connector()
        adapter = self.graph_adapter.build()
        try:
            adapter_meta = adapter.metadata()
        except Exception:
            adapter_meta = None
        mcp_client = self._build_mcp_client()
        telemetry_client = self._build_telemetry_client() if self._resolve_telemetry_flag() else None
        tool_hint_registry = ToolHintRegistry()
        tool_manager = self._build_tool_manager(
            llm_connector=llm_connector,
            mcp_client=mcp_client,
            telemetry_client=telemetry_client,
            tool_hint_registry=tool_hint_registry,
        )
        adapter_disabled = disabled_tools_for_adapter(adapter_meta) if adapter_meta is not None else set()
        tool_hint_registry.set_disabled_tools(
            merge_disabled_tools(tool_hint_registry.get_disabled_tool_names(), adapter_disabled)
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
        graph_store = self._resolve_graph_store(adapter)
        reporter = DeepSearchReporter(
            template_store=None,
            config=self.reporter.model_dump(),
            llm_connector=llm_connector,
            graph_store=graph_store,
        )
        adapter_meta_payload: Any | None = adapter_meta
        if adapter_meta is not None:
            if hasattr(adapter_meta, "model_dump"):
                try:
                    adapter_meta_payload = adapter_meta.model_dump(exclude_none=True)
                except TypeError:
                    adapter_meta_payload = adapter_meta.model_dump()
            elif is_dataclass(adapter_meta):
                adapter_meta_payload = asdict(adapter_meta)
        return DeepSearchService(
            planner=planner,
            graph_loop=graph_loop,
            reporter=reporter,
            tool_manager=tool_manager,
            config={
                "name": "deepsearch-service",
                "fingerprint": self._fingerprint(),
                "artifact_dir": self.tool_manager.artifact_dir,
                "quality_loop": self.quality_loop.model_dump(),
                "deterministic_routing": self.deterministic_routing.model_dump(),
                "tool_budget": self.tool_budget.model_dump(),
                "artifacts": self.artifacts.model_dump(),
                "adapter": adapter_meta_payload,
                "disabled_tools": sorted(tool_hint_registry.get_disabled_tool_names()),
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
        retriever_path = payload.pop("search_retriever_config_path", None)
        payload["llm_connector"] = payload.get("llm_connector") or llm_connector
        if retriever_path:
            dense, bm25 = _build_search_retrievers(retriever_path)
            _inject_search_retrievers(payload, dense=dense, bm25=bm25)
        if not payload.get("artifact_dir"):
            raise ValueError("tool_manager.artifact_dir is required (no implicit default).")
        return DeepSearchToolManager(
            tool_configs=payload,
            telemetry_client=telemetry_client,
            mcp_client=mcp_client,
            tool_hint_registry=tool_hint_registry,
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
