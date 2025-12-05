"""Pydantic models shared across DeepSearch services, API, and CLI layers."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlanSpec(BaseModel):
    """Single step emitted by the planner and replayed by the reasoning loop."""

    step_id: str = Field(..., description="Deterministic identifier shared with execution logs")
    description: str = Field(..., description="Human-readable plan instruction")
    channel: str = Field(..., description="Channel label such as graph/text/web")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional parameters for executors")


class EvidenceChunk(BaseModel):
    """Unified evidence representation shared across graph and external channels."""

    chunk_id: str = Field(..., description="Stable ID so reasoning steps can reference this chunk")
    source: str = Field(..., description="Logical origin of the evidence such as hipporag/graphsearch/web")
    content: str = Field(..., description="Evidence text or compressed summary")
    score: Optional[float] = Field(None, description="Confidence or relevance score assigned by retrievers")
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Node/edge identifiers, URLs, or other context metadata"
    )


class DeepSearchResult(BaseModel):
    """Response returned to API/CLI callers."""

    answer: str
    evidences: List[EvidenceChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphQueryContext(BaseModel):
    """Carries graph-specific metadata that needs to reach adapters or external tools."""

    adapter_name: str = Field(..., description="Registered GraphAdapter name driving this request")
    owner_id: Optional[str] = Field(None, description="User identifier for graph isolation")
    question: Optional[str] = Field(None, description="Original natural language question")
    seed_entities: List[str] = Field(default_factory=list, description="Seed entities highlighted during traversal")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra context such as hop limits or filters")


class ToolExecutionLog(BaseModel):
    """Structured snapshot of a tool invocation, used for telemetry and audits."""

    tool_name: str = Field(..., description="Tool identifier reported by the MCP server")
    server_name: Optional[str] = Field(None, description="Logical server name configured by DeepSearch")
    arguments_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Arguments sent to the tool")
    response_excerpt: Optional[str] = Field(None, description="Short textual preview of the response")
    latency_ms: Optional[int] = Field(None, description="Measured latency in milliseconds")
    graph_context: Optional[GraphQueryContext] = Field(
        None, description="Graph context injected while invoking the tool"
    )
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional diagnostics or metrics")


class GapDetectionResult(BaseModel):
    """Represents coverage analysis used to decide whether external channels must run."""

    coverage_score: float = Field(..., description="Estimated portion of the question covered by current evidence")
    confidence_score: float = Field(..., description="Model-estimated confidence in the intermediate answer")
    should_trigger_external: bool = Field(..., description="Whether to invoke web/code channels to fill gaps")
    reason: Optional[str] = Field(None, description="Human-readable reason for the decision")
    missing_topics: List[str] = Field(default_factory=list, description="Topics still uncovered by the graph search")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Raw metrics for logging and debugging")


class GraphTraversalRecord(BaseModel):
    """Captures one traversal attempt executed by the GraphReasoningLoop."""

    step_id: str = Field(..., description="Identifier matching the originating PlanStep")
    strategy: str = Field(..., description="Traversal strategy label (ppr_chain, semantic_chain, etc.)")
    hop_count: int = Field(..., description="Number of hops explored during this traversal")
    visited_nodes: List[str] = Field(default_factory=list, description="Node identifiers touched during traversal")
    visited_edges: List[str] = Field(default_factory=list, description="Edge identifiers touched during traversal")
    seed_entities: List[str] = Field(default_factory=list, description="Seed entities used as graph entry points")
    retrieved_chunks: List[str] = Field(default_factory=list, description="Chunk identifiers surfaced by this traversal")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Adapter-specific diagnostics and scores")


class ReasoningStepRecord(BaseModel):
    """Detailed reasoning step emitted by Planner/GraphReasoningLoop."""

    step_id: str = Field(..., description="Unique identifier for correlating plan and execution")
    description: str = Field(..., description="Human-readable explanation of the step intent")
    channel: str = Field(..., description="Channel type such as graph/text/web")
    status: str = Field(..., description="Execution status (queued/running/done/skipped)")
    evidence_ids: List[str] = Field(default_factory=list, description="EvidenceChunk IDs consumed in this step")
    produced_evidence_ids: List[str] = Field(default_factory=list, description="EvidenceChunk IDs created by this step")
    tool_logs: List[ToolExecutionLog] = Field(default_factory=list, description="Tool invocations triggered here")
    output_summary: Optional[str] = Field(None, description="Short natural language summary of the step result")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="LLM cost, latency, score, etc.")


class DeepSearchTrace(BaseModel):
    """Top-level trace tying together plan, execution, evidence, and decisions."""

    plan_steps: List[PlanSpec] = Field(default_factory=list, description="Plan emitted by the planner")
    reasoning_steps: List[ReasoningStepRecord] = Field(default_factory=list, description="Executed reasoning steps")
    traversals: List[GraphTraversalRecord] = Field(default_factory=list, description="Graph traversal records")
    evidences: List[EvidenceChunk] = Field(default_factory=list, description="All evidence chunks accumulated")
    gap_result: Optional[GapDetectionResult] = Field(None, description="Gap detection outcome for this run")
    external_calls: List[ToolExecutionLog] = Field(default_factory=list, description="External tool invocations")
    final_answer: Optional[str] = Field(None, description="LLM answer before formatting")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Run-level telemetry such as config fingerprint")
