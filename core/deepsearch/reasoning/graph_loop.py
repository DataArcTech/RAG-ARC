"""Graph-first reasoning loop that orchestrates adapter traversals and tool calls."""
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from encapsulation.data_model.deepsearch import (
    EvidenceChunk,
    GraphQueryContext,
    GraphTraversalRecord,
    PlanSpec,
    ReasoningStepRecord,
    ToolExecutionLog,
    ToolResultPayload,
)
from core.deepsearch.tooling import DeepSearchToolManager
from core.graph_adapter.base import GraphAccessScope, GraphDeepSearchAdapter
from core.graph_adapter.scope_provider import require_scope

from .traversal import GraphTraversalExecutor, GraphTraversalSettings

logger = logging.getLogger(__name__)


class GraphReasoningLoop:
    """Run multi-step graph reasoning using adapters, graph tools, and MCP routing."""

    def __init__(
        self,
        adapter: GraphDeepSearchAdapter,
        llm_connector,
        strategy_config,
        *,
        tool_manager: DeepSearchToolManager | None = None,
    ):
        # adapter: dynamically injected HippoRAG/GraphSearch implementation
        self.adapter = adapter
        # llm_connector: reserved for prompts or LLM backed tools (kept for parity with tool configs)
        self.llm_connector = llm_connector
        # strategy_config: Chain-of-Exploration parameters controlling traversal depth/filters
        self.strategy_config = strategy_config
        self.tool_manager = tool_manager
        self.traversal_settings = self._build_traversal_settings(strategy_config)
        self.traversal_executor = GraphTraversalExecutor(
            adapter=self.adapter,
            settings=self.traversal_settings,
        )
        self._adapter_metadata = self._resolve_adapter_metadata()
        self._think_config = self._build_think_config(strategy_config)
        self._think_run_count = 0

    async def run(
        self,
        question: str,
        plan_steps: Sequence[Dict[str, Any] | PlanSpec],
        *,
        graph_context: Optional[GraphQueryContext] = None,
    ) -> Dict[str, Any]:
        """Execute planner steps, trigger adapter traversals, and return a structured trace."""

        if not question:
            raise ValueError("question must be a non-empty string")
        normalized_steps = self._normalize_plan_steps(plan_steps)
        if not normalized_steps:
            raise ValueError("plan_steps must contain at least one step")

        context = self._prepare_graph_context(graph_context, question, normalized_steps)
        traversals: List[GraphTraversalRecord] = []
        evidences: List[EvidenceChunk] = []
        tool_runs: List[Dict[str, Any]] = []
        think_notes: List[Dict[str, Any]] = []
        pending_external: List[Dict[str, Any]] = []
        final_reasoning: List[ReasoningStepRecord] = []
        completed_internal_steps = 0
        if any(entry["run_with_adapter"] for entry in normalized_steps):
            await self.traversal_executor.prepare(context)

        for entry in normalized_steps:
            spec = entry["spec"]
            record = self._empty_record(spec)

            if not entry["enabled"]:
                record.status = "skipped"
                record.diagnostics.setdefault("reason", "disabled_by_planner")
                final_reasoning.append(record)
                continue

            if entry["requires_external"]:
                record.status = "pending_external"
                record.diagnostics.setdefault("reason", "requires_external_channel")
                pending_external.append(self._pending_external_payload(entry))
                final_reasoning.append(record)
                continue

            if entry["run_with_adapter"]:
                traversal_record, reasoning_record, new_evidences = await self.traversal_executor.run_step(
                    spec,
                    context,
                    tool_args=entry["tool_args"],
                )
                reasoning_record.diagnostics.setdefault("tool", entry["tool"] or "graph_adapter.query")
                if traversal_record:
                    traversals.append(traversal_record)
                evidences.extend(new_evidences)
                final_reasoning.append(reasoning_record)
                if reasoning_record.status == "done":
                    completed_internal_steps += 1
                    coverage_metrics = self._coverage_snapshot(
                        evidence_count=len(evidences),
                        source_labels=[chunk.source for chunk in evidences],
                        completed_steps=completed_internal_steps,
                        total_steps=len(normalized_steps),
                    )
                    await self._maybe_run_periodic_think(
                        question=question,
                        context=context,
                        evidences=evidences,
                        reasoning_log=final_reasoning,
                        tool_runs=tool_runs,
                        think_notes=think_notes,
                        coverage_metrics=coverage_metrics,
                        completed_steps=completed_internal_steps,
                        total_steps=len(normalized_steps),
                    )
                continue

            if entry["should_invoke_tool"] and not self.tool_manager:
                record.status = "skipped"
                record.diagnostics.setdefault("reason", "tool_manager_disabled")
                record.diagnostics.setdefault("tool", entry["tool"])
                final_reasoning.append(record)
                continue

            if entry["should_invoke_tool"]:
                coverage_hint = self._coverage_snapshot(
                    evidence_count=len(evidences),
                    source_labels=[chunk.source for chunk in evidences],
                    completed_steps=completed_internal_steps,
                    total_steps=len(normalized_steps),
                )
                try:
                    result = await self._invoke_tool(
                        tool_name=entry["tool"],
                        step=entry,
                        context=context,
                        question=question,
                        accumulated_evidence=evidences,
                        coverage_hint=coverage_hint,
                    )
                except Exception as exc:  # pragma: no cover - defensive guardrails
                    logger.warning("Tool %s failed for %s: %s", entry["tool"], spec.step_id, exc)
                    record.status = "failed"
                    record.diagnostics.setdefault("error", str(exc))
                    final_reasoning.append(record)
                    continue

                evidences.extend(result.evidences)
                record.status = "done"
                record.output_summary = result.summary
                record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
                record.diagnostics.setdefault("tool", entry["tool"])
                log_entry = ToolExecutionLog(
                    tool_name=result.tool_name,
                    server_name=None,
                    arguments_snapshot=entry["tool_args"],
                    response_excerpt=result.summary[:200] if result.summary else None,
                    latency_ms=None,
                    graph_context=context,
                    extra={
                        "channel": spec.channel,
                        "profile": result.profile,
                        "determinism": result.determinism,
                    },
                )
                record.tool_logs.append(log_entry)
                tool_runs.append(
                    {
                        "plan_step_id": spec.step_id,
                        "tool_name": result.tool_name,
                        "channel": spec.channel,
                        "result": result.model_dump(),
                    }
                )
                for note in result.think_notes:
                    think_notes.append(note.model_dump(exclude_none=True))
                completed_internal_steps += 1
                coverage_hint = self._coverage_snapshot(
                    evidence_count=len(evidences),
                    source_labels=[chunk.source for chunk in evidences],
                    completed_steps=completed_internal_steps,
                    total_steps=len(normalized_steps),
                )
                await self._maybe_run_periodic_think(
                    question=question,
                    context=context,
                    evidences=evidences,
                    reasoning_log=final_reasoning,
                    tool_runs=tool_runs,
                    think_notes=think_notes,
                    coverage_metrics=coverage_hint,
                    completed_steps=completed_internal_steps,
                    total_steps=len(normalized_steps),
                )
                final_reasoning.append(record)
                continue

            record.status = "skipped"
            record.diagnostics.setdefault("reason", "no_tool_available")
            final_reasoning.append(record)

        coverage_metrics = self._coverage_snapshot(
            evidence_count=len(evidences),
            source_labels=[chunk.source for chunk in evidences],
            completed_steps=completed_internal_steps,
            total_steps=len(normalized_steps),
        )

        return {
            "question": question,
            "graph_context": context.model_dump(exclude_none=True),
            "adapter_metadata": self._adapter_metadata,
            "plan_steps": [entry["spec"].model_dump() for entry in normalized_steps],
            "graph_traversals": [record.model_dump() for record in traversals],
            "reasoning_steps": [record.model_dump() for record in final_reasoning],
            "evidences": [chunk.model_dump() for chunk in evidences],
            "tool_results": tool_runs,
            "pending_external": pending_external,
            "think_notes": think_notes,
            "coverage_metrics": coverage_metrics,
        }

    # ------------------------------------------------------------------
    def _build_traversal_settings(self, config) -> GraphTraversalSettings:
        if isinstance(config, GraphTraversalSettings):
            return config
        if hasattr(config, "model_dump"):
            payload = config.model_dump()
        elif isinstance(config, dict):
            payload = dict(config)
        else:
            payload = getattr(config, "__dict__", {})
        allowed = {
            key: payload.get(key)
            for key in ("strategy_name", "allow_semantic_channel", "chain_depth")
            if key in payload and payload.get(key) is not None
        }
        return GraphTraversalSettings(**allowed)

    def _build_think_config(self, config) -> Dict[str, Any]:
        defaults = {
            "tool_name": "graph.think",
            "cadence": 0,
            "min_coverage": 0.75,
        }
        if not config:
            return defaults
        if hasattr(config, "model_dump"):
            try:
                payload = config.model_dump()
            except TypeError:
                payload = config.model_dump(exclude_none=True)
        elif isinstance(config, dict):
            payload = dict(config)
        else:
            payload = getattr(config, "__dict__", {})
        think_section = payload.get("think") if isinstance(payload.get("think"), dict) else {}
        tool_name = think_section.get("tool_name") or payload.get("think_tool_name") or defaults["tool_name"]
        cadence = think_section.get("every_n_steps") or payload.get("think_every_n_steps") or defaults["cadence"]
        min_coverage = (
            think_section.get("min_coverage")
            or payload.get("think_min_coverage")
            or defaults["min_coverage"]
        )
        cfg = {
            "tool_name": str(tool_name).strip() if tool_name else "",
            "cadence": int(cadence) if cadence is not None else 0,
            "min_coverage": float(min_coverage) if min_coverage is not None else defaults["min_coverage"],
        }
        if cfg["cadence"] < 0:
            cfg["cadence"] = 0
        return cfg

    def _resolve_adapter_metadata(self) -> Dict[str, Any]:
        metadata = getattr(self.adapter, "metadata", None)
        if callable(metadata):
            raw = metadata()
        else:
            raw = None
        if raw is None:
            return {}
        if hasattr(raw, "model_dump"):
            return raw.model_dump(exclude_none=True)
        if is_dataclass(raw):
            return asdict(raw)
        return {key: value for key, value in raw.__dict__.items() if not key.startswith("_")}

    def _build_graph_context(
        self,
        question: str,
        *,
        access_scope: GraphAccessScope,
        seed_entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphQueryContext:
        adapter_name = self._adapter_metadata.get("adapter_name") or getattr(self.adapter, "name", "graph_adapter")
        context_metadata = {
            "strategy": self.traversal_settings.strategy_name,
            "chain_depth": self.traversal_settings.chain_depth,
        }
        if metadata:
            context_metadata.update(metadata)
        return GraphQueryContext(
            adapter_name=adapter_name,
            question=question,
            metadata=context_metadata,
            seed_entities=list(seed_entities or []),
            access_scope=access_scope,
        )

    def _prepare_graph_context(
        self,
        context: Optional[GraphQueryContext],
        question: str,
        normalized_steps: Sequence[Dict[str, Any]],
    ) -> GraphQueryContext:
        if context is None:
            scope = require_scope()
            context = self._build_graph_context(question, access_scope=scope)
        else:
            # copy existing context to avoid mutating caller-owned instances
            try:
                context = context.model_copy()
            except AttributeError:
                context = GraphQueryContext(**context.model_dump())

        scope = context.resolve_scope()
        if scope is None:
            scope = require_scope()
            context = context.model_copy(update={"access_scope": scope})

        seed_entities = self._collect_seed_entities(context, normalized_steps)
        scheduler_hints = self._collect_scheduler_hints(normalized_steps)
        metadata = dict(context.metadata or {})
        if scheduler_hints:
            planner_hints = metadata.setdefault("planner_hints", {})
            planner_hints.setdefault("per_step", {}).update(scheduler_hints)
        if seed_entities:
            metadata.setdefault("planner_hints", {}).setdefault("seed_entities", seed_entities)
        return context.model_copy(update={"seed_entities": seed_entities, "metadata": metadata})

    def _normalize_plan_steps(self, plan_steps: Sequence[Dict[str, Any] | PlanSpec]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, raw in enumerate(plan_steps):
            if isinstance(raw, PlanSpec):
                spec = raw
                raw_dict = {
                    "step_id": spec.step_id,
                    "description": spec.description,
                    "channel": spec.channel,
                    "metadata": spec.metadata,
                }
            else:
                metadata = dict(raw.get("metadata") or {})
                spec = PlanSpec(
                    step_id=str(raw.get("step_id") or f"plan_{idx+1:02d}"),
                    description=str(raw.get("description") or ""),
                    channel=str(raw.get("channel") or "graph"),
                    metadata=metadata,
                )
                raw_dict = dict(raw)

            tool = raw_dict.get("tool") or spec.metadata.get("tool")
            tool_args = raw_dict.get("tool_args") or spec.metadata.get("tool_args") or {}
            enabled = raw_dict.get("enabled")
            requires_external = raw_dict.get("requires_external") or spec.metadata.get("requires_external")
            channel = (spec.channel or "graph").lower()
            requires_external = bool(requires_external or channel == "web")
            run_with_adapter = bool(
                not requires_external
                and (channel == "graph")
                and (not tool or tool == "graph_adapter.query")
            )
            should_invoke_tool = bool(
                not requires_external
                and channel in {"graph", "text"}
                and tool
                and tool != "graph_adapter.query"
            )
            normalized.append(
                {
                    "spec": spec,
                    "tool": tool,
                    "tool_args": tool_args if isinstance(tool_args, dict) else {},
                    "enabled": True if enabled is None else bool(enabled),
                    "requires_external": requires_external,
                    "run_with_adapter": run_with_adapter,
                    "should_invoke_tool": should_invoke_tool,
                    "channel": channel,
                    "raw": raw_dict,
                }
            )
        return normalized

    async def _invoke_tool(
        self,
        *,
        tool_name: str,
        step: Dict[str, Any],
        context: GraphQueryContext,
        question: str,
        accumulated_evidence: List[EvidenceChunk],
        coverage_hint: Dict[str, Any],
    ) -> ToolResultPayload:
        if not self.tool_manager:
            raise RuntimeError("GraphReasoningLoop cannot invoke tools without a tool manager")
        payload = self._build_tool_payload(
            plan_step_id=step["spec"].step_id,
            question=question,
            context=context,
            evidences=accumulated_evidence,
            coverage_hint=coverage_hint,
            extra=step["tool_args"],
        )
        return await self.tool_manager.invoke(tool_name, payload=payload)

    def _build_tool_payload(
        self,
        *,
        plan_step_id: str,
        question: str,
        context: GraphQueryContext,
        evidences: List[EvidenceChunk],
        coverage_hint: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "plan_step": plan_step_id,
            "context_evidences": [chunk.model_dump() for chunk in evidences],
            "adapter": self.adapter,
            "access_scope": context.resolve_scope(),
            "extra": dict(extra or {}),
            "graph_context": context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_hint,
        }

    @staticmethod
    def _empty_record(spec: PlanSpec) -> ReasoningStepRecord:
        return ReasoningStepRecord(
            step_id=spec.step_id,
            description=spec.description,
            channel=spec.channel,
            status="pending",
        )

    @staticmethod
    def _pending_external_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
        spec: PlanSpec = entry["spec"]
        return {
            "step_id": spec.step_id,
            "description": spec.description,
            "channel": spec.channel,
            "tool": entry["tool"],
            "tool_args": entry["tool_args"],
            "metadata": spec.metadata,
        }

    @staticmethod
    def _coverage_snapshot(
        *,
        evidence_count: int,
        source_labels: Optional[Sequence[str]],
        completed_steps: int,
        total_steps: int,
    ) -> Dict[str, Any]:
        unique_sources = len({label for label in source_labels or [] if label})
        coverage_ratio = (completed_steps / total_steps) if total_steps else 0.0
        return {
            "evidence_count": evidence_count,
            "unique_source_count": unique_sources,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "coverage_ratio": round(coverage_ratio, 3),
        }

    async def _maybe_run_periodic_think(
        self,
        *,
        question: str,
        context: GraphQueryContext,
        evidences: List[EvidenceChunk],
        reasoning_log: List[ReasoningStepRecord],
        tool_runs: List[Dict[str, Any]],
        think_notes: List[Dict[str, Any]],
        coverage_metrics: Dict[str, Any],
        completed_steps: int,
        total_steps: int,
    ) -> None:
        if not self._should_run_think(completed_steps, coverage_metrics):
            return
        if not self.tool_manager or not self._think_config["tool_name"]:
            return
        self._think_run_count += 1
        think_step_id = f"think_auto_{self._think_run_count:02d}"
        record = ReasoningStepRecord(
            step_id=think_step_id,
            description="Periodic think checkpoint",
            channel="graph",
            status="running",
        )
        reasoning_log.append(record)
        payload = self._build_tool_payload(
            plan_step_id=think_step_id,
            question=question,
            context=context,
            evidences=evidences,
            coverage_hint=coverage_metrics,
            extra={
                "trigger": "periodic_think",
                "completed_steps": completed_steps,
                "total_steps": total_steps,
            },
        )
        try:
            result = await self.tool_manager.invoke(self._think_config["tool_name"], payload=payload)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            record.status = "failed"
            record.diagnostics.setdefault("error", str(exc))
            record.diagnostics.setdefault("reason", "periodic_think")
            return

        evidences.extend(result.evidences)
        record.status = "done"
        record.output_summary = result.summary
        record.produced_evidence_ids = [chunk.chunk_id for chunk in result.evidences]
        record.diagnostics.setdefault("reason", "periodic_think")
        log_entry = ToolExecutionLog(
            tool_name=result.tool_name,
            server_name=None,
            arguments_snapshot={"trigger": "periodic_think"},
            response_excerpt=result.summary[:200] if result.summary else None,
            latency_ms=None,
            graph_context=context,
            extra={
                "channel": "graph",
                "profile": result.profile,
                "determinism": result.determinism,
                "trigger": "periodic_think",
            },
        )
        record.tool_logs.append(log_entry)
        tool_runs.append(
            {
                "plan_step_id": think_step_id,
                "tool_name": result.tool_name,
                "channel": "graph",
                "result": result.model_dump(),
            }
        )
        for note in result.think_notes:
            think_notes.append(note.model_dump(exclude_none=True))

    def _should_run_think(self, completed_steps: int, coverage_metrics: Dict[str, Any]) -> bool:
        cadence = self._think_config["cadence"]
        if cadence <= 0 or completed_steps <= 0 or (completed_steps % cadence) != 0:
            return False
        coverage_ratio = coverage_metrics.get("coverage_ratio") or 0.0
        return coverage_ratio < self._think_config["min_coverage"]

    def _collect_seed_entities(
        self,
        context: GraphQueryContext,
        steps: Sequence[Dict[str, Any]],
    ) -> List[str]:
        seeds: Set[str] = set()

        def _ingest(value: Any) -> None:
            if not value:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _ingest(item)
                return
            token = str(value).strip()
            if token:
                seeds.add(token)

        for existing in getattr(context, "seed_entities", []) or []:
            _ingest(existing)

        metadata_payload = getattr(context, "metadata", {}) or {}
        _ingest(metadata_payload.get("seed_entities"))
        planner_hint = metadata_payload.get("planner_hints") or {}
        _ingest(planner_hint.get("seed_entities"))

        for entry in steps:
            spec: PlanSpec = entry["spec"]
            _ingest(spec.metadata.get("seed_entities"))
            tool_args = entry.get("tool_args") or {}
            _ingest(tool_args.get("seed_entities"))
            _ingest(spec.metadata.get("seed_nodes"))
            _ingest(tool_args.get("seed_nodes"))

        return list(seeds)

    @staticmethod
    def _collect_scheduler_hints(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        hints: Dict[str, Any] = {}
        for entry in steps:
            spec: PlanSpec = entry["spec"]
            collected: Dict[str, Any] = {}
            metadata = spec.metadata or {}
            for key in ("scheduler", "scheduler_hint", "priority", "tool_profile"):
                if metadata.get(key) is not None:
                    collected[key] = metadata[key]
            tool_args = entry.get("tool_args") or {}
            if tool_args.get("scheduler") is not None:
                collected["scheduler"] = tool_args["scheduler"]
            if collected:
                hints[spec.step_id] = collected
        return hints
