"""Report composer for DeepSearch.

This reporter converts DeepSearch execution traces into a readable, end-user report by
prompting an LLM with the collected evidence, highlights, and execution metadata.
"""
import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk

from config.output_limits import (
    DEEPSEARCH_GRAPH_EDGE_LIMIT,
    DEEPSEARCH_TOP_CHUNKS,
    DEEPSEARCH_TOP_SEED_ENTITIES,
)
from core.deepsearch.report.consistency_checker import ConsistencyChecker
from core.deepsearch.report.citation_agent import CitationAgent
from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter, render_markdown_from_structured
from core.presentation.evidence import build_deepsearch_evidence


def _resolve_consistency_check_flag(config_value: bool) -> bool:
    """Resolve consistency check flag with environment variable override.

    Environment variable DEEPSEARCH_CONSISTENCY_CHECK takes precedence.
    Valid truthy values: 1, true, yes, on
    Valid falsy values: 0, false, no, off
    """
    env_value = os.getenv("DEEPSEARCH_CONSISTENCY_CHECK")
    if env_value is None:
        return config_value
    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return config_value


def _resolve_parallel_sections_flag(config_value: bool) -> bool:
    """Resolve parallel sections flag with environment variable override.

    Environment variable DEEPSEARCH_PARALLEL_SECTIONS takes precedence.
    Valid truthy values: 1, true, yes, on
    Valid falsy values: 0, false, no, off
    """
    env_value = os.getenv("DEEPSEARCH_PARALLEL_SECTIONS")
    if env_value is None:
        return config_value
    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return config_value


class DeepSearchReporter:
    """Generate a structured report from DeepSearch traces."""

    STRUCTURED_REPORT_VERSION = "2.0"

    def __init__(
        self,
        template_store,
        config,
        *,
        llm_connector: Any | None = None,
        graph_store: Any | None = None,
    ):
        self.template_store = template_store
        self.config = config or {}
        self.llm_connector = llm_connector
        self.graph_store = graph_store

        self.max_highlights = int(self.config.get("max_highlights", 6))
        self.max_evidence_items = DEEPSEARCH_TOP_CHUNKS
        self.report_temperature = float(self.config.get("report_temperature", 0.2))
        self.report_max_evidence_chars = int(self.config.get("report_max_evidence_chars", 900))
        self.report_max_graph_chain_items = DEEPSEARCH_GRAPH_EDGE_LIMIT
        self.enable_llm_report = bool(self.config.get("enable_llm_report", True))
        self.enable_consistency_check = _resolve_consistency_check_flag(
            bool(self.config.get("enable_consistency_check", True))
        )
        self.consistency_temperature = float(self.config.get("consistency_temperature", 0.0))
        self.enable_citation_agent = bool(self.config.get("enable_citation_agent", True))
        self.parallel_sections = _resolve_parallel_sections_flag(
            bool(self.config.get("parallel_sections", False))
        )
        self.max_parallel_sections = int(self.config.get("max_parallel_sections", 4))

    async def compose(
        self,
        reasoning_trace: Dict[str, Any],
        external_evidence: Optional[Iterable[Dict[str, Any] | EvidenceChunk]] = None,
    ) -> Dict[str, Any]:
        """Compose a report payload for downstream clients (API/CLI/MCP)."""

        if self.enable_llm_report and self.llm_connector is None:
            raise RuntimeError("LLM connector is required for LLM report generation")

        trace = reasoning_trace or {}
        question = (trace.get("question") or "").strip()
        if not question:
            raise ValueError("DeepSearch reasoning trace is missing 'question'")

        evidences = self._merge_evidences(trace.get("evidences"), external_evidence)
        reasoning_steps = trace.get("reasoning_steps") or []
        highlights = self._build_highlights(reasoning_steps)
        coverage_metrics = trace.get("coverage_metrics") or {}
        gap_result = trace.get("gap_result") or {}
        request_context = self._extract_request_context(trace)

        metadata: Dict[str, Any] = {
            "question": question,
            "adapter_metadata": trace.get("adapter_metadata") or {},
            "graph_summary": self._graph_summary(trace.get("graph_traversals") or []),
            "plan": {
                "steps": trace.get("plan_steps") or [],
                "completed": sum(1 for step in (reasoning_steps or []) if step.get("status") == "done"),
            },
            "reasoning_steps": reasoning_steps,
            "think_notes": trace.get("think_notes") or [],
            "tool_results": trace.get("tool_results") or [],
            "coverage_metrics": coverage_metrics,
            "gap_result": gap_result,
            "pending_external": trace.get("pending_external") or [],
            "parallel_thinking_runs": int(self.config.get("parallel_thinking_runs", 1)),
            "report_profile": {
                "include_graph_viz": bool(self.config.get("include_graph_viz", False)),
                "enable_custom_summary": bool(self.config.get("enable_custom_summary", False)),
            },
        }
        if request_context:
            metadata["request_context"] = request_context
        if metadata["report_profile"]["include_graph_viz"]:
            metadata["graph_visualization"] = trace.get("graph_traversals") or []

        context = self._build_llm_context(
            trace=trace,
            highlights=highlights,
            evidences=evidences,
            coverage=coverage_metrics,
            gap_result=gap_result,
            request_context=request_context,
        )
        if not self.enable_llm_report:
            markdown_text = _render_fallback_markdown(
                question=question,
                final_answer=str(trace.get("final_answer") or "").strip(),
                highlights=highlights,
                evidences=evidences,
            )
            markdown_text = _append_chunk_evidence(
                markdown_text,
                evidences=evidences,
                max_items=self.max_evidence_items,
            )
            markdown_text = _append_graph_appendix(
                markdown_text,
                graph_evidence=context.get("graph_evidence") or {},
            )
            structured_report = {
                "format_version": self.STRUCTURED_REPORT_VERSION,
                "title": question,
                "summary": str(trace.get("final_answer") or "").strip(),
                "sections": [],
                "limitations": [],
                "next_steps": [],
                "citations": [],
                "graph_evidence": context.get("graph_evidence") or {},
                "text": markdown_text,
                "context": request_context or {},
                "generation": {"mode": "fallback"},
            }
            if self.enable_citation_agent:
                structured_report, audit = CitationAgent().process(
                    structured_report=structured_report,
                    evidences=evidences,
                    graph_evidence=context.get("graph_evidence") or {},
                    max_chunk_index_items=self.max_evidence_items,
                )
                metadata["citation_audit"] = audit
            metadata["structured_report"] = {key: structured_report[key] for key in structured_report if key != "text"}
            return {
                "question": question,
                "answer": markdown_text,
                "evidences": evidences,
                "highlights": highlights,
                "structured_report": structured_report,
                "metadata": metadata,
            }

        writer = DeepSearchLLMReportWriter(
            self.llm_connector,
            temperature=self.report_temperature,
            max_evidence_items=self.max_evidence_items,
            max_evidence_chars=self.report_max_evidence_chars,
            max_graph_chain_items=self.report_max_graph_chain_items,
            parallel_sections=self.parallel_sections,
            max_parallel_sections=self.max_parallel_sections,
        )
        outline = await writer.build_outline(question=question, context=context)
        if self.parallel_sections and len(outline) > 2:
            structured_llm = await writer.write_report_parallel(question=question, outline=outline, context=context)
        else:
            structured_llm = await writer.write_report(question=question, outline=outline, context=context)
        if self.enable_citation_agent:
            structured_llm, audit = CitationAgent().process(
                structured_report=structured_llm,
                evidences=evidences,
                graph_evidence=context.get("graph_evidence") or {},
                max_chunk_index_items=self.max_evidence_items,
            )
            metadata["citation_audit"] = audit
        markdown_text = render_markdown_from_structured(structured_llm)
        markdown_text = _append_chunk_evidence(
            markdown_text,
            evidences=evidences,
            max_items=self.max_evidence_items,
        )
        markdown_text = _append_graph_appendix(
            markdown_text,
            graph_evidence=context.get("graph_evidence") or {},
        )

        structured_report = {
            "format_version": self.STRUCTURED_REPORT_VERSION,
            "title": structured_llm.get("title") or question,
            "summary": structured_llm.get("summary") or "",
            "sections": structured_llm.get("sections") or [],
            "limitations": structured_llm.get("limitations") or [],
            "next_steps": structured_llm.get("next_steps") or [],
            "citations": structured_llm.get("citations") or [],
            "evidence_index": structured_llm.get("evidence_index") or [],
            "graph_evidence": context.get("graph_evidence") or {},
            "text": markdown_text,
            "context": request_context or {},
            "generation": {"mode": "llm"},
        }

        if self.enable_consistency_check and evidences:
            checker = ConsistencyChecker(
                self.llm_connector,
                temperature=self.consistency_temperature,
                max_retries=int(self.config.get("consistency_max_retries", 2)),
            )
            result = await checker.check(
                report_markdown=markdown_text,
                evidences=evidences,
                structured_report=structured_report,
                max_evidence_items=self.max_evidence_items,
                max_evidence_chars=self.report_max_evidence_chars,
            )
            structured_report["consistency_check"] = result.model_dump()
            if not result.is_consistent:
                metadata["quality_warnings"] = [issue.model_dump() for issue in result.issues]

        metadata["structured_report"] = {key: structured_report[key] for key in structured_report if key != "text"}

        return {
            "question": question,
            "answer": markdown_text,
            "evidences": evidences,
            "highlights": highlights,
            "structured_report": structured_report,
            "metadata": metadata,
        }

    def _build_llm_context(
        self,
        *,
        trace: Dict[str, Any],
        highlights: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        gap_result: Dict[str, Any],
        request_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan_steps = trace.get("plan_steps") or []
        reasoning_steps = trace.get("reasoning_steps") or []
        tool_results = trace.get("tool_results") or []
        pending_external = trace.get("pending_external") or []

        methodology = {
            "plan_steps": plan_steps,
            "reasoning_steps": [
                {
                    "step_id": step.get("step_id"),
                    "description": step.get("description"),
                    "channel": step.get("channel"),
                    "status": step.get("status"),
                    "output_summary": step.get("output_summary"),
                    "tool": step.get("tool")
                    or (step.get("metadata") or {}).get("tool")
                    or (step.get("diagnostics") or {}).get("tool"),
                }
                for step in reasoning_steps
                if isinstance(step, dict)
            ],
            "tool_results": [
                {
                    "plan_step_id": entry.get("plan_step_id"),
                    "tool_name": entry.get("tool_name"),
                    "channel": entry.get("channel"),
                    "summary": (entry.get("result") or {}).get("summary") if isinstance(entry.get("result"), dict) else None,
                    "diagnostics": (entry.get("result") or {}).get("diagnostics") if isinstance(entry.get("result"), dict) else None,
                }
                for entry in tool_results
                if isinstance(entry, dict)
            ],
        }

        coverage_bundle = {
            "coverage_metrics": coverage,
            "gap_result": gap_result,
            "pending_external": pending_external,
        }

        return {
            "question": trace.get("question") or "",
            "final_answer": trace.get("final_answer") or "",
            "highlights": highlights,
            "evidences": evidences,
            "graph_chain": (trace.get("graph_chain") or [])[: self.report_max_graph_chain_items],
            "graph_evidence": self._build_graph_evidence(trace, evidences),
            "methodology": methodology,
            "coverage": coverage_bundle,
            "request_context": request_context,
        }

    def _merge_evidences(
        self,
        internal: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
        external: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in self._iter_evidences(internal, external):
            chunk_id = chunk.get("chunk_id") or self._hash_content(chunk)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk.setdefault("chunk_id", chunk_id)
            merged.append(chunk)
            if self.max_evidence_items is not None and self.max_evidence_items > 0 and len(merged) >= self.max_evidence_items:
                break
        return merged

    def _iter_evidences(
        self,
        internal: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
        external: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
    ) -> Iterable[Dict[str, Any]]:
        for payload in (internal or []):
            normalized = self._normalize_evidence(payload)
            if normalized:
                yield normalized
        for payload in (external or []):
            normalized = self._normalize_evidence(payload)
            if normalized:
                yield normalized

    @staticmethod
    def _normalize_evidence(payload: Dict[str, Any] | EvidenceChunk | None) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if isinstance(payload, EvidenceChunk):
            return payload.model_dump()
        if isinstance(payload, dict) and payload.get("content"):
            return dict(payload)
        return None

    @staticmethod
    def _hash_content(chunk: Dict[str, Any]) -> str:
        digest = hashlib.sha256((chunk.get("source", "") + "::" + chunk.get("content", "")).encode("utf-8"))
        return f"anon-{digest.hexdigest()[:12]}"

    def _extract_request_context(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        graph_context = trace.get("graph_context") or {}
        metadata = graph_context.get("metadata") or {}
        request_metadata = metadata.get("request_metadata")
        context: Dict[str, Any] = {}
        if isinstance(request_metadata, dict):
            for key, value in request_metadata.items():
                if value is None:
                    continue
                context[str(key)] = str(value)
        access_scope = graph_context.get("access_scope") or {}
        scope_id = access_scope.get("scope_id")
        if scope_id and "scope_id" not in context:
            context["scope_id"] = str(scope_id)
        return context

    def _build_highlights(self, reasoning_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        highlights: List[Dict[str, Any]] = []
        for step in reasoning_steps:
            if step.get("status") != "done":
                continue
            summary = (step.get("output_summary") or "").strip()
            if not summary:
                continue
            highlights.append(
                {
                    "step_id": step.get("step_id"),
                    "description": step.get("description"),
                    "summary": summary,
                    "evidence_ids": step.get("produced_evidence_ids") or [],
                }
            )
            if self.max_highlights > 0 and len(highlights) >= self.max_highlights:
                break
        return highlights

    @staticmethod
    def _graph_summary(traversals: List[Dict[str, Any]]) -> Dict[str, Any]:
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        for record in traversals:
            for node in record.get("visited_nodes", []) or []:
                if node:
                    node_ids.add(str(node))
            for edge in record.get("visited_edges", []) or []:
                if edge:
                    edge_ids.add(str(edge))
        return {
            "traversal_count": len(traversals),
            "unique_nodes": len(node_ids),
            "unique_edges": len(edge_ids),
        }

    def _build_graph_evidence(self, trace: Dict[str, Any], evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a graph evidence bundle (seeds/chain) when possible."""

        payload = {
            "reasoning": {"evidences": evidences},
            "report": {"evidences": evidences},
            "graph_chain": trace.get("graph_chain") or [],
        }
        return build_deepsearch_evidence(payload, chunk_limit=self.max_evidence_items, graph_store=self.graph_store)


def _append_chunk_evidence(
    markdown_text: str,
    *,
    evidences: List[Dict[str, Any]],
    max_items: int | None = None,
    preview_length: int = 100,
) -> str:
    """Append a compact section listing chunk evidence with content preview."""

    if not evidences:
        return markdown_text

    capped = evidences[:max_items] if max_items is not None and max_items > 0 else evidences
    if not capped:
        return markdown_text

    blocks: List[str] = ["", "## Appendix: Chunk Evidence"]
    for entry in capped:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        source = str(entry.get("source") or "").strip()
        content = str(entry.get("content") or "").strip()
        if not chunk_id:
            continue
        preview = content[:preview_length].replace("\n", " ").strip()
        if len(content) > preview_length:
            preview += "..."
        source_tag = f" ({source})" if source else ""
        blocks.append(f"- [{chunk_id}]{source_tag}: {preview}")

    return (markdown_text or "").rstrip() + "\n" + "\n".join(blocks).rstrip() + "\n"


def _append_graph_appendix(markdown_text: str, *, graph_evidence: Dict[str, Any], max_items: int = 40) -> str:
    """Append a compact appendix listing graph evidence artifacts."""

    if not isinstance(graph_evidence, dict):
        return markdown_text

    seeds = list(graph_evidence.get("seed_entities") or [])
    chain = list(graph_evidence.get("graph_chain") or [])
    stats = graph_evidence.get("graph_stats") or {}

    if not any((seeds, chain, stats)):
        return markdown_text

    blocks: List[str] = ["", "## Appendix: Graph Evidence"]
    if stats:
        node_count = stats.get("nodes")
        edge_count = stats.get("edges")
        category_count = len(stats.get("categories") or []) if isinstance(stats.get("categories"), list) else None
        blocks.append(
            f"- Graph stats: nodes={node_count}, edges={edge_count}, categories={category_count}"
        )
    if seeds:
        blocks.append("")
        blocks.append("### Seed Entities")
        if DEEPSEARCH_TOP_SEED_ENTITIES is not None:
            seeds = seeds[: max(DEEPSEARCH_TOP_SEED_ENTITIES, 0)]
        blocks.extend(f"- {item}" for item in seeds)
    if chain:
        blocks.append("")
        blocks.append("### Graph Chain")
        if DEEPSEARCH_GRAPH_EDGE_LIMIT is not None:
            chain = chain[: max(DEEPSEARCH_GRAPH_EDGE_LIMIT, 0)]
        blocks.extend(f"{idx}. {edge}" for idx, edge in enumerate(chain, start=1))

    return (markdown_text or "").rstrip() + "\n" + "\n".join(blocks).rstrip() + "\n"


def _render_fallback_markdown(
    *,
    question: str,
    final_answer: str,
    highlights: List[Dict[str, Any]],
    evidences: List[Dict[str, Any]],
) -> str:
    blocks: List[str] = [f"# {question}"]
    answer = final_answer or ""
    if answer:
        blocks.extend(["", "## Answer", answer])
    if highlights:
        blocks.append("")
        blocks.append("## Highlights")
        for item in highlights:
            summary = str(item.get("summary") or "").strip()
            if summary:
                blocks.append(f"- {summary}")
    if evidences:
        blocks.append("")
        blocks.append("## Evidence Index")
        for entry in evidences:
            if not isinstance(entry, dict):
                continue
            chunk_id = str(entry.get("chunk_id") or "").strip()
            source = str(entry.get("source") or "").strip()
            content = str(entry.get("content") or "").strip()
            if chunk_id and content:
                source_tag = f" ({source})" if source else ""
                blocks.append(f"- [{chunk_id}]{source_tag}: {content}")
    return "\n".join(blocks).strip()
