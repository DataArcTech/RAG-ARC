"""Report composer for DeepSearch.

This reporter converts DeepSearch execution traces into a readable, end-user report by
prompting an LLM with the collected evidence, highlights, and execution metadata.
"""
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_ids import hashed_chunk_id

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


def _resolve_citation_alias_flag(config_value: bool) -> bool:
    """Resolve citation alias flag with environment variable override.

    Environment variable DEEPSEARCH_CITATION_ALIASES takes precedence.
    """

    env_value = os.getenv("DEEPSEARCH_CITATION_ALIASES")
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
        self.citation_aliases = _resolve_citation_alias_flag(
            bool(self.config.get("citation_aliases", True))
        )

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
        llm_evidences, alias_bundle = self._alias_evidences_for_llm(evidences) if self.citation_aliases else (evidences, None)
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
        if alias_bundle:
            metadata["citation_aliases"] = {
                "enabled": True,
                "count": alias_bundle.get("count"),
                "sample": alias_bundle.get("sample"),
            }
        if request_context:
            metadata["request_context"] = request_context
        if metadata["report_profile"]["include_graph_viz"]:
            metadata["graph_visualization"] = trace.get("graph_traversals") or []

        context = self._build_llm_context(
            trace=trace,
            highlights=highlights,
            evidences=llm_evidences,
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
                    citation_aliases=(alias_bundle.get("alias_to_original") if alias_bundle else None),
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
        if alias_bundle:
            structured_llm, alias_diag = self._rewrite_citation_aliases(structured_llm, alias_bundle)
            if alias_diag:
                metadata["citation_alias_rewrite"] = alias_diag
        if self.enable_citation_agent:
            structured_llm, audit = CitationAgent().process(
                structured_report=structured_llm,
                evidences=evidences,
                graph_evidence=context.get("graph_evidence") or {},
                max_chunk_index_items=self.max_evidence_items,
                citation_aliases=(alias_bundle.get("alias_to_original") if alias_bundle else None),
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

    @staticmethod
    def _alias_evidences_for_llm(evidences: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
        """Replace long chunk IDs with stable short aliases for LLM prompting.

        The alias is treated as the chunk_id inside prompts, then rewritten back to the original
        chunk_id before running CitationAgent / ConsistencyChecker.
        """

        alias_to_original: Dict[str, str] = {}
        original_to_alias: Dict[str, str] = {}
        aliased: List[Dict[str, Any]] = []
        for idx, evidence in enumerate(evidences, start=1):
            if not isinstance(evidence, dict):
                continue
            original_id = str(evidence.get("chunk_id") or "").strip()
            if not original_id:
                continue
            alias = f"chunk_{idx:03d}"
            alias_to_original[alias] = original_id
            original_to_alias[original_id] = alias
            copied = dict(evidence)
            copied["chunk_id"] = alias
            provenance = copied.get("provenance") or {}
            if isinstance(provenance, dict):
                meta = provenance.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                meta.setdefault("original_chunk_id", original_id)
                provenance["metadata"] = meta
                copied["provenance"] = provenance
            aliased.append(copied)
        if not aliased:
            return evidences, None
        sample = [{"alias": a, "chunk_id": b} for a, b in list(alias_to_original.items())[:10]]
        return aliased, {
            "count": len(alias_to_original),
            "alias_to_original": alias_to_original,
            "original_to_alias": original_to_alias,
            "sample": sample,
        }

    _CITATION_RE = re.compile(r"(?:\[(?P<bracket>[^\[\]]{1,64})\]|【(?P<cjk>[^【】]{1,64})】)")

    def _rewrite_citation_aliases(
        self,
        structured: Dict[str, Any],
        alias_bundle: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        alias_to_original = alias_bundle.get("alias_to_original") or {}
        if not isinstance(alias_to_original, dict) or not alias_to_original:
            return structured, {}
        normalized_map = {str(k).strip().lower(): str(v) for k, v in alias_to_original.items()}

        replaced = 0
        unknown: List[str] = []

        def _normalize_token(token: str) -> str | None:
            raw = (token or "").strip()
            if not raw:
                return None
            low = raw.lower()

            def _lookup(candidate: str) -> str | None:
                key = (candidate or "").strip().lower()
                if not key:
                    return None
                if key in normalized_map:
                    return normalized_map[key]
                return None

            direct = _lookup(low)
            if direct:
                return direct

            cleaned = low.strip("`'\"(){}<>.,;: ")
            cleaned_lookup = _lookup(cleaned)
            if cleaned_lookup:
                return cleaned_lookup

            compact = re.sub(r"\s+", "", cleaned)
            compact_lookup = _lookup(compact)
            if compact_lookup:
                return compact_lookup

            if compact.isdigit():
                idx = int(compact)
                mapped = _lookup(f"chunk_{idx:03d}")
                if mapped:
                    return mapped

            if "chunk" not in compact:
                return None

            token_digits = re.sub(r"[^0-9]", "", compact)
            if token_digits.isdigit():
                idx = int(token_digits)
                mapped = _lookup(f"chunk_{idx:03d}")
                if mapped:
                    return mapped
            return None

        def _rewrite_text(text: str) -> str:
            nonlocal replaced

            def _sub(match: re.Match[str]) -> str:
                nonlocal replaced
                token = match.group("bracket") or match.group("cjk") or ""
                mapped = _normalize_token(token)
                if mapped:
                    replaced += 1
                    return f"[{mapped}]"

                low = token.strip().lower()
                if "chunk" in low and any(sep in low for sep in (",", ";", " ")):
                    parts = re.findall(r"chunk\s*[_-]?\s*\d{1,4}", token, flags=re.IGNORECASE)
                    mapped_parts: List[str] = []
                    unmapped_parts: List[str] = []
                    for part in parts:
                        mapped_part = _normalize_token(part)
                        if mapped_part:
                            mapped_parts.append(mapped_part)
                        else:
                            unmapped_parts.append(part)
                    if mapped_parts and not unmapped_parts:
                        replaced += len(mapped_parts)
                        return " ".join(f"[{item}]" for item in mapped_parts)
                    if unmapped_parts:
                        unknown.extend([part.strip() for part in unmapped_parts if str(part).strip()])

                if "chunk" in low:
                    unknown.append(token.strip())
                return match.group(0)

            return self._CITATION_RE.sub(_sub, text)

        rewritten = dict(structured)
        for key in ("summary", "title"):
            if isinstance(rewritten.get(key), str):
                rewritten[key] = _rewrite_text(rewritten[key])
        sections = rewritten.get("sections")
        if isinstance(sections, list):
            new_sections: List[Dict[str, Any]] = []
            for section in sections:
                if not isinstance(section, dict):
                    continue
                updated = dict(section)
                body = updated.get("body_markdown")
                if isinstance(body, str):
                    updated["body_markdown"] = _rewrite_text(body)
                new_sections.append(updated)
            rewritten["sections"] = new_sections
        for list_key in ("limitations", "next_steps"):
            items = rewritten.get(list_key)
            if isinstance(items, list):
                rewritten[list_key] = [_rewrite_text(str(item)) if isinstance(item, str) else item for item in items]
        citations = rewritten.get("citations")
        if isinstance(citations, list):
            new_citations: List[Dict[str, Any]] = []
            for entry in citations:
                if not isinstance(entry, dict):
                    continue
                updated = dict(entry)
                ev_id = updated.get("evidence_id")
                if isinstance(ev_id, str):
                    mapped = _normalize_token(ev_id)
                    if mapped:
                        updated["evidence_id"] = mapped
                        replaced += 1
                new_citations.append(updated)
            rewritten["citations"] = new_citations

        diag = {
            "replaced": replaced,
            "unknown_aliases": sorted({token for token in unknown})[:20],
        }
        return rewritten, diag

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

        graph_evidence_full = self._build_graph_evidence(trace, evidences)
        graph_evidence_llm = self._slim_graph_evidence_for_llm(graph_evidence_full)

        return {
            "question": trace.get("question") or "",
            "final_answer": trace.get("final_answer") or "",
            "highlights": highlights,
            "evidences": evidences,
            "graph_chain": (trace.get("graph_chain") or [])[: self.report_max_graph_chain_items],
            "graph_evidence": graph_evidence_llm,
            "methodology": methodology,
            "coverage": coverage_bundle,
            "request_context": request_context,
        }

    def _merge_evidences(
        self,
        internal: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
        external: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
    ) -> List[Dict[str, Any]]:
        internal_items = [item for item in (self._normalize_evidence(payload) for payload in (internal or [])) if item]
        external_items = [item for item in (self._normalize_evidence(payload) for payload in (external or [])) if item]

        max_items = self.max_evidence_items if self.max_evidence_items is not None else None
        if max_items is not None and max_items <= 0:
            return []

        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _add(items: List[Dict[str, Any]], limit: int | None) -> None:
            nonlocal merged
            for chunk in items:
                source = str(chunk.get("source") or "").strip()
                chunk_id = chunk.get("chunk_id") or self._hash_content(chunk)
                key = f"{source}::{chunk_id}"
                if key in seen:
                    continue
                seen.add(key)
                chunk.setdefault("chunk_id", chunk_id)
                merged.append(chunk)
                if limit is not None and len(merged) >= limit:
                    return

        if external_items and max_items is not None:
            external_budget = min(len(external_items), max(2, max_items // 3))
            internal_budget = max(0, max_items - external_budget)
            _add(internal_items, internal_budget)
            _add(external_items, max_items)
            if len(merged) < max_items:
                _add(internal_items, max_items)
                _add(external_items, max_items)
            return merged[:max_items]

        _add(internal_items, max_items)
        if max_items is not None and len(merged) >= max_items:
            return merged[:max_items]
        _add(external_items, max_items)
        return merged[:max_items] if max_items is not None else merged

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
        return hashed_chunk_id(source=str(chunk.get("source") or ""), content=str(chunk.get("content") or ""))

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

    @staticmethod
    def _slim_graph_evidence_for_llm(graph_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce graph evidence payload for LLM prompts to avoid context blow-ups.

        The full public payload can still include detailed chunk/graph snapshots, but the report-writing
        prompt should only receive compact graph signals (seeds + chain + stats) because the authoritative
        evidence snippets are provided separately.
        """

        if not isinstance(graph_evidence, dict):
            return {}
        seed_entities = graph_evidence.get("seed_entities") if isinstance(graph_evidence.get("seed_entities"), list) else []
        graph_chain = graph_evidence.get("graph_chain") if isinstance(graph_evidence.get("graph_chain"), list) else []
        graph_stats = graph_evidence.get("graph_stats") if isinstance(graph_evidence.get("graph_stats"), dict) else {}
        return {
            "seed_entities": seed_entities[:12],
            "graph_chain": graph_chain[:40],
            "graph_stats": graph_stats,
        }


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
