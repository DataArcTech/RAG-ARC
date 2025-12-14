"""Reporter that fuses graph reasoning traces and evidence into API-facing payloads."""
import hashlib
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk


class DeepSearchReporter:
    """Aggregate reasoning traces, evidences, and metrics into a structured report."""

    ANSWER_TEMPLATE_KEY = "deepsearch.report.answer_header"

    def __init__(self, template_store, config):
        self.template_store = template_store
        self.config = config or {}
        self.max_highlights = int(self.config.get("max_highlights", 4))
        self.max_evidence_items = int(self.config.get("max_evidence_items", 24))

    def compose(
        self,
        reasoning_trace: Dict[str, Any],
        external_evidence: Optional[Iterable[Dict[str, Any] | EvidenceChunk]] = None,
    ) -> Dict[str, Any]:
        """Return DeepSearch report dict combining graph reasoning with external context."""

        trace = reasoning_trace or {}
        question = trace.get("question", "").strip()
        adapter_metadata = trace.get("adapter_metadata") or {}

        evidences = self._merge_evidences(trace.get("evidences"), external_evidence)
        reasoning_steps = trace.get("reasoning_steps") or []
        highlights = self._build_highlights(reasoning_steps)
        coverage_metrics = trace.get("gap_result") or trace.get("coverage_metrics") or {}
        request_context = self._extract_request_context(trace)
        answer = self._render_answer(trace, highlights, request_context, coverage_metrics)
        cover_metrics = coverage_metrics

        metadata = {
            "question": question,
            "adapter_metadata": adapter_metadata,
            "graph_summary": self._graph_summary(trace.get("graph_traversals") or []),
            "plan": {
                "steps": trace.get("plan_steps") or [],
                "completed": sum(1 for step in (trace.get("reasoning_steps") or []) if step.get("status") == "done"),
            },
            "reasoning_steps": trace.get("reasoning_steps") or [],
            "think_notes": trace.get("think_notes") or [],
            "tool_results": trace.get("tool_results") or [],
            "coverage_metrics": cover_metrics,
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
            metadata.setdefault("graph_visualization", trace.get("graph_traversals") or [])

        structured = self._build_structured_report(
            question=question,
            summary_text=answer,
            highlights=highlights,
            evidences=evidences,
            plan_steps=trace.get("plan_steps") or [],
            reasoning_steps=reasoning_steps,
            coverage=cover_metrics,
            request_context=request_context,
        )
        structured_meta = {key: structured[key] for key in structured if key != "text"}
        metadata["structured_report"] = structured_meta

        return {
            "question": question,
            "answer": answer,
            "evidences": evidences,
            "highlights": highlights,
            "structured_report": structured_meta,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
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
            if self.max_evidence_items > 0 and len(merged) >= self.max_evidence_items:
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

    @staticmethod
    def _normalize_evidence(payload: Dict[str, Any] | EvidenceChunk | None) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if isinstance(payload, EvidenceChunk):
            return payload.model_dump()
        if isinstance(payload, dict):
            content = payload.get("content")
            if content:
                return dict(payload)
        return None

    @staticmethod
    def _hash_content(chunk: Dict[str, Any]) -> str:
        digest = hashlib.sha256((chunk.get("source", "") + "::" + chunk.get("content", "")).encode("utf-8"))
        return f"anon-{digest.hexdigest()[:12]}"

    def _format_request_context_lines(self, request_context: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for key, value in request_context.items():
            if value is None:
                continue
            lines.append(f"- {key}: {value}")
        return lines

    def _format_request_context_section(self, request_context: Dict[str, Any]) -> str:
        lines = self._format_request_context_lines(request_context)
        if not lines:
            return ""
        return "## Request Context\n" + "\n".join(lines)

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

    def _render_answer(
        self,
        trace: Dict[str, Any],
        highlights: List[Dict[str, Any]],
        request_context: Dict[str, Any],
        coverage: Dict[str, Any],
    ) -> str:
        final_answer = (trace.get("final_answer") or "").strip()
        blocks: List[str] = []
        if final_answer:
            blocks.append(f"## Final Answer\n{final_answer}")
        else:
            header = self._template(self.ANSWER_TEMPLATE_KEY, "Key findings from graph reasoning:")
            if not highlights:
                blocks.append(header + "\n- Evidence collected but no stable reasoning summary was produced.")
            else:
                lines = [header]
                for idx, highlight in enumerate(highlights, start=1):
                    summary = (highlight.get("summary") or "").strip()
                    if summary:
                        lines.append(f"{idx}. {summary}")
                blocks.append("\n".join(lines))

        context_block = self._format_request_context_section(request_context)
        if context_block:
            blocks.append(context_block)

        key_lines = []
        for idx, highlight in enumerate(highlights, start=1):
            summary = (highlight.get("summary") or "").strip()
            if summary:
                key_lines.append(f"{idx}. {summary}")
        if key_lines:
            blocks.append("## Key Findings\n" + "\n".join(key_lines))

        confidence_block = self._confidence_narrative(coverage)
        if confidence_block:
            blocks.append("## Confidence & Next Steps\n" + confidence_block)

        return "\n\n".join(block for block in blocks if block).strip()

    def _graph_summary(self, traversals: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    def _template(self, key: str, default: str) -> str:
        if not self.template_store:
            return default
        getter = getattr(self.template_store, "get", None)
        if callable(getter):
            value = getter(key)
            if value:
                return str(value)
        if isinstance(self.template_store, dict):
            value = self.template_store.get(key)
            if value:
                return str(value)
        return default

    # ------------------------------------------------------------------
    def _build_structured_report(
        self,
        *,
        question: str,
        summary_text: str,
        highlights: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
        plan_steps: List[Dict[str, Any]],
        reasoning_steps: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        request_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble a Markdown report with sections, evidence, and reasoning context."""

        title = question or "DeepSearch Analysis"
        sanitized_summary = (summary_text or "").strip() or "No definitive conclusion was produced; review evidence and reasoning."
        evidence_map = {item.get("chunk_id"): item for item in evidences if item.get("chunk_id")}

        key_findings, highlight_ids = self._format_highlights(highlights, evidence_map)
        evidence_lines, citations = self._format_evidence_section(evidences)
        methodology_lines = self._format_methodology_section(plan_steps, reasoning_steps)
        coverage_lines = self._format_coverage_section(coverage)
        context_lines = self._format_request_context_lines(request_context)
        confidence_blurb = self._confidence_narrative(coverage)

        sections = [
            {"title": "Summary", "body": sanitized_summary},
        {"title": "Key Findings", "body": "\n".join(key_findings) if key_findings else "No prioritized findings were recorded."},
            {
                "title": "Supporting Evidence",
                "body": "\n".join(evidence_lines) if evidence_lines else "No evidence snippets were captured.",
            },
            {
                "title": "Methodology",
                "body": "\n".join(methodology_lines) if methodology_lines else "Plan execution logs were unavailable.",
            },
            {
                "title": "Coverage & Confidence",
                "body": (confidence_blurb + ("\n" if confidence_blurb and coverage_lines else "") + "\n".join(coverage_lines)).strip()
                if (confidence_blurb or coverage_lines)
                else "No coverage or confidence metrics reported.",
            },
        ]
        if context_lines:
            sections.insert(1, {"title": "Research Context", "body": "\n".join(context_lines)})

        text_blocks: List[str] = [f"# {title}"]
        for section in sections:
            text_blocks.extend(["", f"## {section['title']}", section["body"]])
        full_text = "\n".join(text_blocks).strip()

        return {
            "title": title,
            "summary": sanitized_summary,
            "sections": sections,
            "citations": citations,
            "highlight_evidence_ids": highlight_ids,
            "text": full_text,
            "context": request_context or {},
        }

    def _format_highlights(
        self,
        highlights: List[Dict[str, Any]],
        evidence_map: Dict[str, Dict[str, Any]],
    ) -> tuple[List[str], List[str]]:
        """Return ordered highlight strings and the evidence IDs they reference."""

        lines: List[str] = []
        referenced_ids: List[str] = []
        for idx, block in enumerate(highlights, start=1):
            summary = (block.get("summary") or "").strip()
            if not summary:
                continue
            evidence_ids = [eid for eid in (block.get("evidence_ids") or []) if eid]
            referenced_ids.extend(evidence_ids)
            reference = self._format_evidence_refs(evidence_ids, evidence_map)
            lines.append(f"{idx}. {summary}{reference}")
        return lines, referenced_ids

    def _format_evidence_refs(
        self,
        evidence_ids: List[str],
        evidence_map: Dict[str, Dict[str, Any]],
    ) -> str:
        """Format citation labels referencing the evidence map."""

        labels: List[str] = []
        for eid in evidence_ids:
            chunk = evidence_map.get(eid)
            if not chunk:
                continue
            source = chunk.get("source") or "graph"
            labels.append(f"{eid}@{source}")
        if not labels:
            return ""
        return f" (Evidence: {', '.join(labels)})"

    def _format_evidence_section(
        self,
        evidences: List[Dict[str, Any]],
        *,
        preview_chars: int = 220,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Summarize collected evidences for the report body and citation payload."""

        lines: List[str] = []
        citations: List[Dict[str, Any]] = []
        for chunk in evidences:
            chunk_id = chunk.get("chunk_id") or self._hash_content(chunk)
            content = self._truncate_text(chunk.get("content") or "", preview_chars)
            source = chunk.get("source") or "graph"
            score = chunk.get("score")
            lines.append(f"- [{chunk_id}] {source}: {content}")
            citations.append(
                {
                    "evidence_id": chunk_id,
                    "source": source,
                    "excerpt": content,
                    "score": score,
                }
            )
        return lines, citations

    def _format_methodology_section(
        self,
        plan_steps: List[Dict[str, Any]],
        reasoning_steps: List[Dict[str, Any]],
    ) -> List[str]:
        """Link plan steps with executed reasoning summaries."""

        lines: List[str] = []
        reasoning_lookup = {step.get("step_id"): step for step in reasoning_steps if step.get("step_id")}
        for step in plan_steps:
            step_id = step.get("step_id")
            description = step.get("description") or ""
            channel = step.get("channel") or "graph"
            execution = reasoning_lookup.get(step_id, {})
            status = execution.get("status") or "pending"
            output = (execution.get("output_summary") or "").strip() or "No summary captured."
            tool = execution.get("tool") or (execution.get("metadata") or {}).get("tool")
            if not tool:
                diagnostics = execution.get("diagnostics") or {}
                tool = diagnostics.get("tool")
            tool_label = f"{tool}" if tool else channel
            lines.append(f"- {step_id or '?'} | {description} ({tool_label}) -> {status}: {output}")
        return lines

    @staticmethod
    def _format_coverage_section(coverage: Dict[str, Any]) -> List[str]:
        """Format coverage/confidence metrics."""

        lines: List[str] = []
        for key, value in coverage.items():
            if value is None:
                continue
            lines.append(f"- {key}: {value}")
        return lines

    @staticmethod
    def _confidence_narrative(coverage: Dict[str, Any]) -> str:
        if not coverage:
            return ""
        ratio = coverage.get("coverage_ratio") or coverage.get("coverage_score")
        confidence = coverage.get("confidence") or coverage.get("confidence_score")
        missing = coverage.get("missing_topics") or []
        parts: List[str] = []
        if confidence is not None:
            parts.append(f"Self-reported confidence: {confidence}")
        if ratio is not None:
            parts.append(f"Estimated coverage ratio: {ratio}")
        if missing:
            parts.append("Outstanding topics: " + ", ".join(str(topic) for topic in missing))
        if not parts:
            return ""
        return "\n".join(f"- {line}" for line in parts)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Collapse whitespace and truncate evidence snippets for readability."""

        collapsed = " ".join((text or "").split())
        if not collapsed:
            return ""
        if len(collapsed) <= max_chars:
            return collapsed
        return f"{collapsed[:max_chars].rstrip()}..."
