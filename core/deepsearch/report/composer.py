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
        highlights = self._build_highlights(trace.get("reasoning_steps") or [])
        answer = self._render_answer(trace, highlights)
        cover_metrics = trace.get("gap_result") or trace.get("coverage_metrics") or {}

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

        if metadata["report_profile"]["include_graph_viz"]:
            metadata.setdefault("graph_visualization", trace.get("graph_traversals") or [])

        return {
            "question": question,
            "answer": answer,
            "evidences": evidences,
            "highlights": highlights,
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

    def _render_answer(self, trace: Dict[str, Any], highlights: List[Dict[str, Any]]) -> str:
        final_answer = (trace.get("final_answer") or "").strip()
        if final_answer:
            return final_answer

        header = self._template(self.ANSWER_TEMPLATE_KEY, "Key findings from graph reasoning:")
        if not highlights:
            return header + "\n- Evidence collected but no stable reasoning summary was produced."

        lines = [header]
        for idx, highlight in enumerate(highlights, start=1):
            summary = highlight.get("summary", "").strip()
            if not summary:
                continue
            lines.append(f"{idx}. {summary}")
        return "\n".join(lines)

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
