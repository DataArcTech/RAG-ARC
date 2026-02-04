from typing import Any, Dict, Iterable, List, Tuple

from core.utils.json_safe import json_safe

from .artifact_views_v2 import build_ref


def _coerce_evidence_id(evidence: Dict[str, Any]) -> str | None:
    if not isinstance(evidence, dict):
        return None
    for key in ("chunk_id", "evidence_id"):
        raw = evidence.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _extract_report_evidence_ids(report: Dict[str, Any]) -> List[str]:
    """Prefer citeable evidence ids from the structured report over raw report evidences.

    Rationale:
    - report["evidences"] may include navigation snippets (bm25/faiss/etc) used during reasoning.
    - structured_report.source_key_map / citations represent what the report can actually cite.
    """

    if not isinstance(report, dict):
        return []
    structured = report.get("structured_report")
    if isinstance(structured, dict):
        source_key_map = structured.get("source_key_map")
        if isinstance(source_key_map, dict) and source_key_map:
            pairs: List[tuple[int, str]] = []
            for key, ev_id in source_key_map.items():
                try:
                    key_num = int(str(key).strip())
                except Exception:  # noqa: BLE001
                    continue
                token = str(ev_id or "").strip()
                if token:
                    pairs.append((key_num, token))
            pairs.sort(key=lambda item: item[0])
            ordered = [ev_id for _k, ev_id in pairs]
            # Dedup while preserving order.
            out: List[str] = []
            seen: set[str] = set()
            for ev_id in ordered:
                if ev_id in seen:
                    continue
                seen.add(ev_id)
                out.append(ev_id)
            return out

        citations = structured.get("citations")
        if isinstance(citations, list) and citations:
            out: List[str] = []
            seen: set[str] = set()
            for entry in citations:
                if not isinstance(entry, dict):
                    continue
                token = str(entry.get("evidence_id") or entry.get("chunk_id") or "").strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                out.append(token)
            if out:
                return out

    return []


def build_evidence_pool_v2(
    *,
    reasoning: Dict[str, Any],
    report: Dict[str, Any],
    artifact_version: int = 2,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Build a single evidence pool and return (pool, reasoning_evidence_ids, report_evidence_ids)."""

    evidences_by_id: Dict[str, Any] = {}
    order: List[str] = []

    def _add_evidence(evidence: Dict[str, Any]) -> str | None:
        ev_id = _coerce_evidence_id(evidence)
        if not ev_id:
            return None
        if ev_id not in evidences_by_id:
            order.append(ev_id)
        evidences_by_id[ev_id] = json_safe(dict(evidence))
        return ev_id

    reasoning_evidence_ids: List[str] = []
    raw_reasoning_evidences = reasoning.get("evidences") if isinstance(reasoning, dict) else None
    if isinstance(raw_reasoning_evidences, list):
        for item in raw_reasoning_evidences:
            if isinstance(item, dict):
                ev_id = _add_evidence(item)
                if ev_id:
                    reasoning_evidence_ids.append(ev_id)

    raw_tool_results = reasoning.get("tool_results") if isinstance(reasoning, dict) else None
    if isinstance(raw_tool_results, list):
        for entry in raw_tool_results:
            if not isinstance(entry, dict):
                continue
            result = entry.get("result")
            if not isinstance(result, dict):
                continue
            tool_evidences = result.get("evidences")
            if not isinstance(tool_evidences, list):
                continue
            for item in tool_evidences:
                if isinstance(item, dict):
                    _add_evidence(item)

    report_evidence_ids: List[str] = []
    raw_report_evidences = report.get("evidences") if isinstance(report, dict) else None
    # Prefer citeable report evidence ids from the structured report.
    cited_ids = _extract_report_evidence_ids(report if isinstance(report, dict) else {})
    if cited_ids:
        for ev_id in cited_ids:
            token = str(ev_id or "").strip()
            if not token:
                continue
            # Ensure the evidence object exists in the pool (it usually comes from tool_results already).
            if token not in evidences_by_id and isinstance(raw_report_evidences, list):
                for item in raw_report_evidences:
                    if not isinstance(item, dict):
                        continue
                    if _coerce_evidence_id(item) == token:
                        _add_evidence(item)
                        break
            if token in evidences_by_id:
                report_evidence_ids.append(token)
    elif isinstance(raw_report_evidences, list):
        # Backward-compatible behavior: include all report evidences when we have no structured citation info.
        for item in raw_report_evidences:
            if isinstance(item, dict):
                ev_id = _add_evidence(item)
                if ev_id:
                    report_evidence_ids.append(ev_id)

    pool = {
        "artifact_version": int(artifact_version),
        "kind": "evidence_pool",
        "order": list(order),
        "evidences_by_id": evidences_by_id,
    }
    return pool, reasoning_evidence_ids, report_evidence_ids


def _ref(*, file: str, json_pointer: str | None = None, enabled: bool) -> Dict[str, Any]:
    return build_ref(file=file, json_pointer=json_pointer, enabled=enabled)


def dedupe_reasoning_v2(
    *,
    reasoning: Dict[str, Any],
    refs_enabled: bool,
    evidence_pool_filename: str,
    plan_filename: str,
    evidence_ids: List[str],
) -> Dict[str, Any]:
    payload = dict(reasoning or {})

    payload.pop("evidences", None)
    payload["evidence_pool_ref"] = _ref(file=evidence_pool_filename, enabled=refs_enabled)
    payload["evidence_ids"] = list(evidence_ids or [])

    # Avoid repeating plan steps across plan_result.json / state snapshot.
    if "plan_steps" in payload:
        payload.pop("plan_steps", None)
    payload["plan_ref"] = _ref(file=plan_filename, json_pointer="/plan", enabled=refs_enabled)

    # Remove duplicate evidences embedded in tool_results.
    tool_results = payload.get("tool_results")
    if isinstance(tool_results, list):
        updated: List[Dict[str, Any]] = []
        for entry in tool_results:
            if not isinstance(entry, dict):
                continue
            entry_copy = dict(entry)
            result = entry_copy.get("result")
            if isinstance(result, dict) and "evidences" in result:
                res_copy = dict(result)
                raw_evs = res_copy.pop("evidences", None)
                ids: List[str] = []
                if isinstance(raw_evs, list):
                    for item in raw_evs:
                        if isinstance(item, dict):
                            ev_id = _coerce_evidence_id(item)
                            if ev_id:
                                ids.append(ev_id)
                if ids:
                    res_copy["evidence_ids"] = ids
                entry_copy["result"] = res_copy
            updated.append(entry_copy)
        payload["tool_results"] = updated

    return json_safe(payload)


def dedupe_report_v2(
    *,
    report: Dict[str, Any],
    refs_enabled: bool,
    report_markdown_filename: str,
    evidence_pool_filename: str,
    reasoning_filename: str,
    plan_filename: str,
    evidence_ids: List[str],
) -> Dict[str, Any]:
    payload = dict(report or {})

    # Keep the markdown in report.md only.
    payload.pop("answer", None)
    payload["answer_ref"] = _ref(file=report_markdown_filename, enabled=refs_enabled)

    payload.pop("evidences", None)
    payload["evidence_pool_ref"] = _ref(file=evidence_pool_filename, enabled=refs_enabled)
    payload["evidence_ids"] = list(evidence_ids or [])

    structured = payload.get("structured_report")
    if isinstance(structured, dict):
        structured_copy = dict(structured)
        if "text" in structured_copy:
            structured_copy.pop("text", None)
            structured_copy["text_ref"] = _ref(file=report_markdown_filename, enabled=refs_enabled)
        payload["structured_report"] = structured_copy

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        meta_copy = dict(metadata)
        # Remove duplicated blocks; replace with refs (keeps payload small + debuggable).
        if "tool_results" in meta_copy:
            meta_copy.pop("tool_results", None)
            meta_copy["tool_results_ref"] = _ref(file=reasoning_filename, json_pointer="/tool_results", enabled=refs_enabled)
        if "reasoning_steps" in meta_copy:
            meta_copy.pop("reasoning_steps", None)
            meta_copy["reasoning_steps_ref"] = _ref(file=reasoning_filename, json_pointer="/reasoning_steps", enabled=refs_enabled)
        if "think_notes" in meta_copy:
            meta_copy.pop("think_notes", None)
            meta_copy["think_notes_ref"] = _ref(file=reasoning_filename, json_pointer="/think_notes", enabled=refs_enabled)
        if "coverage_metrics" in meta_copy:
            meta_copy.pop("coverage_metrics", None)
            meta_copy["coverage_metrics_ref"] = _ref(file=reasoning_filename, json_pointer="/coverage_metrics", enabled=refs_enabled)
        if "graph_visualization" in meta_copy:
            meta_copy.pop("graph_visualization", None)
            meta_copy["graph_visualization_ref"] = _ref(file=reasoning_filename, json_pointer="/graph_traversals", enabled=refs_enabled)
        if "plan" in meta_copy:
            meta_copy.pop("plan", None)
            meta_copy["plan_ref"] = _ref(file=plan_filename, json_pointer="/plan", enabled=refs_enabled)
        if "structured_report" in meta_copy:
            meta_copy.pop("structured_report", None)
            meta_copy["structured_report_ref"] = _ref(file="report.json", json_pointer="/structured_report", enabled=refs_enabled)
        meta_copy["reasoning_ref"] = _ref(file=reasoning_filename, enabled=refs_enabled)
        payload["metadata"] = meta_copy

    return json_safe(payload)
