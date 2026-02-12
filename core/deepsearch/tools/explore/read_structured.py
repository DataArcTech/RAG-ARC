"""Structured reading tools for long documents.

DeepSearch contract:
- `search.*` outputs are navigation-only and MUST NOT be cited.
- `read.pages` is the primary evidence path and MUST return *full-page* content.

Implementation note:
- We read from graph-stored `Chunk` nodes and rely on PageIndex page metadata
  (`page_start/page_end`, `chunk_index`) to assemble page-level evidence.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import pageindex as pageindex_cfg
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from config.core.deepsearch import tool_defaults
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.deepsearch.utils.ids import normalize_uuid
from core.deepsearch.utils.node_types import normalize_node_type
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher
from core.utils.json_extract import safe_json_loads

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        parsed = safe_json_loads(raw, expected="dict")
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_filename(rows: Sequence["_ChunkRow"]) -> Optional[str]:
    for row in rows:
        meta = row.metadata or {}
        for key in ("filename", "source_file_name", "file_name", "source_path", "path"):
            token = str(meta.get(key) or "").strip()
            if token:
                return token
    return None


@dataclass(frozen=True)
class _ChunkRow:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]

    @property
    def chunk_index(self) -> int:
        token = _coerce_int(self.metadata.get("chunk_index"))
        return token if token is not None else 1_000_000

    @property
    def page_start(self) -> Optional[int]:
        return _coerce_int(self.metadata.get("page_start"))

    @property
    def page_end(self) -> Optional[int]:
        return _coerce_int(self.metadata.get("page_end"))


async def _fetch_file_chunks(
    *,
    request: ToolRunRequest,
    file_id: str,
    page_start: int,
    page_end: int,
) -> Tuple[List[_ChunkRow], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"file_id": file_id, "page_start": page_start, "page_end": page_end}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return [], diagnostics
    if request.adapter is None or not adapter_supports_cypher(request.adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return [], diagnostics
    if request.access_scope is None:
        diagnostics["reason"] = "owner_scope_missing"
        return [], diagnostics

    cypher = (
        "MATCH (c:Chunk)\n"
        "WHERE c.source_file_id = $file_id AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
        "  AND c.page_start IS NOT NULL AND c.page_end IS NOT NULL\n"
        "  AND c.page_start <= $page_end AND c.page_end >= $page_start\n"
        "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata, "
        "c.page_start AS page_start, c.page_end AS page_end\n"
    )
    # NOTE: Do not apply a LIMIT here. `read.pages` must be able to return a full page even when
    # the file contains many chunks. Selection should be controlled by the requested page range.
    params = {"file_id": file_id, "page_start": page_start, "page_end": page_end}
    async with adapter_locked(request.adapter):
        rows = await request.adapter.acypher(cypher, params, access_scope=request.access_scope)

    diagnostics["query_mode"] = "indexed_page_overlap"
    diagnostics["rows_scanned"] = len(rows or [])

    # Backward compatibility for legacy chunks that only keep page info in metadata JSON.
    if not rows:
        legacy_cypher = (
            "MATCH (c:Chunk)\n"
            "WHERE c.source_file_id = $file_id AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata\n"
        )
        async with adapter_locked(request.adapter):
            rows = await request.adapter.acypher(legacy_cypher, {"file_id": file_id}, access_scope=request.access_scope)
        diagnostics["query_mode"] = "legacy_full_scan_fallback"
        diagnostics["rows_scanned"] = len(rows or [])

    parsed: List[_ChunkRow] = []
    missing = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        meta = _parse_metadata(row.get("metadata"))
        row_page_start = row.get("page_start")
        row_page_end = row.get("page_end")
        if row_page_start is not None:
            meta["page_start"] = row_page_start
        if row_page_end is not None:
            meta["page_end"] = row_page_end
        if not meta:
            missing += 1
        parsed.append(_ChunkRow(chunk_id=chunk_id, content=str(row.get("content") or ""), metadata=meta))
    diagnostics["missing_metadata_rows"] = missing
    return parsed, diagnostics


def _join_full_page(chunks: Sequence[_ChunkRow]) -> tuple[str, list[str]]:
    """Join all chunk blocks on a page without truncation."""

    used_ids: list[str] = []
    parts: list[str] = []
    for row in chunks:
        block = (row.content or "").strip()
        if not block:
            continue
        if parts:
            parts.append("")
        parts.append(block)
        used_ids.append(row.chunk_id)
    return "\n".join(parts).strip(), used_ids


def _median(values: Sequence[int]) -> int:
    nums = [int(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return 0
    nums.sort()
    mid = len(nums) // 2
    return int(nums[mid]) if len(nums) % 2 == 1 else int((nums[mid - 1] + nums[mid]) / 2)


class ReadPagesTool(GraphTool):
    descriptor = ToolDescriptor(
        name="read.pages",
        channel="graph",
        description="Read all chunks overlapping a page range (requires PageIndex page metadata).",
        speed="fast",
        cost="low",
        strategy_tags=("read", "pages", "pageindex", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.read.pages",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "Target file_id (required)."},
                "page_start": {"type": "integer", "minimum": 0, "description": "Start page (inclusive)."},
                "page_end": {"type": "integer", "minimum": 0, "description": "End page (inclusive)."},
                "goal": {"type": "string", "description": "Why you are reading these pages (for provenance)."},
            },
            required_extra_fields=("file_id", "page_start", "page_end"),
        ),
        example_args={
            "question": "Read the warning page",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "page_start": 6, "page_end": 6, "goal": "capture warning text"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        file_id_raw = str(extra.get("file_id") or "").strip()
        file_id = normalize_uuid(file_id_raw)
        page_start = _coerce_int(extra.get("page_start"))
        page_end = _coerce_int(extra.get("page_end"))
        goal = str(extra.get("goal") or "").strip() or None
        if not file_id or page_start is None or page_end is None:
            reason = "missing_args"
            if file_id_raw and not file_id:
                reason = "invalid_file_id_format"
            return ToolResult(
                summary="read.pages skipped: missing/invalid file_id or missing page range (use search.file/section.select).",
                diagnostics={"reason": reason, "file_id_raw": file_id_raw or None, "page_start": page_start, "page_end": page_end},
            )

        if page_end < page_start:
            page_start, page_end = page_end, page_start

        all_rows, fetch_diag = await _fetch_file_chunks(
            request=request,
            file_id=file_id,
            page_start=page_start,
            page_end=page_end,
        )
        if not all_rows:
            return ToolResult(summary="read.pages returned no chunks.", diagnostics={**fetch_diag, "reason": fetch_diag.get("reason") or "empty_file"})

        def _row_overlaps_page(row: _ChunkRow, page: int) -> bool:
            ps = row.page_start
            pe = row.page_end
            if ps is None and pe is None:
                return False
            if ps is None:
                ps = pe
            if pe is None:
                pe = ps
            if ps is None or pe is None:
                return False
            return ps <= page <= pe

        evidences: list[EvidenceChunk] = []
        page_summaries: list[str] = []
        filename = _extract_filename(all_rows)
        page_diag: dict[int, dict[str, Any]] = {}
        page_char_counts: dict[int, int] = {}
        for page in range(page_start, page_end + 1):
            page_rows = [row for row in all_rows if _row_overlaps_page(row, page)]
            if not page_rows:
                continue
            page_rows.sort(key=lambda r: (r.chunk_index, r.chunk_id))
            content, used_ids = _join_full_page(page_rows)
            used_id_set = set(used_ids)

            node_type_counts: Dict[str, int] = {}
            image_urls: List[str] = []
            for row in page_rows:
                if row.chunk_id not in used_id_set:
                    continue
                meta = row.metadata or {}
                token = normalize_node_type(meta.get("semantic_unit_type"))
                node_type_counts[token] = node_type_counts.get(token, 0) + 1
                urls = meta.get("image_urls")
                if isinstance(urls, list):
                    image_urls.extend([str(u).strip() for u in urls if str(u or "").strip()])

            metadata_payload: Dict[str, Any] = {"source_file_id": file_id, "page_start": page, "page_end": page}
            if filename:
                metadata_payload["filename"] = filename
            if image_urls:
                metadata_payload["image_urls"] = list(dict.fromkeys(image_urls))

            evidences.append(
                EvidenceChunk(
                    chunk_id=hashed_chunk_id(source="read.pages", content=f"{file_id}:{page}-{page}:{len(used_ids)}"),
                    source="read.pages",
                    content=content,
                    kind=EVIDENCE_KIND_PRIMARY,
                    score=None,
                    provenance={
                        "goal": goal,
                        "source_file_id": file_id,
                        "page_start": page,
                        "page_end": page,
                        "chunk_ids": used_ids,
                        "truncated": False,
                        "node_type": "page",
                        "node_types": node_type_counts,
                        "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
                        "metadata": metadata_payload,
                    },
                )
            )
            page_summaries.append(f"p{page}: {len(used_ids)} chunks, {len(content)} chars")
            char_count = int(len(content))
            page_char_counts[page] = char_count
            page_diag[page] = {
                "used_chunk_count": len(used_ids),
                "char_count": char_count,
                "node_types": node_type_counts,
                "image_url_count": len(image_urls),
            }

        if not evidences:
            return ToolResult(
                summary="read.pages returned no chunks (page metadata missing or range unmatched).",
                diagnostics={**fetch_diag, "reason": "pages_not_found", "page_start": page_start, "page_end": page_end},
            )

        summary = "read.pages returned full page evidence: " + "; ".join(page_summaries)
        # Navigation-only continuity hints:
        # - Tables/equations often span adjacent pages.
        # - List-heavy or unusually dense pages may continue to p±1.
        # This tool MUST NOT auto-fetch; it only suggests the next read.pages call.
        suggested_expansions: list[dict[str, Any]] = []
        if bool(getattr(tool_defaults, "READ_PAGES_SIGNALS_ENABLED", True)):
            delta = int(getattr(tool_defaults, "READ_PAGES_SIGNALS_EXPAND_DELTA_PAGES", 1) or 1)
            delta = max(1, delta)
            abs_min_chars = int(getattr(tool_defaults, "READ_PAGES_SIGNALS_LONG_PAGE_MIN_CHARS", 0) or 0)
            abs_min_chunks = int(getattr(tool_defaults, "READ_PAGES_SIGNALS_DENSE_PAGE_MIN_CHUNKS", 0) or 0)
            mult = float(getattr(tool_defaults, "READ_PAGES_SIGNALS_MEDIAN_MULTIPLIER", 1.0) or 1.0)
            list_min_chunks = int(getattr(tool_defaults, "READ_PAGES_SIGNALS_LIST_MIN_CHUNKS", 0) or 0)

            median_chars = _median(list(page_char_counts.values()))
            median_chunks = _median([int(info.get("used_chunk_count") or 0) for info in page_diag.values() if isinstance(info, dict)])

            reasons: set[str] = set()
            for _p, info in page_diag.items():
                if not isinstance(info, dict):
                    continue
                node_types = info.get("node_types") or {}
                char_count = int(info.get("char_count") or 0)
                used_chunk_count = int(info.get("used_chunk_count") or 0)

                if int(node_types.get("table") or 0) > 0 or int(node_types.get("equation") or 0) > 0:
                    reasons.add("table_or_equation_detected")
                if list_min_chunks > 0 and int(node_types.get("list") or 0) >= list_min_chunks:
                    reasons.add("list_detected")

                # Dense/long-page heuristic: absolute or unusually large vs median within this call.
                if abs_min_chars > 0 and char_count >= abs_min_chars:
                    reasons.add("long_page_detected")
                elif median_chars > 0 and mult > 1.0 and char_count >= int(median_chars * mult):
                    reasons.add("long_page_detected")

                if abs_min_chunks > 0 and used_chunk_count >= abs_min_chunks:
                    reasons.add("dense_page_detected")
                elif median_chunks > 0 and mult > 1.0 and used_chunk_count >= int(median_chunks * mult):
                    reasons.add("dense_page_detected")

            if reasons:
                lo = max(0, int(page_start) - delta)
                hi = int(page_end) + delta
                if lo != page_start or hi != page_end:
                    suggested_expansions.append(
                        {
                            "tool": "read.pages",
                            "args": {"file_id": file_id, "page_start": lo, "page_end": hi, "goal": "expand contiguous pages for continuity (navigation hint)"},
                            "reason": ",".join(sorted(reasons)),
                        }
                    )
                    summary = summary.rstrip() + f" TIP: continuity hint ({','.join(sorted(reasons))}); consider expanding to p{lo}-p{hi}."
        diagnostics = {
            **fetch_diag,
            "page_start": page_start,
            "page_end": page_end,
            "pages_returned": len(evidences),
            "pages": page_diag,
            "suggested_next_steps": suggested_expansions,
            "signals": {
                "median_page_chars": int(median_chars) if "median_chars" in locals() else None,
                "median_page_chunks": int(median_chunks) if "median_chunks" in locals() else None,
            },
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)


__all__ = ["ReadPagesTool"]
