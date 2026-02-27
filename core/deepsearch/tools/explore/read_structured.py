"""Structured reading tools for long documents.

DeepSearch contract:
- Routing/navigation outputs (locate/toc/tree) are navigation-only and MUST NOT be cited.
- `read.pages` is the primary evidence path and MUST return *full-page* content.

Implementation note:
- We read from graph-stored `Chunk` nodes and rely on PageIndex page metadata
  (`page_start/page_end`, `chunk_index`) to assemble page-level evidence.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import pageindex as pageindex_cfg
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from config.core.deepsearch import tool_defaults
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.deepsearch.utils.ids import normalize_uuid, resolve_file_ref
from core.deepsearch.utils.node_types import normalize_node_type
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher
from core.utils.json_extract import safe_json_loads
from framework.virtual_paths import is_io_path
from framework.virtual_paths import resolve_io_to_local_path

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


def _load_mineru_content_list_pages(
    *,
    file_id: str,
    page_start: int,
    page_end: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Fallback loader for MinerU `*_content_list*.json` when graph chunks are unavailable."""

    diagnostics: dict[str, Any] = {"file_id": file_id, "page_start": page_start, "page_end": page_end}
    base = str(os.getenv("PARSER_OUTPUT_DIR", "io://parsed_files") or "io://parsed_files").strip() or "io://parsed_files"

    raw_json: str | None = None
    candidate_ref: str | None = None

    if is_io_path(base):
        doc_dir = f"{base.rstrip('/')}/mineru/{file_id}"
        backend = str(os.getenv("IO_STORE_BACKEND", "localdb") or "localdb").strip().lower() or "localdb"
        if backend != "minio":
            # Unit-test/dev path: map io:// to local filesystem, honoring env overrides.
            try:
                doc_dir_local = resolve_io_to_local_path(doc_dir)
            except Exception as exc:  # noqa: BLE001
                diagnostics["reason"] = "resolve_io_failed"
                diagnostics["error"] = str(exc)
                return {}, diagnostics
            matches = sorted([p for p in doc_dir_local.glob("*_content_list*.json") if p.is_file()])
            if not matches:
                diagnostics["reason"] = "content_list_missing"
                return {}, diagnostics
            candidate_ref = str(matches[0])
            raw_json = matches[0].read_text(encoding="utf-8", errors="ignore")
        else:
            # MinIO path: must go through IOManager listing/reads.
            try:
                import app_registration

                io_manager = app_registration.registrator.get_object("io_manager")
            except Exception:
                io_manager = None
            if io_manager is None:
                diagnostics["reason"] = "io_manager_missing"
                return {}, diagnostics
            try:
                keys = io_manager.list_keys_path(doc_dir, limit=2000)
            except Exception as exc:  # noqa: BLE001
                diagnostics["reason"] = "list_keys_failed"
                diagnostics["error"] = str(exc)
                return {}, diagnostics

            def _basename(ref: str) -> str:
                token = str(ref or "").strip()
                return token.rsplit("/", 1)[-1] if "/" in token else token

            matches: list[str] = []
            for ref in keys or []:
                base_name = _basename(str(ref))
                if base_name.endswith(".json") and "_content_list" in base_name:
                    matches.append(str(ref))
            if not matches:
                diagnostics["reason"] = "content_list_missing"
                return {}, diagnostics
            candidate_ref = sorted(matches)[0]
            raw_json = io_manager.get_text_path(candidate_ref)
    else:
        # Local filesystem base (rare; mostly unit tests).
        try:
            doc_dir = (Path(base).expanduser().resolve() / "mineru" / file_id).resolve()
        except Exception:
            diagnostics["reason"] = "invalid_parser_output_dir"
            return {}, diagnostics
        if not doc_dir.exists():
            diagnostics["reason"] = "content_list_missing"
            return {}, diagnostics
        matches = sorted([p for p in doc_dir.glob("*_content_list*.json") if p.is_file()])
        if not matches:
            diagnostics["reason"] = "content_list_missing"
            return {}, diagnostics
        candidate_ref = str(matches[0])
        raw_json = matches[0].read_text(encoding="utf-8", errors="ignore")

    if raw_json is None:
        diagnostics["reason"] = "content_list_empty"
        diagnostics["candidate"] = candidate_ref
        return {}, diagnostics

    try:
        loaded = json.loads(raw_json)
    except Exception as exc:  # noqa: BLE001
        diagnostics["reason"] = "content_list_invalid_json"
        diagnostics["candidate"] = candidate_ref
        diagnostics["error"] = str(exc)
        return {}, diagnostics
    if not isinstance(loaded, list):
        diagnostics["reason"] = "content_list_unexpected_shape"
        diagnostics["candidate"] = candidate_ref
        return {}, diagnostics

    page_to_parts: dict[int, list[str]] = {}
    for row in loaded:
        if not isinstance(row, dict):
            continue
        idx = _coerce_int(row.get("page_idx"))
        if idx is None:
            continue
        if idx < int(page_start) or idx > int(page_end):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        page_to_parts.setdefault(int(idx), []).append(text)

    page_to_text: dict[int, str] = {}
    for idx, parts in page_to_parts.items():
        joined = "\n".join([p for p in parts if p]).strip()
        if joined:
            page_to_text[int(idx)] = joined

    diagnostics["candidate"] = candidate_ref
    diagnostics["page_count"] = len(page_to_text)
    return page_to_text, diagnostics


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
        description=(
            "Read the full text of specific pages from a file. This is the ONLY tool that returns "
            "citeable source evidence. All page indices are 0-based (page 0 = first page of the PDF). "
            "Always call this before concluding — navigation tools (locate, toc.tree, tree.*) only provide snippets."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("read", "pages", "pageindex", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.read.pages",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "The file_id (UUID) or filename to read from."},
                "page_start": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Start page index, 0-based inclusive (e.g. 0 = first page). page_start==page_end reads one page.",
                },
                "page_end": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "End page index, 0-based inclusive. Must be >= page_start. Omit to read a single page (=page_start).",
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alternative: list of 0-based page indices. Converted to page_start=min, page_end=max. Use page_start/page_end for ranges.",
                },
                "goal": {
                    "type": "string",
                    "description": "Brief note on why these pages are being read (e.g. 'extract revenue table'). Stored in provenance.",
                },
            },
            required_extra_fields=("file_id",),
        ),
        example_args={
            "question": "Read the warning page",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "page_start": 6, "page_end": 6, "goal": "capture warning text"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        file_id, file_id_raw = await resolve_file_ref(
            extra, adapter=request.adapter, access_scope=request.access_scope,
        )
        page_start = _coerce_int(extra.get("page_start"))
        page_end = _coerce_int(extra.get("page_end"))
        # Safety net: accept "pages" list from LLM and convert to range.
        if page_start is None and page_end is None:
            pages = extra.get("pages")
            if isinstance(pages, list) and pages:
                int_pages = [v for v in (_coerce_int(p) for p in pages) if v is not None]
                if int_pages:
                    page_start = min(int_pages)
                    page_end = max(int_pages)
        # Single page shorthand: page_start without page_end (or vice versa).
        if page_start is not None and page_end is None:
            page_end = page_start
        if page_end is not None and page_start is None:
            page_start = page_end
        goal = str(extra.get("goal") or "").strip() or None
        if not file_id or page_start is None or page_end is None:
            reason = "missing_args"
            if file_id_raw and not file_id:
                reason = "invalid_file_id_format"
            return ToolResult(
                summary="read.pages skipped: missing/invalid file_id or missing page range (use locate/page.select).",
                diagnostics={"reason": reason, "file_id_raw": file_id_raw or None, "page_start": page_start, "page_end": page_end},
            )

        if page_end < page_start:
            page_start, page_end = page_end, page_start

        # Soft cap advisory: track whether LLM requested more pages than recommended.
        pages_requested = int(page_end) - int(page_start) + 1
        soft_cap = int(getattr(tool_defaults, "READ_PAGES_SOFT_CAP_ADVISORY", 3) or 3)
        soft_cap_exceeded = pages_requested > soft_cap

        # PageIndex uses 0-based page indices; page 0 is valid.

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

        all_rows, fetch_diag = await _fetch_file_chunks(
            request=request,
            file_id=file_id,
            page_start=int(page_start),
            page_end=int(page_end),
        )
        if not all_rows:
            pages, diag = _load_mineru_content_list_pages(
                file_id=file_id,
                page_start=int(page_start),
                page_end=int(page_end),
            )
            if pages:
                evidences: list[EvidenceChunk] = []
                summaries: list[str] = []
                for p in range(int(page_start), int(page_end) + 1):
                    text = pages.get(p)
                    if not text:
                        continue
                    summaries.append(f"p{p}: content_list {len(text)} chars")
                    evidences.append(
                        EvidenceChunk(
                            chunk_id=hashed_chunk_id(source="read.pages", content=f"{file_id}:{p}-content_list"),
                            source="read.pages",
                            content=text,
                            kind=EVIDENCE_KIND_PRIMARY,
                            score=None,
                            provenance={
                                "goal": goal,
                                "source_file_id": file_id,
                                "page_start": p,
                                "page_end": p,
                                "chunk_ids": [],
                                "truncated": False,
                                "node_type": "page",
                                "node_types": {"text": 1},
                                "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
                                "metadata": {"source_file_id": file_id, "page_start": p, "page_end": p},
                            },
                        )
                    )
                summary = "read.pages returned full page evidence (content_list fallback): " + "; ".join(summaries)
                return ToolResult(
                    summary=summary,
                    evidences=evidences,
                    diagnostics={
                        **(fetch_diag or {}),
                        **(diag or {}),
                        "query_mode": "pageindex_content_list_direct",
                        "page_indexing": "0_based",
                    },
                )

            return ToolResult(
                summary="read.pages returned no chunks.",
                diagnostics={
                    **(fetch_diag or {}),
                    **(diag or {}),
                    "reason": (fetch_diag or {}).get("reason") or (diag or {}).get("reason") or "empty_file",
                    "page_indexing": "0_based",
                },
            )

        filename = _extract_filename(all_rows)
        evidences: list[EvidenceChunk] = []
        page_summaries: list[str] = []
        page_diag: dict[int, dict[str, Any]] = {}
        page_char_counts: dict[int, int] = {}
        for page in range(int(page_start), int(page_end) + 1):
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

        if page_summaries:
            fetch_diag = dict(fetch_diag or {})
            fetch_diag["page_summaries"] = list(page_summaries)[:20]

        if not evidences:
            return ToolResult(
                summary="read.pages returned no chunks (page metadata missing or range unmatched).",
                diagnostics={**fetch_diag, "reason": "pages_not_found", "page_start": int(page_start), "page_end": int(page_end), "page_indexing": "0_based"},
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

            page_reasons: dict[int, set[str]] = {}
            for _p, info in page_diag.items():
                if not isinstance(info, dict):
                    continue
                node_types = info.get("node_types") or {}
                char_count = int(info.get("char_count") or 0)
                used_chunk_count = int(info.get("used_chunk_count") or 0)

                reasons: set[str] = set()
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
                    page_reasons[int(_p)] = reasons

            # Directional continuity hint (navigation-only):
            # - Prefer expanding only toward the boundary that shows "continuation risk"
            #   (reduces needless reads vs symmetric expansion).
            # - Only suggest bidirectional expansion when the evidence likely spans pages
            #   (tables/equations on boundary pages).
            edge_start = int(page_start)
            edge_end = int(page_end)
            start_reasons = page_reasons.get(edge_start, set())
            end_reasons = page_reasons.get(edge_end, set())

            def _has_table_or_equation(rs: set[str]) -> bool:
                return "table_or_equation_detected" in rs

            direction: str | None = None
            used_reasons: set[str] = set()

            if start_reasons or end_reasons:
                if _has_table_or_equation(start_reasons) or _has_table_or_equation(end_reasons):
                    direction = "both"
                    used_reasons = set(start_reasons) | set(end_reasons)
                    lo = max(0, edge_start - delta)
                    hi = edge_end + delta
                else:
                    start_score = len(start_reasons)
                    end_score = len(end_reasons)
                    # Tie-breaker: prefer expanding forward (reading order).
                    if end_score >= start_score:
                        direction = "forward"
                        used_reasons = set(end_reasons) if end_reasons else set(start_reasons)
                        lo = edge_end + 1
                        hi = edge_end + delta
                    else:
                        direction = "backward"
                        used_reasons = set(start_reasons)
                        lo = max(0, edge_start - delta)
                        hi = edge_start - 1

                if direction == "backward" and hi < lo:
                    # Cannot expand backward beyond page 0; prefer a forward expansion instead.
                    direction = "forward"
                    used_reasons = set(end_reasons) if end_reasons else set(start_reasons)
                    lo = edge_end + 1
                    hi = edge_end + delta

                if direction and lo <= hi:
                    suggested_expansions.append(
                        {
                            "tool": "read.pages",
                            "args": {
                                "file_id": file_id,
                                "page_start": lo,
                                "page_end": hi,
                                "goal": "expand contiguous pages for continuity (navigation hint)",
                            },
                            "direction": direction,
                            "reason": ",".join(sorted(used_reasons)) if used_reasons else None,
                        }
                    )
                    reason_text = ",".join(sorted(used_reasons)) if used_reasons else "unknown"
                    summary = summary.rstrip() + f" TIP: continuity hint (direction={direction}; {reason_text}); consider expanding to p{lo}-p{hi}."
        if soft_cap_exceeded:
            summary = summary.rstrip() + (
                f" NOTE: You requested {pages_requested} pages (recommended: ≤{soft_cap})."
                " Prefer targeted 1-3 page reads based on locate/toc.tree results."
            )
        diagnostics = {
            **fetch_diag,
            "page_start": page_start,
            "page_end": page_end,
            "pages_requested": pages_requested,
            "soft_cap_exceeded": soft_cap_exceeded,
            "pages_returned": len(evidences),
            "pages": page_diag,
            "suggested_next_steps": suggested_expansions,
            "signals": {
                "median_page_chars": int(median_chars) if "median_chars" in locals() else None,
                "median_page_chunks": int(median_chunks) if "median_chunks" in locals() else None,
            },
            "page_indexing": "0_based",
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)


__all__ = ["ReadPagesTool"]
