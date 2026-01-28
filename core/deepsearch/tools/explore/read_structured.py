"""Structured reading tools for long documents.

Why these exist:
- `search.*` returns snippets for localization; but long documents often require reading
  the full procedure/checklist/warning block (natural boundary) to avoid chunking loss.

Implementation note:
- We read from graph-stored `Chunk` nodes and filter by PageIndex-enriched metadata
  (`section_id`, `page_start/page_end`, `chunk_index`). This keeps runtime soft-dependent
  on PageIndex: when metadata is missing, the tool returns an observable empty result.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.core.deepsearch import tool_defaults
from config import pageindex as pageindex_cfg
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
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
    def section_id(self) -> str:
        return str(self.metadata.get("section_id") or "").strip()

    @property
    def page_start(self) -> Optional[int]:
        return _coerce_int(self.metadata.get("page_start"))

    @property
    def page_end(self) -> Optional[int]:
        return _coerce_int(self.metadata.get("page_end"))


async def _fetch_file_chunks(*, request: ToolRunRequest, file_id: str) -> Tuple[List[_ChunkRow], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"file_id": file_id}
    if not pageindex_cfg.pageindex_enabled():
        diagnostics["reason"] = "pageindex_disabled"
        return [], diagnostics
    if request.adapter is None or not adapter_supports_cypher(request.adapter):
        diagnostics["reason"] = "cypher_unavailable"
        return [], diagnostics
    if request.access_scope is None:
        diagnostics["reason"] = "owner_scope_missing"
        return [], diagnostics

    scan_limit = int(tool_defaults.TOC_TREE_MAX_CHUNKS_SCANNED)
    cypher = (
        "MATCH (c:Chunk)\n"
        "WHERE c.source_file_id = $file_id AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
        "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata\n"
        "LIMIT $limit\n"
    )
    params = {"file_id": file_id, "limit": max(1, scan_limit)}
    async with adapter_locked(request.adapter):
        rows = await request.adapter.acypher(cypher, params, access_scope=request.access_scope)

    parsed: List[_ChunkRow] = []
    missing = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        meta = _parse_metadata(row.get("metadata"))
        if not meta:
            missing += 1
        parsed.append(_ChunkRow(chunk_id=chunk_id, content=str(row.get("content") or ""), metadata=meta))
    diagnostics["rows_scanned"] = len(rows or [])
    diagnostics["scan_limit"] = scan_limit
    diagnostics["missing_metadata_rows"] = missing
    return parsed, diagnostics


def _truncate_joined(chunks: Sequence[_ChunkRow], *, max_chars: int, max_chunks: int) -> Tuple[str, List[str], bool]:
    used_ids: List[str] = []
    parts: List[str] = []
    truncated = False
    total_chars = 0
    for row in chunks[: max(0, max_chunks)]:
        if not row.content:
            continue
        block = row.content.strip()
        if not block:
            continue
        # Keep paragraph boundaries explicit to reduce "list item loss" in the LLM context.
        candidate = block if not parts else ("\n\n" + block)
        if max_chars > 0 and total_chars + len(candidate) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                snippet = candidate[:remaining].rstrip()
                if snippet:
                    parts.append(snippet)
            truncated = True
            break
        parts.append(candidate if not parts else candidate.lstrip("\n"))
        total_chars += len(candidate)
        used_ids.append(row.chunk_id)
    return "".join(parts).strip(), used_ids, truncated


class ReadSectionTool(GraphTool):
    descriptor = ToolDescriptor(
        name="read.section",
        channel="graph",
        description="Read all chunks belonging to a PageIndex section_id (natural boundary read, chunk-ordered).",
        speed="fast",
        cost="low",
        strategy_tags=("read", "section", "pageindex", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.read.section",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "Target file_id (required)."},
                "section_id": {"type": "string", "description": "Target section_id (required)."},
                "goal": {"type": "string", "description": "Why you are reading this section (for provenance)."},
                "max_chunks": {"type": "integer", "minimum": 1, "description": "Max chunks to include."},
                "max_chars": {"type": "integer", "minimum": 120, "description": "Max chars to include."},
            },
            required_extra_fields=("file_id", "section_id"),
        ),
        example_args={
            "question": "What are the safety precautions?",
            "plan_step": "plan_01",
            "extra": {"file_id": "<file_id>", "section_id": "<section_id>", "goal": "capture full warnings"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        file_id = str(extra.get("file_id") or "").strip()
        section_id = str(extra.get("section_id") or "").strip()
        goal = str(extra.get("goal") or "").strip() or None
        if not file_id or not section_id:
            return ToolResult(summary="read.section skipped: missing file_id/section_id.", diagnostics={"reason": "missing_args"})

        all_rows, fetch_diag = await _fetch_file_chunks(request=request, file_id=file_id)
        if not all_rows:
            return ToolResult(summary="read.section returned no chunks.", diagnostics={**fetch_diag, "reason": fetch_diag.get("reason") or "empty_file"})

        candidates = [row for row in all_rows if row.section_id == section_id]
        if not candidates:
            return ToolResult(
                summary="read.section returned no chunks (section_id not found; PageIndex metadata missing?).",
                diagnostics={**fetch_diag, "reason": "section_not_found", "section_id": section_id},
            )

        candidates.sort(key=lambda r: (r.chunk_index, r.chunk_id))
        max_chunks = _coerce_int(extra.get("max_chunks")) or int(tool_defaults.READ_SECTION_DEFAULT_MAX_CHUNKS)
        max_chars = _coerce_int(extra.get("max_chars")) or int(tool_defaults.READ_SECTION_DEFAULT_MAX_CHARS)
        max_chunks = max(1, max_chunks)
        max_chars = max(120, max_chars)
        content, used_ids, truncated = _truncate_joined(candidates, max_chars=max_chars, max_chunks=max_chunks)
        evidence = EvidenceChunk(
            chunk_id=hashed_chunk_id(source="read.section", content=f"{file_id}:{section_id}:{len(used_ids)}"),
            source="read.section",
            content=content,
            kind=EVIDENCE_KIND_PRIMARY,
            score=None,
            provenance={
                "goal": goal,
                "source_file_id": file_id,
                "section_id": section_id,
                "chunk_ids": used_ids[:200],
                "truncated": truncated,
                "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
            },
        )
        summary = f"read.section returned {len(used_ids)} chunks ({len(content)} chars)" + (" (truncated)." if truncated else ".")
        diagnostics = {**fetch_diag, "section_id": section_id, "used_chunk_count": len(used_ids), "truncated": truncated}
        return ToolResult(summary=summary, evidences=[evidence], diagnostics=diagnostics)


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
                "max_chunks": {"type": "integer", "minimum": 1, "description": "Max chunks to include."},
                "max_chars": {"type": "integer", "minimum": 120, "description": "Max chars to include."},
            },
            required_extra_fields=("file_id", "page_start", "page_end"),
        ),
        example_args={
            "question": "Read the warning page",
            "plan_step": "plan_01",
            "extra": {"file_id": "<file_id>", "page_start": 6, "page_end": 6, "goal": "capture warning text"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        file_id = str(extra.get("file_id") or "").strip()
        page_start = _coerce_int(extra.get("page_start"))
        page_end = _coerce_int(extra.get("page_end"))
        goal = str(extra.get("goal") or "").strip() or None
        if not file_id or page_start is None or page_end is None:
            return ToolResult(summary="read.pages skipped: missing file_id/page_start/page_end.", diagnostics={"reason": "missing_args"})

        if page_end < page_start:
            page_start, page_end = page_end, page_start

        all_rows, fetch_diag = await _fetch_file_chunks(request=request, file_id=file_id)
        if not all_rows:
            return ToolResult(summary="read.pages returned no chunks.", diagnostics={**fetch_diag, "reason": fetch_diag.get("reason") or "empty_file"})

        def _overlaps(row: _ChunkRow) -> bool:
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
            return ps <= page_end and pe >= page_start

        candidates = [row for row in all_rows if _overlaps(row)]
        if not candidates:
            return ToolResult(
                summary="read.pages returned no chunks (page metadata missing or range unmatched).",
                diagnostics={**fetch_diag, "reason": "pages_not_found", "page_start": page_start, "page_end": page_end},
            )

        candidates.sort(key=lambda r: (r.chunk_index, r.chunk_id))
        max_chunks = _coerce_int(extra.get("max_chunks")) or int(tool_defaults.READ_PAGES_DEFAULT_MAX_CHUNKS)
        max_chars = _coerce_int(extra.get("max_chars")) or int(tool_defaults.READ_PAGES_DEFAULT_MAX_CHARS)
        max_chunks = max(1, max_chunks)
        max_chars = max(120, max_chars)
        content, used_ids, truncated = _truncate_joined(candidates, max_chars=max_chars, max_chunks=max_chunks)
        evidence = EvidenceChunk(
            chunk_id=hashed_chunk_id(source="read.pages", content=f"{file_id}:{page_start}-{page_end}:{len(used_ids)}"),
            source="read.pages",
            content=content,
            kind=EVIDENCE_KIND_PRIMARY,
            score=None,
            provenance={
                "goal": goal,
                "source_file_id": file_id,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_ids": used_ids[:200],
                "truncated": truncated,
                "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
            },
        )
        summary = f"read.pages returned {len(used_ids)} chunks ({len(content)} chars)" + (" (truncated)." if truncated else ".")
        diagnostics = {
            **fetch_diag,
            "page_start": page_start,
            "page_end": page_end,
            "used_chunk_count": len(used_ids),
            "truncated": truncated,
        }
        return ToolResult(summary=summary, evidences=[evidence], diagnostics=diagnostics)


__all__ = ["ReadSectionTool", "ReadPagesTool"]
