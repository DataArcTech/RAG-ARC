"""read.neighbors tool: read nearby chunks around a center chunk_id.

This is a deterministic, metadata-driven alternative to adding explicit NEXT_CHUNK edges.
It relies on:
- chunk metadata: `chunk_index` and ideally `source_file_id`.

Soft dependency on PageIndex:
- If metadata is missing, return an observable "neighbors_unavailable" diagnostic.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults
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


def _clamp_window(before: int, after: int) -> Tuple[int, int]:
    before = max(0, int(before))
    after = max(0, int(after))
    cap = int(tool_defaults.READ_NEIGHBORS_MAX_WINDOW)
    if cap > 0:
        before = min(before, cap)
        after = min(after, cap)
    return before, after


@dataclass(frozen=True)
class _ChunkRow:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]

    @property
    def chunk_index(self) -> int:
        token = _coerce_int(self.metadata.get("chunk_index"))
        return token if token is not None else 1_000_000


class ReadNeighborsTool(GraphTool):
    descriptor = ToolDescriptor(
        name="read.neighbors",
        channel="graph",
        description="Read nearby chunks around a center chunk_id using chunk_index ordering.",
        speed="fast",
        cost="low",
        strategy_tags=("read", "neighbors", "pageindex", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.read.neighbors",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "chunk_id": {"type": "string", "description": "Center chunk_id (required)."},
                "before": {"type": "integer", "minimum": 0, "description": "How many chunks before to include."},
                "after": {"type": "integer", "minimum": 0, "description": "How many chunks after to include."},
                "goal": {"type": "string", "description": "Why you are reading neighbors (for provenance)."},
                "max_chunks": {"type": "integer", "minimum": 1, "description": "Max chunks to include."},
                "max_chars": {"type": "integer", "minimum": 120, "description": "Max chars to include."},
            },
            required_extra_fields=("chunk_id",),
        ),
        example_args={
            "question": "Read surrounding context",
            "plan_step": "plan_01",
            "extra": {"chunk_id": "<chunk_id>", "before": 3, "after": 3, "goal": "capture procedure context"},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        if not pageindex_cfg.pageindex_enabled():
            return ToolResult(summary="read.neighbors skipped: PageIndex disabled.", diagnostics={"reason": "pageindex_disabled"})
        if request.adapter is None or not adapter_supports_cypher(request.adapter):
            return ToolResult(summary="read.neighbors skipped: Cypher unavailable.", diagnostics={"reason": "cypher_unavailable"})
        if request.access_scope is None:
            return ToolResult(summary="read.neighbors skipped: missing owner scope.", diagnostics={"reason": "owner_scope_missing"})

        extra = request.extra or {}
        center_chunk_id = str(extra.get("chunk_id") or "").strip()
        if not center_chunk_id:
            return ToolResult(summary="read.neighbors skipped: missing chunk_id.", diagnostics={"reason": "missing_chunk_id"})

        before = _coerce_int(extra.get("before"))
        after = _coerce_int(extra.get("after"))
        before = int(before) if before is not None else int(tool_defaults.READ_NEIGHBORS_DEFAULT_BEFORE)
        after = int(after) if after is not None else int(tool_defaults.READ_NEIGHBORS_DEFAULT_AFTER)
        before, after = _clamp_window(before, after)

        max_chunks = _coerce_int(extra.get("max_chunks")) or int(tool_defaults.READ_NEIGHBORS_DEFAULT_MAX_CHUNKS)
        max_chars = _coerce_int(extra.get("max_chars")) or int(tool_defaults.READ_NEIGHBORS_DEFAULT_MAX_CHARS)
        max_chunks = max(1, int(max_chunks))
        max_chars = max(120, int(max_chars))

        goal = str(extra.get("goal") or "").strip() or None

        # 1) Fetch center chunk (to get file_id + chunk_index).
        cypher_center = (
            "MATCH (c:Chunk)\n"
            "WHERE c.chunk_id = $chunk_id AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "RETURN c.chunk_id AS chunk_id, c.source_file_id AS source_file_id, c.metadata AS metadata\n"
            "LIMIT 1\n"
        )
        async with adapter_locked(request.adapter):
            center_rows = await request.adapter.acypher(
                cypher_center,
                {"chunk_id": center_chunk_id},
                access_scope=request.access_scope,
            )
        center = center_rows[0] if isinstance(center_rows, list) and center_rows and isinstance(center_rows[0], dict) else {}
        meta = _parse_metadata(center.get("metadata"))
        file_id = str(center.get("source_file_id") or meta.get("source_file_id") or meta.get("file_id") or "").strip()
        center_index = _coerce_int(meta.get("chunk_index"))

        diagnostics: Dict[str, Any] = {
            "chunk_id": center_chunk_id,
            "before": before,
            "after": after,
            "max_chunks": max_chunks,
            "max_chars": max_chars,
        }
        if file_id:
            diagnostics["file_id"] = file_id
        if center_index is None:
            diagnostics["reason"] = "missing_chunk_index"
            return ToolResult(
                summary="read.neighbors returned no chunks (missing chunk_index metadata).",
                diagnostics=diagnostics,
            )
        if not file_id:
            diagnostics["reason"] = "missing_file_id"
            return ToolResult(
                summary="read.neighbors returned no chunks (missing source_file_id metadata).",
                diagnostics=diagnostics,
            )

        start = max(0, int(center_index) - before)
        end = int(center_index) + after
        diagnostics["center_chunk_index"] = int(center_index)
        diagnostics["range_start"] = start
        diagnostics["range_end"] = end

        # 2) Scan file chunks and pick those within index window.
        scan_limit = int(tool_defaults.TOC_TREE_MAX_CHUNKS_SCANNED)
        cypher_scan = (
            "MATCH (c:Chunk)\n"
            "WHERE c.source_file_id = $file_id AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata\n"
            "LIMIT $limit\n"
        )
        async with adapter_locked(request.adapter):
            rows = await request.adapter.acypher(
                cypher_scan,
                {"file_id": file_id, "limit": max(1, scan_limit)},
                access_scope=request.access_scope,
            )

        candidates: List[_ChunkRow] = []
        missing_meta = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("chunk_id") or "").strip()
            if not cid:
                continue
            cmeta = _parse_metadata(row.get("metadata"))
            if not cmeta:
                missing_meta += 1
            idx = _coerce_int(cmeta.get("chunk_index"))
            if idx is None:
                continue
            if start <= idx <= end:
                candidates.append(_ChunkRow(chunk_id=cid, content=str(row.get("content") or ""), metadata=cmeta))

        diagnostics["rows_scanned"] = len(rows or [])
        diagnostics["scan_limit"] = scan_limit
        diagnostics["missing_metadata_rows"] = missing_meta
        diagnostics["matched"] = len(candidates)

        if not candidates:
            diagnostics["reason"] = "neighbors_not_found"
            return ToolResult(summary="read.neighbors returned no chunks.", diagnostics=diagnostics)

        candidates.sort(key=lambda r: (r.chunk_index, r.chunk_id))

        used_ids: List[str] = []
        parts: List[str] = []
        total_chars = 0
        truncated = False
        for row in candidates[:max_chunks]:
            block = row.content.strip()
            if not block:
                continue
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

        content = "".join(parts).strip()
        evidence = EvidenceChunk(
            chunk_id=hashed_chunk_id(source="read.neighbors", content=f"{center_chunk_id}:{start}-{end}:{len(used_ids)}"),
            source="read.neighbors",
            content=content,
            kind=EVIDENCE_KIND_PRIMARY,
            score=None,
            provenance={
                "goal": goal,
                "center_chunk_id": center_chunk_id,
                "source_file_id": file_id,
                "chunk_index_center": int(center_index),
                "chunk_index_start": start,
                "chunk_index_end": end,
                "chunk_ids": used_ids[:200],
                "truncated": truncated,
                "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
            },
        )
        diagnostics["used_chunk_count"] = len(used_ids)
        diagnostics["truncated"] = truncated
        summary = f"read.neighbors returned {len(used_ids)} chunks ({len(content)} chars)" + (" (truncated)." if truncated else ".")
        return ToolResult(summary=summary, evidences=[evidence], diagnostics=diagnostics)


__all__ = ["ReadNeighborsTool"]

