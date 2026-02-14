"""LLM-backed tree search for long documents."""
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults
from config.core.deepsearch import runtime_cache_defaults
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.deepsearch.utils.section_tree import (
    build_section_tree,
    fetch_section_nodes,
    fetch_section_tree_fingerprint,
    normalize_file_id,
)
from core.deepsearch.utils.file_scope import FileScope
from core.deepsearch.entity_resolution import build_default_entity_resolver
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher
from core.prompts.deepsearch import SEARCH_ENTITY_EXTRACT_PROMPT_EN
from core.prompts.deepsearch.section_select import (
    SECTION_SELECT_CONSUMER_SYSTEM_PROMPT,
    SECTION_SELECT_CONSUMER_USER_PROMPT,
    SECTION_TREE_SEARCH_SYSTEM_PROMPT,
    SECTION_TREE_SEARCH_USER_PROMPT,
)
from core.retrieval.pageindex_retriever import PageIndexRetriever
from framework.cache import TTLRUCache

from .graph_ops.graph_ops_common import normalize_entity_name
from ..base import (
    GraphTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
    build_input_schema,
)
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from core.deepsearch.utils.llm_json import call_llm_json_with_retry
from .search.base import _SearchToolBase
from .search.faiss import _FaissChannel
from .search.bm25 import _Bm25Channel


@dataclass(frozen=True)
class _CandidateSection:
    section_id: str
    score: float
    title: str | None
    path: str | None
    summary: str | None
    page_start: Optional[int]
    page_end: Optional[int]
    source: str | None = None
    index_score: Optional[float] = None
    node_score: Optional[float] = None
    graph_score: Optional[float] = None
    graph_entities: Optional[List[str]] = None
    hit_count: Optional[int] = None
    chunk_count: Optional[int] = None


def _split_section_card(text: str) -> tuple[str, str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", "", ""
    lines = [line.rstrip() for line in raw.splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return "", "", ""
    title = lines[0].strip()
    path = lines[1].strip() if len(lines) >= 2 else ""
    summary = "\n".join(lines[2:]).strip() if len(lines) >= 3 else ""
    return title, path, summary


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for token in values:
        tid = str(token or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def _suggest_read_pages_from_section_ids(
    *,
    file_id: str,
    section_ids: Sequence[str],
    section_map: Dict[str, Dict[str, Any]],
    max_items: int,
) -> List[Dict[str, Any]]:
    """Suggest read.pages calls from section page hints (navigation only; citeable evidence comes from read.pages)."""

    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for sid in section_ids:
        if max_items > 0 and len(out) >= max_items:
            break
        meta = section_map.get(sid) or {}
        ps = _coerce_int(meta.get("page_start"))
        pe = _coerce_int(meta.get("page_end"))
        if ps is None or pe is None:
            continue
        if pe < ps:
            ps, pe = pe, ps
        key = (file_id, ps, pe)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "tool": "read.pages",
                "args": {"file_id": file_id, "page_start": ps, "page_end": pe},
                "why": "expand_full_context_for_selected_section",
            }
        )
    return out


class SectionSelectTool(_SearchToolBase, _FaissChannel, _Bm25Channel, GraphTool):
    descriptor = ToolDescriptor(
        name="section.select",
        channel="graph",
        description=(
            "LLM-assisted tree search for long documents. "
            "Runs hybrid tree search (LLM node list + value-based node scores) with a queue + consumer loop "
            "to decide which sections/pages to read next. Output is derived guidance (not citable)."
        ),
        speed="medium",
        cost="medium",
        strategy_tags=("section_select", "pageindex", "toc", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="X",
        determinism="llm_light",
        namespace="rag-arc.deepsearch.tools.section.select",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "file_id": {"type": "string", "description": "Target file_id (required)."},
                "focus_query": {"type": "string", "description": "Optional query override."},
                "top_k": {"type": "integer", "minimum": 1, "description": "How many section candidates to consider."},
                "max_rounds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Max consumer rounds for hybrid tree search.",
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How many nodes to consume per round.",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Max subtree depth to expand from selected sections (0 = selected only).",
                },
            },
            required_extra_fields=("file_id",),
        ),
        example_args={
            "question": "What are the cancellation conditions?",
            "plan_step": "plan_01",
            "extra": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "top_k": 8, "max_rounds": 3, "batch_size": 6, "max_depth": 2},
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        pageindex_retriever: PageIndexRetriever | None = None,
        temperature: float | None = None,
        dense_retriever=None,
        bm25_retriever=None,
        section_nodes_cache_enabled: bool | None = None,
        section_nodes_cache_max_entries: int | None = None,
        section_nodes_cache_ttl_seconds: float | None = None,
    ) -> None:
        _SearchToolBase.__init__(self, llm_connector=llm_connector, dense_retriever=dense_retriever, bm25_retriever=bm25_retriever)
        self.llm_connector = llm_connector
        self._pageindex = pageindex_retriever or (PageIndexRetriever() if pageindex_cfg.pageindex_enabled() else None)
        if temperature is None:
            temperature = tool_defaults.SECTION_SELECT_TEMPERATURE
        self.temperature = float(temperature)
        if section_nodes_cache_enabled is None:
            section_nodes_cache_enabled = runtime_cache_defaults.DEFAULT_SECTION_NODES_CACHE_ENABLED
        if section_nodes_cache_max_entries is None:
            section_nodes_cache_max_entries = runtime_cache_defaults.DEFAULT_SECTION_NODES_CACHE_MAX_ENTRIES
        if section_nodes_cache_ttl_seconds is None:
            section_nodes_cache_ttl_seconds = float(runtime_cache_defaults.DEFAULT_SECTION_NODES_CACHE_TTL_SECONDS)
        self._section_nodes_cache: TTLRUCache[tuple[str, str, str], Dict[str, Any]] | None = None
        if section_nodes_cache_enabled and section_nodes_cache_max_entries and section_nodes_cache_ttl_seconds:
            try:
                self._section_nodes_cache = TTLRUCache(
                    max_entries=int(section_nodes_cache_max_entries),
                    ttl_seconds=float(section_nodes_cache_ttl_seconds),
                )
            except Exception:
                self._section_nodes_cache = None

    async def run(self, request: ToolRunRequest) -> ToolResult:
        if self.llm_connector is None:
            return ToolResult(
                summary="section.select skipped: LLM connector unavailable.",
                diagnostics={"reason": "llm_unavailable"},
            )
        if not pageindex_cfg.pageindex_enabled():
            return ToolResult(summary="section.select skipped: PageIndex disabled.", diagnostics={"reason": "pageindex_disabled"})
        if request.access_scope is None or not getattr(request.access_scope, "scope_id", None):
            return ToolResult(summary="section.select skipped: missing owner scope.", diagnostics={"reason": "owner_scope_missing"})

        extra = request.extra or {}
        file_id, file_id_raw = normalize_file_id(extra.get("file_id"))
        if not file_id:
            reason = "missing_file_id" if not file_id_raw else "invalid_file_id_format"
            return ToolResult(
                summary="section.select skipped: invalid/missing file_id (expected UUID; use search.file).",
                diagnostics={"reason": reason, "file_id_raw": file_id_raw or None},
            )

        query = str(extra.get("focus_query") or extra.get("query") or request.question or "").strip()
        if not query:
            return ToolResult(summary="section.select skipped: empty query.", diagnostics={"reason": "empty_query", "file_id": file_id})

        sections: list[Any] = []
        fetch_diag: Dict[str, Any] = {}
        if self._section_nodes_cache is not None:
            try:
                owner_scope_id = str(getattr(request.access_scope, "scope_id", "") or "")
            except Exception:
                owner_scope_id = ""
            if owner_scope_id:
                fp, fp_diag = await fetch_section_tree_fingerprint(
                    adapter=request.adapter,
                    access_scope=request.access_scope,
                    file_id=file_id,
                )
                if fp:
                    key = (owner_scope_id, file_id, fp)
                    cached = self._section_nodes_cache.get(key)
                    if isinstance(cached, dict) and isinstance(cached.get("sections"), list):
                        sections = cached.get("sections") or []
                        fetch_diag = dict(cached.get("fetch_diag") or {})
                        fetch_diag.setdefault("cache", {})
                        fetch_diag["cache"] = {"hit": True, "fingerprint": fp, "fingerprint_diag": fp_diag}
                    else:
                        sections, fetch_diag = await fetch_section_nodes(
                            adapter=request.adapter,
                            access_scope=request.access_scope,
                            file_id=file_id,
                            file_id_raw=file_id_raw,
                            include_node_types=True,
                        )
                        fetch_diag.setdefault("cache", {})
                        fetch_diag["cache"] = {"hit": False, "fingerprint": fp, "fingerprint_diag": fp_diag}
                        try:
                            self._section_nodes_cache.set(
                                key,
                                {"sections": list(sections), "fetch_diag": dict(fetch_diag)},
                            )
                        except Exception:
                            pass
                else:
                    sections, fetch_diag = await fetch_section_nodes(
                        adapter=request.adapter,
                        access_scope=request.access_scope,
                        file_id=file_id,
                        file_id_raw=file_id_raw,
                        include_node_types=True,
                    )
        else:
            sections, fetch_diag = await fetch_section_nodes(
                adapter=request.adapter,
                access_scope=request.access_scope,
                file_id=file_id,
                file_id_raw=file_id_raw,
                include_node_types=True,
            )
        if not sections:
            reason = fetch_diag.get("reason") or "no_sections_found"
            diagnostics = {**fetch_diag, "reason": reason}
            if reason == "pageindex_disabled":
                return ToolResult(summary="section.select skipped: PageIndex disabled.", diagnostics=diagnostics)
            if reason == "cypher_unavailable":
                return ToolResult(summary="section.select skipped: Cypher unavailable.", diagnostics=diagnostics)
            if reason == "owner_scope_missing":
                return ToolResult(summary="section.select skipped: missing owner scope.", diagnostics=diagnostics)
            return ToolResult(summary="section.select skipped: no sections available for file.", diagnostics=diagnostics)

        max_sections = max(1, int(tool_defaults.SECTION_SELECT_MAX_SECTIONS))
        sections = sections[:max_sections]
        section_map: Dict[str, Dict[str, Any]] = {}
        for node in sections:
            section_map[node.section_id] = {
                "section_id": node.section_id,
                "section_path": node.path,
                "section_title": node.title,
                "section_level": node.level,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "parent_id": node.parent_id,
                "node_types": dict(node.node_types or {}),
            }
        tree, printed, orphaned = build_section_tree(sections)

        top_k = int(extra.get("top_k") or tool_defaults.SECTION_SELECT_CANDIDATE_TOP_K)
        top_k = max(1, top_k)
        index_candidates = await self._retrieve_candidates(
            query=query,
            file_id=file_id,
            owner_id=str(request.access_scope.scope_id),
            top_k=top_k,
        )
        value_candidates = await self._retrieve_value_candidates(
            request=request,
            query=query,
            file_id=file_id,
            top_k=top_k,
            section_map=section_map,
        )
        graph_candidates, graph_diag = await self._graph_entity_candidates(
            request=request,
            query=query,
            file_id=file_id,
            section_map=section_map,
            top_k=top_k,
        )
        section_ids = set(section_map.keys())
        index_candidates = [c for c in index_candidates if c.section_id in section_ids]
        value_candidates = [c for c in value_candidates if c.section_id in section_ids]
        graph_candidates = [c for c in graph_candidates if c.section_id in section_ids]
        candidates = self._merge_candidates(index_candidates, value_candidates, graph_candidates)[:top_k]
        candidate_ids = [c.section_id for c in candidates]
        entity_hints = await self._collect_entity_hints(
            request=request,
            file_id=file_id,
            section_ids=candidate_ids,
        )
        # Attach multi-signal hints onto section_map so downstream LLM steps (consumer) can see
        # *why* a node is in the queue (graph/index/value signals). These are navigation-only hints.
        self._attach_candidate_signals(section_map, candidates)
        for cand in candidates:
            if cand.summary:
                section_map[cand.section_id]["summary"] = cand.summary

        catalog = self._build_catalog_payload(section_map, candidates, entity_hints)
        tree_payload = self._enrich_tree(tree, section_map, entity_hints)
        llm_nodes, tree_diag = await self._run_llm_tree_search(
            question=query,
            file_id=file_id,
            tree_payload=tree_payload,
            candidates_payload=catalog["candidates"],
            section_map=section_map,
        )
        queue_ids, queue_items = self._build_queue(
            llm_nodes=llm_nodes,
            index_candidates=index_candidates,
            value_candidates=value_candidates,
            graph_candidates=graph_candidates,
            section_map=section_map,
        )
        # Persist queue source/scores for consumer routing decisions.
        self._attach_queue_meta(section_map, queue_items)
        if queue_ids:
            missing_hints = [sid for sid in queue_ids if sid not in entity_hints]
            if missing_hints:
                more_hints = await self._collect_entity_hints(
                    request=request,
                    file_id=file_id,
                    section_ids=missing_hints,
                )
                if more_hints:
                    merged = dict(entity_hints)
                    merged.update(more_hints)
                    entity_hints = merged

        consumer = await self._consume_queue(
            question=query,
            queue_ids=queue_ids,
            section_map=section_map,
            entity_hints=entity_hints,
            max_rounds=_coerce_int(extra.get("max_rounds")) or int(tool_defaults.SECTION_SELECT_MAX_ROUNDS),
            batch_size=_coerce_int(extra.get("batch_size")) or int(tool_defaults.SECTION_SELECT_CONSUMER_BATCH_SIZE),
        )
        selection = consumer["selection"]
        parse_diag = consumer["parse_diagnostics"]
        if not selection["primary_section_ids"] and candidate_ids:
            selection["primary_section_ids"] = [candidate_ids[0]]
            selection["explanation"] = selection.get("explanation") or "fallback to top candidate section."
            if isinstance(parse_diag, list):
                parse_diag.append({"fallback": "top_candidate"})

        selected_ids = _dedupe_keep_order(selection["primary_section_ids"] + selection["supplementary_section_ids"])
        max_depth = _coerce_int(extra.get("max_depth"))
        if max_depth is None:
            max_depth = int(tool_defaults.SECTION_SELECT_DEFAULT_MAX_DEPTH)
        max_depth = max(0, int(max_depth))
        subtree_ids = self._expand_subtree(tree, selected_ids, max_depth=max_depth)
        selected_entity_hints = {sid: entity_hints[sid] for sid in selected_ids if sid in entity_hints}
        selected_node_types = {sid: section_map.get(sid, {}).get("node_types", {}) for sid in selected_ids}
        subtree_entity_hints = await self._collect_subtree_entity_hints(
            request=request,
            file_id=file_id,
            section_ids=subtree_ids,
        )
        seed_entities: List[str] = []
        for values in subtree_entity_hints.values():
            seed_entities.extend(values)
        seed_entities = _dedupe_keep_order(seed_entities)
        max_seeds = int(tool_defaults.SECTION_SELECT_SEED_MAX)
        if max_seeds > 0:
            seed_entities = seed_entities[:max_seeds]
        payload = {
            "query": query,
            "file_id": file_id,
            "selected_section_ids": selected_ids,
            "primary_section_ids": selection["primary_section_ids"],
            "supplementary_section_ids": selection["supplementary_section_ids"],
            "subtree_section_ids": subtree_ids,
            "subtree_max_depth": max_depth,
            "explanation": selection.get("explanation") or "",
            "queue_ids": queue_ids,
            "queue_items": queue_items,
            "consumer_rounds": consumer["rounds"],
            "enough_info": consumer["enough_info"],
            "section_entity_hints": selected_entity_hints,
            "section_node_types": selected_node_types,
            "subtree_entity_hints": subtree_entity_hints,
            "seed_entities": seed_entities,
        }
        suggested_reads = _suggest_read_pages_from_section_ids(
            file_id=file_id,
            section_ids=selection["primary_section_ids"] or selected_ids,
            section_map=section_map,
            max_items=max(0, int(tool_defaults.SEARCH_SUGGESTED_READ_MAX)),
        )
        payload["suggested_reads"] = suggested_reads
        payload["suggested_actions"] = [
            {
                "tool": "search.scoped.graph",
                "args": {
                    "file_id": file_id,
                    "section_ids": subtree_ids,
                    "seed_entities": seed_entities,
                    "focus_query": query,
                },
                "why": "Project graph retrieval onto the selected subtree to avoid drifting across unrelated sections/files.",
            },
            *(
                [
                    {
                        "tool": "search.global.graph",
                        "args": {
                            "seed_entities": seed_entities,
                            "focus_query": query,
                            "top_k": int(getattr(tool_defaults, "SECTION_SELECT_CROSS_DOC_BRIDGE_TOP_K", 10) or 10),
                            "use_ppr": False,
                            "enable_llm_rerank": False,
                            "enable_entity_fallback": False,
                        },
                        "why": "Cross-document graph bridging (routing only): discover other files likely related to the current subtree via shared entities; then route to the new file_id(s) and read.pages for evidence.",
                    }
                ]
                if seed_entities and bool(getattr(tool_defaults, "SECTION_SELECT_CROSS_DOC_BRIDGE_SUGGEST_ENABLED", True))
                else []
            ),
            *suggested_reads,
        ]
        summary = self._build_summary(section_map, payload)
        evidence = self._build_evidence(request=request, payload=payload, summary=summary)
        diagnostics = {
            **fetch_diag,
            "tree": tree_payload,
            "printed_nodes": printed,
            "orphaned_nodes": orphaned,
            "candidates": [c.__dict__ for c in candidates],
            "graph_candidates": [c.__dict__ for c in graph_candidates],
            "graph_candidate_diag": graph_diag,
            "entity_hints": entity_hints,
            "subtree_entity_hints": subtree_entity_hints,
            "seed_entities": seed_entities,
            "selection": payload,
            "parse_diagnostics": parse_diag,
            "tree_search": tree_diag,
            "queue_items": queue_items,
            "consumer_rounds": consumer["rounds"],
        }
        return ToolResult(summary=summary, evidences=[evidence], diagnostics=diagnostics)

    async def _retrieve_candidates(
        self,
        *,
        query: str,
        file_id: str,
        owner_id: str,
        top_k: int,
    ) -> List[_CandidateSection]:
        if self._pageindex is None or not pageindex_cfg.section_index_enabled():
            return []
        hits = self._pageindex.retrieve_sections(query, owner_id=owner_id, file_ids=[file_id])
        candidates: List[_CandidateSection] = []
        for hit in hits:
            meta = getattr(hit, "metadata", None) or {}
            section_id = str(meta.get("section_id") or getattr(hit, "id", "")).strip()
            if not section_id:
                continue
            score = meta.get("score")
            score_f = float(score) if isinstance(score, (int, float)) else 0.0
            title, path, summary = _split_section_card(getattr(hit, "content", ""))
            candidates.append(
                _CandidateSection(
                    section_id=section_id,
                    score=score_f,
                    title=title or str(meta.get("section_title") or "").strip() or None,
                    path=path or str(meta.get("section_path") or "").strip() or None,
                    summary=summary or None,
                    page_start=_coerce_int(meta.get("page_start")),
                    page_end=_coerce_int(meta.get("page_end")),
                    source="section_index",
                    index_score=score_f,
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    async def _retrieve_value_candidates(
        self,
        *,
        request: ToolRunRequest,
        query: str,
        file_id: str,
        top_k: int,
        section_map: Dict[str, Dict[str, Any]],
    ) -> List[_CandidateSection]:
        channels = list(getattr(tool_defaults, "SECTION_SELECT_VALUE_CHANNELS", ("faiss", "bm25")) or [])
        channels = [str(ch).strip().lower() for ch in channels if str(ch or "").strip()]
        if not channels:
            return []
        top_k_chunks = int(getattr(tool_defaults, "SECTION_SELECT_VALUE_TOP_K", 0))
        if top_k_chunks <= 0:
            return []

        file_scope = FileScope(file_ids=frozenset({file_id}), filename_contains=(), source="section_select")
        channel_results: Dict[str, Any] = {}
        hits: Dict[str, Dict[str, Any]] = {}

        async def _consume(channel: str, evidences: Sequence[EvidenceChunk]) -> None:
            for ev in evidences or []:
                prov = ev.provenance if isinstance(ev.provenance, dict) else {}
                meta = prov.get("metadata") if isinstance(prov.get("metadata"), dict) else {}
                section_id = str(meta.get("section_id") or "").strip()
                if not section_id:
                    chunk_meta = meta.get("chunk_metadata")
                    if isinstance(chunk_meta, dict):
                        section_id = str(chunk_meta.get("section_id") or "").strip()
                if not section_id:
                    continue
                chunk_id = str(ev.chunk_id or "").strip()
                if not chunk_id:
                    continue
                bucket = hits.setdefault(
                    section_id,
                    {
                        "scores": [],
                        "chunk_ids": set(),
                        "hit_count": 0,
                        "channel_hits": {},
                    },
                )
                if chunk_id in bucket["chunk_ids"]:
                    continue
                bucket["chunk_ids"].add(chunk_id)
                score = ev.score if isinstance(ev.score, (int, float)) else 0.0
                bucket["scores"].append(float(score))
                bucket["hit_count"] += 1
                channel_hits = dict(bucket["channel_hits"])
                channel_hits[channel] = int(channel_hits.get(channel) or 0) + 1
                bucket["channel_hits"] = channel_hits

        if "faiss" in channels:
            result = await self._run_faiss(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope)
            channel_results["faiss"] = result.diagnostics
            await _consume("faiss", result.evidences)
        if "bm25" in channels:
            result = await self._run_bm25(request=request, query=query, top_k=top_k_chunks, file_scope=file_scope)
            channel_results["bm25"] = result.diagnostics
            await _consume("bm25", result.evidences)

        if not hits:
            return []

        chunk_counts = await self._fetch_section_chunk_counts(
            request=request,
            file_id=file_id,
            section_ids=list(hits.keys()),
        )

        candidates: List[_CandidateSection] = []
        for section_id, bucket in hits.items():
            scores = list(bucket.get("scores") or [])
            if not scores:
                continue
            chunk_total = chunk_counts.get(section_id)
            if not isinstance(chunk_total, int) or chunk_total < 0:
                chunk_total = len(scores)
            node_score = sum(scores) / math.sqrt(chunk_total + 1)
            meta = section_map.get(section_id) or {}
            candidates.append(
                _CandidateSection(
                    section_id=section_id,
                    score=node_score,
                    title=str(meta.get("section_title") or meta.get("section_path") or "").strip() or None,
                    path=str(meta.get("section_path") or "").strip() or None,
                    summary=None,
                    page_start=_coerce_int(meta.get("page_start")),
                    page_end=_coerce_int(meta.get("page_end")),
                    source="value_node_score",
                    node_score=node_score,
                    hit_count=int(bucket.get("hit_count") or 0),
                    chunk_count=int(chunk_total),
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    async def _extract_query_entities(self, *, query: str, limit: int) -> List[str]:
        if limit <= 0:
            return []
        messages = [
            {"role": "system", "content": SEARCH_ENTITY_EXTRACT_PROMPT_EN},
            {"role": "user", "content": f"Query: {query}"},
        ]
        kwargs: Dict[str, Any] = {
            "temperature": float(tool_defaults.SEARCH_ENTITY_EXTRACT_TEMPERATURE),
            "max_tokens": int(tool_defaults.SEARCH_ENTITY_EXTRACT_MAX_TOKENS),
        }
        payload = await call_llm_json_with_retry(
            llm_connector=self.llm_connector,
            messages=messages,
            expected="object",
            temperature=float(kwargs["temperature"]),
            max_tokens=int(kwargs["max_tokens"]),
        )
        if not isinstance(payload, dict):
            return []
        # Online term extraction: accept both {terms:[...]} (preferred) and legacy {entities:[...]} payloads.
        items = payload.get("terms")
        if not isinstance(items, list):
            items = payload.get("entities")
        if not isinstance(items, list):
            return []
        results: List[str] = []
        for item in items:
            if len(results) >= int(limit):
                break
            if isinstance(item, str):
                token = item.strip()
                if token:
                    results.append(token)
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("name") or item.get("term") or "").strip()
                if text:
                    results.append(text)
        return _dedupe_keep_order(results)

    async def _graph_entity_candidates(
        self,
        *,
        request: ToolRunRequest,
        query: str,
        file_id: str,
        section_map: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[_CandidateSection], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {"enabled": True}
        adapter = request.adapter
        if adapter is None or not adapter_supports_cypher(adapter):
            diagnostics["reason"] = "cypher_unavailable"
            return [], diagnostics

        max_terms = int(getattr(tool_defaults, "SECTION_SELECT_GRAPH_ENTITY_MATCH_MAX_TERMS", 6))
        max_terms = max(1, max_terms)
        limit = int(tool_defaults.SEARCH_GRAPH_ENTITY_SEED_TOP_K)
        entities = await self._extract_query_entities(query=query, limit=limit)
        diagnostics["query_entities"] = entities

        # Normalize extracted entities into graph terms (for entity_name_normalized / alias_text_normalized matching).
        terms: List[str] = []
        for ent in entities:
            token = normalize_entity_name(ent)
            if token:
                terms.append(token)
        terms = _dedupe_keep_order(terms)[:max_terms]

        max_sections = max(1, int(tool_defaults.SECTION_SELECT_MAX_SECTIONS))
        top_k = max(1, int(top_k))
        limit_sections = min(max_sections, top_k)

        rows: List[Dict[str, Any]] = []
        if terms:
            enable_contains = bool(getattr(tool_defaults, "SECTION_SELECT_GRAPH_ENTITY_MATCH_ENABLE_CONTAINS", True))
            entity_limit = int(getattr(tool_defaults, "SECTION_SELECT_GRAPH_ENTITY_MATCH_PER_TERM_ENTITY_LIMIT", 8))
            entity_limit = max(1, min(entity_limit, 50))
            mode_clause = " OR e.entity_name_normalized CONTAINS term" if enable_contains else ""

            cypher = (
                "UNWIND $terms AS term\n"
                "CALL {\n"
                "  WITH term\n"
                "  MATCH (e:Entity)\n"
                "  WHERE COALESCE(e.owner_id, $global_owner) = $owner_id\n"
                "    AND (e.entity_name_normalized = term" + mode_clause + ")\n"
                "  RETURN collect(DISTINCT e)[..$entity_limit] AS entities\n"
                "}\n"
                "CALL {\n"
                "  WITH term\n"
                "  OPTIONAL MATCH (a:EntityAlias)-[:ALIAS_OF]->(c:EntityCanonical)<-[:CANONICAL_OF]-(e2:Entity)\n"
                "  WHERE COALESCE(a.owner_id, $global_owner) = $owner_id\n"
                "    AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
                "    AND COALESCE(e2.owner_id, $global_owner) = $owner_id\n"
                "    AND a.alias_text_normalized CONTAINS term\n"
                "  RETURN collect(DISTINCT e2)[..$entity_limit] AS alias_entities\n"
                "}\n"
                "WITH term, (entities + alias_entities) AS es\n"
                "UNWIND es AS e\n"
                "MATCH (e)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(s:Section)\n"
                "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
                "RETURN s.section_id AS section_id,\n"
                "       count(DISTINCT c.chunk_id) AS mentions,\n"
                "       collect(DISTINCT term)[..$term_limit] AS matched_terms,\n"
                "       collect(DISTINCT e.entity_name)[..$term_limit] AS matched_entities\n"
                "ORDER BY mentions DESC\n"
                "LIMIT $limit\n"
            )
            params = {
                "terms": list(terms),
                "file_id": file_id,
                "limit": int(limit_sections),
                "entity_limit": entity_limit,
                "term_limit": max(1, int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX)),
            }
            async with adapter_locked(adapter):
                rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)
            diagnostics["graph_term_mode"] = {
                "terms": terms,
                "enable_contains": enable_contains,
                "entity_limit": entity_limit,
            }
        else:
            diagnostics["graph_term_mode"] = {"terms": [], "reason": "no_query_entities"}

        candidates: List[_CandidateSection] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            if not section_id or section_id not in section_map:
                continue
            try:
                mentions = float(row.get("mentions") or 0.0)
            except Exception:
                mentions = 0.0
            matched = row.get("matched_entities") if isinstance(row.get("matched_entities"), list) else []
            matched_entities = [str(item).strip() for item in matched if str(item or "").strip()]
            meta = section_map.get(section_id) or {}
            summary = None
            if matched_entities:
                preview = matched_entities[: max(1, int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX))]
                summary = "graph_entities: " + ", ".join(preview)
            candidates.append(
                _CandidateSection(
                    section_id=section_id,
                    score=mentions,
                    title=str(meta.get("section_title") or meta.get("section_path") or "").strip() or None,
                    path=str(meta.get("section_path") or "").strip() or None,
                    summary=summary,
                    page_start=_coerce_int(meta.get("page_start")),
                    page_end=_coerce_int(meta.get("page_end")),
                    source="graph_entities",
                    graph_score=mentions,
                    graph_entities=matched_entities,
                )
            )

        # Light bridge (online/deterministic):
        # Expand query terms via alias/canonicalization and embedding-nearest neighbors (entity FAISS),
        # then project expanded entities back to file-scoped sections. This is a navigation-only signal.
        bridge_diag: Dict[str, Any] = {}
        if bool(getattr(tool_defaults, "SECTION_SELECT_GRAPH_BRIDGE_ENABLED", True)):
            bridge_candidates, bridge_diag = await self._graph_bridge_candidates(
                request=request,
                file_id=file_id,
                section_map=section_map,
                raw_terms=list(terms),
                limit_sections=limit_sections,
            )
            candidates.extend(bridge_candidates)
        diagnostics["bridge"] = bridge_diag

        candidates.sort(key=lambda c: c.score, reverse=True)
        diagnostics["candidate_count"] = len(candidates)
        if candidates:
            return candidates[:limit_sections], diagnostics

        # Fallback: use adapter retrieval (embedding/PPR-informed) scoped to the file to harvest chunk signals
        # when entity extraction/matching fails. This keeps the design aligned with PageIndex: chunks guide
        # *where to read*, but are still navigation-only.
        if not bool(getattr(tool_defaults, "SECTION_SELECT_GRAPH_RETRIEVAL_FALLBACK_ENABLED", True)):
            diagnostics["fallback"] = {"enabled": False, "reason": "disabled"}
            return [], diagnostics

        if not hasattr(adapter, "aquery_subgraph"):
            diagnostics["fallback"] = {"enabled": True, "reason": "adapter_missing_aquery_subgraph"}
            return [], diagnostics

        top_k_chunks = int(getattr(tool_defaults, "SECTION_SELECT_GRAPH_RETRIEVAL_FALLBACK_TOP_K_CHUNKS", 12))
        top_k_chunks = max(1, min(top_k_chunks, 50))
        file_scope = FileScope(file_ids=frozenset({file_id}), filename_contains=(), source="section_select_graph_fallback")

        try:
            async with adapter_locked(adapter):
                payload = await adapter.aquery_subgraph(
                    query,
                    channel="graph",
                    access_scope=request.access_scope,
                    query_options={"top_k": top_k_chunks, "file_scope": file_scope.as_dict(), "export_subgraph": False},
                )
        except Exception as exc:  # noqa: BLE001
            diagnostics["fallback"] = {"enabled": True, "reason": "adapter_query_failed", "error": str(exc)}
            return [], diagnostics

        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(chunks, list) or not chunks:
            diagnostics["fallback"] = {"enabled": True, "reason": "empty_chunks"}
            return [], diagnostics

        hits: Dict[str, Dict[str, Any]] = {}
        for item in chunks:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            section_id = str(meta.get("section_id") or "").strip()
            if not section_id:
                chunk_meta = meta.get("chunk_metadata")
                if isinstance(chunk_meta, dict):
                    section_id = str(chunk_meta.get("section_id") or "").strip()
            if not section_id or section_id not in section_map:
                continue
            try:
                score = float(meta.get("score") or 0.0)
            except Exception:
                score = 0.0
            bucket = hits.setdefault(section_id, {"scores": [], "hit_count": 0})
            bucket["scores"].append(score)
            bucket["hit_count"] += 1

        if not hits:
            diagnostics["fallback"] = {"enabled": True, "reason": "no_section_hits"}
            return [], diagnostics

        chunk_counts = await self._fetch_section_chunk_counts(
            request=request,
            file_id=file_id,
            section_ids=list(hits.keys()),
        )
        fallback_candidates: List[_CandidateSection] = []
        for section_id, bucket in hits.items():
            scores = list(bucket.get("scores") or [])
            if not scores:
                continue
            chunk_total = chunk_counts.get(section_id)
            if not isinstance(chunk_total, int) or chunk_total < 0:
                chunk_total = len(scores)
            node_score = sum(scores) / math.sqrt(chunk_total + 1)
            meta = section_map.get(section_id) or {}
            fallback_candidates.append(
                _CandidateSection(
                    section_id=section_id,
                    score=node_score,
                    title=str(meta.get("section_title") or meta.get("section_path") or "").strip() or None,
                    path=str(meta.get("section_path") or "").strip() or None,
                    summary="graph_retrieval: chunk-aggregated section score",
                    page_start=_coerce_int(meta.get("page_start")),
                    page_end=_coerce_int(meta.get("page_end")),
                    source="graph_retrieval",
                    graph_score=node_score,
                    hit_count=int(bucket.get("hit_count") or 0),
                    chunk_count=int(chunk_total),
                )
            )
        fallback_candidates.sort(key=lambda c: c.score, reverse=True)
        diagnostics["fallback"] = {
            "enabled": True,
            "reason": "ok",
            "top_k_chunks": top_k_chunks,
            "section_hits": len(fallback_candidates),
        }
        return fallback_candidates[:limit_sections], diagnostics

    async def _graph_bridge_candidates(
        self,
        *,
        request: ToolRunRequest,
        file_id: str,
        section_map: Dict[str, Dict[str, Any]],
        raw_terms: Sequence[str],
        limit_sections: int,
    ) -> tuple[List[_CandidateSection], Dict[str, Any]]:
        adapter = request.adapter
        if adapter is None or not adapter_supports_cypher(adapter):
            return [], {"enabled": True, "reason": "cypher_unavailable"}
        access_scope = request.access_scope
        if access_scope is None or not getattr(access_scope, "scope_id", None):
            return [], {"enabled": True, "reason": "owner_scope_missing"}

        max_terms = int(getattr(tool_defaults, "SECTION_SELECT_GRAPH_BRIDGE_MAX_TERMS", 4))
        max_terms = max(0, min(max_terms, 32))
        term_list = [str(t or "").strip() for t in raw_terms if str(t or "").strip()]
        if max_terms > 0:
            term_list = term_list[:max_terms]
        if not term_list:
            return [], {"enabled": True, "reason": "no_terms"}

        per_term = int(getattr(tool_defaults, "SECTION_SELECT_GRAPH_BRIDGE_CANDIDATES_PER_TERM", 6))
        per_term = max(1, min(per_term, 50))
        max_entity_ids = int(getattr(tool_defaults, "SECTION_SELECT_GRAPH_BRIDGE_MAX_ENTITY_IDS", 32))
        max_entity_ids = max(1, min(max_entity_ids, 200))

        resolver = build_default_entity_resolver(candidate_limit=max(12, per_term))

        expanded: Dict[str, Dict[str, Any]] = {}
        diag_terms: List[Dict[str, Any]] = []
        for term in term_list:
            try:
                res = await resolver.resolve(adapter=adapter, access_scope=access_scope, raw_entity=term, entity_type_hint="")
            except Exception as exc:  # noqa: BLE001
                diag_terms.append({"term": term, "error": str(exc)})
                continue

            cands = list(res.candidates or [])
            # Prefer resolved candidate if available, otherwise take top-N candidates.
            chosen = []
            if res.resolved_candidate is not None:
                chosen.append(res.resolved_candidate)
            for cand in cands:
                if len(chosen) >= per_term:
                    break
                if res.resolved_candidate is not None and cand.entity_id == res.resolved_candidate.entity_id:
                    continue
                chosen.append(cand)

            diag_terms.append(
                {
                    "term": term,
                    "normalized": res.normalized,
                    "resolved": bool(res.resolved),
                    "chosen": [
                        {
                            "entity_id": c.entity_id,
                            "entity_name": c.entity_name,
                            "strategy": c.strategy,
                            "score": float(c.score),
                        }
                        for c in chosen
                    ],
                }
            )
            for c in chosen:
                if len(expanded) >= max_entity_ids:
                    break
                expanded.setdefault(
                    c.entity_id,
                    {
                        "entity_id": c.entity_id,
                        "entity_name": c.entity_name,
                        "strategies": set(),
                        "max_score": 0.0,
                    },
                )
                expanded[c.entity_id]["strategies"].add(str(c.strategy))
                expanded[c.entity_id]["max_score"] = max(float(expanded[c.entity_id]["max_score"]), float(c.score))
            if len(expanded) >= max_entity_ids:
                break

        entity_ids = list(expanded.keys())
        if not entity_ids:
            return [], {"enabled": True, "reason": "no_expanded_entities", "terms": diag_terms}

        # Project expanded entities -> file-scoped sections via mentions.
        cypher = (
            "UNWIND $entity_ids AS eid\n"
            "MATCH (e:Entity {entity_id: eid})\n"
            "WHERE COALESCE(e.owner_id, $global_owner) = $owner_id\n"
            "MATCH (e)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(s:Section)\n"
            "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
            "RETURN s.section_id AS section_id,\n"
            "       count(DISTINCT c.chunk_id) AS mentions,\n"
            "       collect(DISTINCT e.entity_name)[..$entity_name_limit] AS matched_entities\n"
            "ORDER BY mentions DESC\n"
            "LIMIT $limit\n"
        )
        params = {
            "entity_ids": entity_ids,
            "file_id": file_id,
            "limit": int(max(1, int(limit_sections))),
            "entity_name_limit": max(1, int(getattr(tool_defaults, "SECTION_SELECT_ENTITY_HINTS_MAX", 6))),
        }
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=access_scope)

        out: List[_CandidateSection] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            if not section_id or section_id not in section_map:
                continue
            try:
                mentions = float(row.get("mentions") or 0.0)
            except Exception:
                mentions = 0.0
            matched = row.get("matched_entities") if isinstance(row.get("matched_entities"), list) else []
            matched_entities = [str(item).strip() for item in matched if str(item or "").strip()]
            meta = section_map.get(section_id) or {}
            summary = None
            if matched_entities:
                preview = matched_entities[: max(1, int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX))]
                summary = "graph_bridge_entities: " + ", ".join(preview)
            out.append(
                _CandidateSection(
                    section_id=section_id,
                    score=mentions,
                    title=str(meta.get("section_title") or meta.get("section_path") or "").strip() or None,
                    path=str(meta.get("section_path") or "").strip() or None,
                    summary=summary,
                    page_start=_coerce_int(meta.get("page_start")),
                    page_end=_coerce_int(meta.get("page_end")),
                    source="graph_bridge",
                    graph_score=mentions,
                    graph_entities=matched_entities or None,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out[: max(1, int(limit_sections))], {
            "enabled": True,
            "reason": "ok" if out else "empty_projection",
            "terms": diag_terms,
            "expanded_entity_ids": len(entity_ids),
            "projected_sections": len(out),
        }

    @staticmethod
    def _merge_candidates(
        index_candidates: Sequence[_CandidateSection],
        value_candidates: Sequence[_CandidateSection],
        graph_candidates: Sequence[_CandidateSection] | None = None,
    ) -> List[_CandidateSection]:
        merged: Dict[str, _CandidateSection] = {}
        for cand in index_candidates or []:
            merged[cand.section_id] = cand
        for cand in value_candidates or []:
            existing = merged.get(cand.section_id)
            if existing is None:
                merged[cand.section_id] = cand
                continue
            index_score = existing.index_score if existing.index_score is not None else None
            node_score = cand.node_score if cand.node_score is not None else existing.node_score
            score = max([val for val in [index_score, node_score] if isinstance(val, (int, float))] or [existing.score])
            source = "+".join([s for s in [existing.source, cand.source] if s]) or None
            merged[cand.section_id] = _CandidateSection(
                section_id=cand.section_id,
                score=float(score),
                title=existing.title or cand.title,
                path=existing.path or cand.path,
                summary=existing.summary or cand.summary,
                page_start=existing.page_start if existing.page_start is not None else cand.page_start,
                page_end=existing.page_end if existing.page_end is not None else cand.page_end,
                source=source,
                index_score=index_score,
                node_score=node_score,
                graph_score=existing.graph_score,
                graph_entities=existing.graph_entities,
                hit_count=cand.hit_count or existing.hit_count,
                chunk_count=cand.chunk_count or existing.chunk_count,
            )
        for cand in graph_candidates or []:
            existing = merged.get(cand.section_id)
            if existing is None:
                merged[cand.section_id] = cand
                continue
            scores = [
                existing.index_score,
                existing.node_score,
                existing.graph_score,
                cand.graph_score,
                existing.score,
                cand.score,
            ]
            score = max([val for val in scores if isinstance(val, (int, float))] or [existing.score])
            source = "+".join([s for s in [existing.source, cand.source] if s]) or None
            graph_entities = []
            if existing.graph_entities:
                graph_entities.extend(existing.graph_entities)
            if cand.graph_entities:
                graph_entities.extend(cand.graph_entities)
            merged[cand.section_id] = _CandidateSection(
                section_id=cand.section_id,
                score=float(score),
                title=existing.title or cand.title,
                path=existing.path or cand.path,
                summary=existing.summary or cand.summary,
                page_start=existing.page_start if existing.page_start is not None else cand.page_start,
                page_end=existing.page_end if existing.page_end is not None else cand.page_end,
                source=source,
                index_score=existing.index_score,
                node_score=existing.node_score,
                graph_score=cand.graph_score or existing.graph_score,
                graph_entities=_dedupe_keep_order(graph_entities),
                hit_count=existing.hit_count or cand.hit_count,
                chunk_count=existing.chunk_count or cand.chunk_count,
            )
        return sorted(merged.values(), key=lambda c: c.score, reverse=True)

    @staticmethod
    def _enrich_tree(tree: Dict[str, Any], section_map: Dict[str, Dict[str, Any]], entity_hints: Dict[str, List[str]]) -> Dict[str, Any]:
        """Enrich and compact a section tree payload for LLM routing.

        Goals:
        - Preserve only fields used for routing (reduce prompt size + artifact IO).
        - Attach summary/entity_hints from section_map when available.
        """

        def _compact_node(node: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            section_id = str(node.get("section_id") or "").strip() or None
            title = str(node.get("title") or "").strip() or None
            path = node.get("path")
            if not isinstance(path, str) or not path.strip():
                path = node.get("section_path")
            path = str(path or "").strip() or None
            summary = str(node.get("summary") or "").strip() or None
            page_start = node.get("page_start")
            page_end = node.get("page_end")
            node_types = node.get("node_types") if isinstance(node.get("node_types"), dict) else None
            signals = node.get("signals") if isinstance(node.get("signals"), dict) else None
            queue_meta = node.get("queue_meta") if isinstance(node.get("queue_meta"), dict) else None
            entity_hint_list = node.get("entity_hints") if isinstance(node.get("entity_hints"), list) else None

            if section_id is not None:
                out["section_id"] = section_id
            if title is not None:
                out["title"] = title
            if path is not None:
                out["path"] = path
            if summary is not None:
                out["summary"] = summary
            if isinstance(page_start, int):
                out["page_start"] = page_start
            if isinstance(page_end, int):
                out["page_end"] = page_end
            if node_types:
                compact_types = {str(k): int(v) for k, v in node_types.items() if isinstance(v, int) and v > 0}
                if compact_types:
                    out["node_types"] = compact_types
            if signals:
                out["signals"] = dict(signals)
            if queue_meta:
                out["queue_meta"] = dict(queue_meta)
            if entity_hint_list:
                out["entity_hints"] = [str(x).strip() for x in entity_hint_list if str(x or "").strip()]
            children = node.get("children") if isinstance(node.get("children"), list) else []
            out_children: list[dict[str, Any]] = []
            for child in children:
                if isinstance(child, dict):
                    out_children.append(_compact_node(child))
            if out_children:
                out["children"] = out_children
            return out

        def _walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            sid = str(node.get("section_id") or "").strip()
            if sid and sid in section_map:
                meta = section_map.get(sid, {})
                if meta.get("summary"):
                    node["summary"] = meta.get("summary")
                if sid in entity_hints:
                    node["entity_hints"] = entity_hints[sid]
                if isinstance(meta.get("section_path"), str) and meta.get("section_path"):
                    node.setdefault("path", meta.get("section_path"))
            children = node.get("children") if isinstance(node.get("children"), list) else []
            for child in children:
                _walk(child)

        _walk(tree)
        # Compact at the end to reduce prompt/diagnostic payload size.
        return _compact_node(tree)

    @staticmethod
    def _parse_tree_search_output(payload: Any, section_map: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {"raw_type": type(payload).__name__ if payload is not None else "none"}
        if not isinstance(payload, dict):
            diagnostics["reason"] = "invalid_payload"
            return [], diagnostics
        raw_list = payload.get("node_list") or payload.get("nodes") or payload.get("section_ids") or []
        if isinstance(raw_list, (str, int)):
            raw_list = [raw_list]
        if not isinstance(raw_list, (list, tuple, set)):
            diagnostics["reason"] = "invalid_node_list"
            return [], diagnostics
        out: List[str] = []
        for item in raw_list:
            token = str(item or "").strip()
            if not token or token not in section_map:
                continue
            if token in out:
                continue
            out.append(token)
        diagnostics["node_count"] = len(out)
        return out, diagnostics

    async def _run_llm_tree_search(
        self,
        *,
        question: str,
        file_id: str,
        tree_payload: Dict[str, Any],
        candidates_payload: List[Dict[str, Any]],
        section_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        max_tokens = int(getattr(tool_defaults, "SECTION_SELECT_TREE_SEARCH_MAX_TOKENS", 0) or 0)
        if max_tokens <= 0:
            max_tokens = None
        prompt = SECTION_TREE_SEARCH_USER_PROMPT.format(
            question=question,
            file_id=file_id,
            tree_json=json.dumps(tree_payload, ensure_ascii=False),
            candidates_json=json.dumps(candidates_payload, ensure_ascii=False),
        )
        messages = [
            {"role": "system", "content": SECTION_TREE_SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        parsed = await call_llm_json_with_retry(
            llm_connector=self.llm_connector,
            messages=messages,
            expected="dict",
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        node_list, parse_diag = self._parse_tree_search_output(parsed, section_map)
        diagnostics = {"parse": parse_diag}
        return node_list, diagnostics

    @staticmethod
    def _build_queue(
        *,
        llm_nodes: Sequence[str],
        index_candidates: Sequence[_CandidateSection],
        value_candidates: Sequence[_CandidateSection],
        graph_candidates: Sequence[_CandidateSection],
        section_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        section_ids = set(section_map.keys())
        index_ids = [c.section_id for c in index_candidates if c.section_id in section_ids]
        value_ids = [c.section_id for c in value_candidates if c.section_id in section_ids]
        graph_ids = [c.section_id for c in graph_candidates if c.section_id in section_ids]
        llm_ids = [sid for sid in llm_nodes if sid in section_ids]

        meta: Dict[str, Dict[str, Any]] = {}
        for cand in index_candidates:
            meta.setdefault(cand.section_id, {})["index_score"] = cand.index_score
        for cand in value_candidates:
            meta.setdefault(cand.section_id, {})["node_score"] = cand.node_score
        for cand in graph_candidates:
            meta.setdefault(cand.section_id, {})["graph_score"] = cand.graph_score

        sources = [
            ("llm_tree", llm_ids),
            ("graph_entities", graph_ids),
            ("section_index", index_ids),
            ("value_node_score", value_ids),
        ]
        pointers = {name: 0 for name, _ in sources}
        seen: set[str] = set()
        queue: List[str] = []
        queue_items: List[Dict[str, Any]] = []

        while True:
            progressed = False
            for name, ids in sources:
                idx = pointers[name]
                if idx >= len(ids):
                    continue
                progressed = True
                sid = ids[idx]
                pointers[name] = idx + 1
                if sid in seen:
                    continue
                seen.add(sid)
                item = {"section_id": sid, "source": name}
                item.update(meta.get(sid, {}))
                queue.append(sid)
                queue_items.append(item)
            if not progressed:
                break

        return queue, queue_items

    def _build_batch_payload(
        self,
        *,
        batch_ids: Sequence[str],
        section_map: Dict[str, Dict[str, Any]],
        entity_hints: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for sid in batch_ids:
            meta = section_map.get(sid)
            if not isinstance(meta, dict):
                continue
            node = {
                "section_id": sid,
                "title": meta.get("section_title"),
                "path": meta.get("section_path"),
                "summary": meta.get("summary") or "",
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "node_types": meta.get("node_types") or {},
            }
            # Optional navigation-only hints (for selection only; NOT citeable evidence):
            # - signals: multi-channel relevance hints (section_index/value_node_score/graph_entities)
            # - queue_meta: how this node entered the current queue (source + scores)
            signals = meta.get("signals")
            if isinstance(signals, dict) and signals:
                node["signals"] = signals
            queue_meta = meta.get("queue_meta")
            if isinstance(queue_meta, dict) and queue_meta:
                node["queue_meta"] = queue_meta
            if sid in entity_hints:
                node["entity_hints"] = entity_hints[sid]
            payload.append(node)
        return payload

    @staticmethod
    def _attach_candidate_signals(section_map: Dict[str, Dict[str, Any]], candidates: Sequence[_CandidateSection]) -> None:
        for cand in candidates or []:
            meta = section_map.get(cand.section_id)
            if not isinstance(meta, dict):
                continue
            signals: Dict[str, Any] = dict(meta.get("signals") or {}) if isinstance(meta.get("signals"), dict) else {}
            if cand.index_score is not None:
                signals["index_score"] = cand.index_score
            if cand.node_score is not None:
                signals["node_score"] = cand.node_score
            if cand.graph_score is not None:
                signals["graph_score"] = cand.graph_score
            if cand.source:
                signals["source"] = cand.source
            if cand.hit_count is not None:
                signals["hit_count"] = cand.hit_count
            if cand.chunk_count is not None:
                signals["chunk_count"] = cand.chunk_count
            if cand.graph_entities:
                signals["graph_entities"] = cand.graph_entities
            if signals:
                meta["signals"] = signals

    @staticmethod
    def _attach_queue_meta(section_map: Dict[str, Dict[str, Any]], queue_items: Sequence[Dict[str, Any]]) -> None:
        for item in queue_items or []:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("section_id") or "").strip()
            if not sid:
                continue
            meta = section_map.get(sid)
            if not isinstance(meta, dict):
                continue
            queue_meta: Dict[str, Any] = {}
            source = str(item.get("source") or "").strip()
            if source:
                queue_meta["source"] = source
            for key in ("index_score", "node_score", "graph_score"):
                val = item.get(key)
                if isinstance(val, (int, float)):
                    queue_meta[key] = float(val)
            if queue_meta:
                meta["queue_meta"] = queue_meta

    @staticmethod
    def _parse_consumer_output(payload: Any, section_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        selection, parse_diag = SectionSelectTool._parse_selection(payload, section_map)
        enough_info = False
        if isinstance(payload, dict):
            enough_info = bool(payload.get("enough_info") or payload.get("enough"))
        parse_diag["enough_info"] = enough_info
        return selection, parse_diag, enough_info

    async def _run_consumer_llm(
        self,
        *,
        question: str,
        batch_payload: List[Dict[str, Any]],
        selected_primary: List[str],
        selected_supplementary: List[str],
        section_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        max_tokens = int(getattr(tool_defaults, "SECTION_SELECT_CONSUMER_MAX_TOKENS", 0) or 0)
        if max_tokens <= 0:
            max_tokens = None
        prompt = SECTION_SELECT_CONSUMER_USER_PROMPT.format(
            question=question,
            primary_ids=json.dumps(selected_primary, ensure_ascii=False),
            supplementary_ids=json.dumps(selected_supplementary, ensure_ascii=False),
            batch_json=json.dumps(batch_payload, ensure_ascii=False),
        )
        messages = [
            {"role": "system", "content": SECTION_SELECT_CONSUMER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        parsed = await call_llm_json_with_retry(
            llm_connector=self.llm_connector,
            messages=messages,
            expected="dict",
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return self._parse_consumer_output(parsed, section_map)

    async def _consume_queue(
        self,
        *,
        question: str,
        queue_ids: Sequence[str],
        section_map: Dict[str, Dict[str, Any]],
        entity_hints: Dict[str, List[str]],
        max_rounds: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        max_rounds = max(1, int(max_rounds))
        batch_size = max(1, int(batch_size))
        queue = [sid for sid in queue_ids if sid in section_map]
        selected_primary: List[str] = []
        selected_supplementary: List[str] = []
        rounds: List[Dict[str, Any]] = []
        parse_diagnostics: List[Dict[str, Any]] = []
        enough_info = False
        offset = 0

        for round_idx in range(max_rounds):
            batch_ids: List[str] = []
            while offset < len(queue) and len(batch_ids) < batch_size:
                sid = queue[offset]
                offset += 1
                if sid in batch_ids:
                    continue
                batch_ids.append(sid)
            if not batch_ids:
                break
            batch_payload = self._build_batch_payload(
                batch_ids=batch_ids,
                section_map=section_map,
                entity_hints=entity_hints,
            )
            selection, diag, enough_info = await self._run_consumer_llm(
                question=question,
                batch_payload=batch_payload,
                selected_primary=selected_primary,
                selected_supplementary=selected_supplementary,
                section_map=section_map,
            )
            parse_diagnostics.append(diag)
            for sid in selection.get("primary_section_ids", []):
                if sid not in selected_primary:
                    selected_primary.append(sid)
            for sid in selection.get("supplementary_section_ids", []):
                if sid not in selected_supplementary:
                    selected_supplementary.append(sid)
            rounds.append(
                {
                    "round": round_idx + 1,
                    "batch_section_ids": batch_ids,
                    "primary_section_ids": selection.get("primary_section_ids", []),
                    "supplementary_section_ids": selection.get("supplementary_section_ids", []),
                    "enough_info": enough_info,
                    "explanation": selection.get("explanation") or "",
                    "queue_remaining": max(0, len(queue) - offset),
                }
            )
            if enough_info:
                break

        if not selected_primary and queue:
            selected_primary = [queue[0]]
            parse_diagnostics.append({"fallback": "top_queue"})

        selection = {
            "primary_section_ids": selected_primary,
            "supplementary_section_ids": selected_supplementary,
            "explanation": rounds[-1]["explanation"] if rounds else "",
        }
        return {
            "selection": selection,
            "rounds": rounds,
            "enough_info": enough_info,
            "parse_diagnostics": parse_diagnostics,
        }

    async def _collect_entity_hints(
        self,
        *,
        request: ToolRunRequest,
        file_id: str,
        section_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        max_hints = int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX)
        max_hints = max(0, max_hints)
        if not section_ids or max_hints <= 0:
            return {}
        adapter = request.adapter
        if adapter is None or not adapter_supports_cypher(adapter):
            return {}
        cypher = (
            "UNWIND $section_ids AS sid\n"
            "MATCH (s:Section {section_id: sid})\n"
            "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
            "MATCH (s)-[:HAS_CHUNK]->(c:Chunk)\n"
            "WHERE COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "MATCH (c)-[:MENTIONS]->(e:Entity)\n"
            "RETURN sid AS section_id, e.entity_name AS entity, count(*) AS mentions\n"
            "ORDER BY section_id, mentions DESC\n"
        )
        params = {"section_ids": list(section_ids), "file_id": file_id}
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)
        hints: Dict[str, List[str]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            entity = str(row.get("entity") or "").strip()
            if not section_id or not entity:
                continue
            bucket = hints.setdefault(section_id, [])
            if entity in bucket:
                continue
            bucket.append(entity)
        for key, values in list(hints.items()):
            hints[key] = values[:max_hints]
        return hints

    async def _fetch_section_chunk_counts(
        self,
        *,
        request: ToolRunRequest,
        file_id: str,
        section_ids: Sequence[str],
    ) -> Dict[str, int]:
        if not section_ids:
            return {}
        adapter = request.adapter
        if adapter is None or not adapter_supports_cypher(adapter):
            return {}
        section_ids = [str(sid).strip() for sid in section_ids if str(sid or "").strip()]
        if not section_ids:
            return {}
        cypher = (
            "UNWIND $section_ids AS sid\n"
            "MATCH (s:Section {section_id: sid})-[:HAS_CHUNK]->(c:Chunk)\n"
            "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
            "RETURN sid AS section_id, count(c) AS chunk_count\n"
        )
        params = {"section_ids": list(section_ids), "file_id": file_id}
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)
        counts: Dict[str, int] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("section_id") or "").strip()
            if not sid:
                continue
            try:
                counts[sid] = int(row.get("chunk_count") or 0)
            except Exception:
                counts[sid] = int(counts.get(sid) or 0)
        return counts

    async def _collect_subtree_entity_hints(
        self,
        *,
        request: ToolRunRequest,
        file_id: str,
        section_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        max_hints = int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX)
        max_hints = max(0, max_hints)
        if not section_ids or max_hints <= 0:
            return {}
        adapter = request.adapter
        if adapter is None or not adapter_supports_cypher(adapter):
            return {}
        cypher = (
            "UNWIND $section_ids AS sid\n"
            "MATCH (s:Section {section_id: sid})-[:HAS_CHILD]->(t:TreeNode)\n"
            "WHERE s.source_file_id = $file_id AND COALESCE(s.owner_id, $global_owner) = $owner_id\n"
            "MATCH (t)-[:HAS_CHUNK]->(c:Chunk)\n"
            "WHERE COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "MATCH (c)-[:MENTIONS]->(e:Entity)\n"
            "RETURN sid AS section_id, e.entity_name AS entity, count(*) AS mentions\n"
            "ORDER BY section_id, mentions DESC\n"
        )
        params = {"section_ids": list(section_ids), "file_id": file_id}
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)
        hints: Dict[str, List[str]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            section_id = str(row.get("section_id") or "").strip()
            entity = str(row.get("entity") or "").strip()
            if not section_id or not entity:
                continue
            bucket = hints.setdefault(section_id, [])
            if entity in bucket:
                continue
            bucket.append(entity)
        for key, values in list(hints.items()):
            hints[key] = values[:max_hints]
        return hints

    def _build_catalog_payload(
        self,
        section_map: Dict[str, Dict[str, Any]],
        candidates: Sequence[_CandidateSection],
        entity_hints: Dict[str, List[str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        max_sections = max(1, int(tool_defaults.SECTION_SELECT_MAX_SECTIONS))
        ordered_sections = list(section_map.values())
        ordered_sections.sort(
            key=lambda s: (
                s.get("page_start") if isinstance(s.get("page_start"), int) else 1_000_000,
                str(s.get("section_path") or ""),
            )
        )
        catalog = ordered_sections[:max_sections]
        for item in catalog:
            sid = str(item.get("section_id") or "")
            if sid and sid in entity_hints:
                item["entity_hints"] = entity_hints[sid]

        candidate_payload: List[Dict[str, Any]] = []
        for cand in candidates:
            payload = {
                "section_id": cand.section_id,
                "score": cand.score,
                "source": cand.source,
                "index_score": cand.index_score,
                "node_score": cand.node_score,
                "graph_score": cand.graph_score,
                "hit_count": cand.hit_count,
                "chunk_count": cand.chunk_count,
                "title": cand.title,
                "path": cand.path,
                "summary": cand.summary,
                "page_start": cand.page_start,
                "page_end": cand.page_end,
            }
            if cand.graph_entities:
                payload["graph_entities"] = cand.graph_entities
            if cand.section_id in entity_hints:
                payload["entity_hints"] = entity_hints[cand.section_id]
            node_types = section_map.get(cand.section_id, {}).get("node_types")
            if isinstance(node_types, dict) and node_types:
                payload["node_types"] = node_types
            candidate_payload.append(payload)
        return {"catalog": catalog, "candidates": candidate_payload}

    @staticmethod
    def _parse_selection(payload: Any, section_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {"raw_type": type(payload).__name__ if payload is not None else "none"}
        out = {"primary_section_ids": [], "supplementary_section_ids": [], "explanation": ""}
        if not isinstance(payload, dict):
            diagnostics["reason"] = "invalid_payload"
            return out, diagnostics

        def _filter_ids(raw: Any) -> List[str]:
            items: List[str] = []
            if isinstance(raw, (list, tuple, set)):
                raw_items = raw
            elif raw:
                raw_items = [raw]
            else:
                raw_items = []
            for item in raw_items:
                token = str(item or "").strip()
                if token and token in section_map:
                    items.append(token)
            return _dedupe_keep_order(items)

        out["primary_section_ids"] = _filter_ids(payload.get("primary_section_ids") or payload.get("primary"))
        out["supplementary_section_ids"] = _filter_ids(payload.get("supplementary_section_ids") or payload.get("supplementary"))
        explanation = payload.get("explanation") or payload.get("reasoning") or ""
        out["explanation"] = str(explanation).strip()
        diagnostics["validated_primary"] = len(out["primary_section_ids"])
        diagnostics["validated_supplementary"] = len(out["supplementary_section_ids"])
        return out, diagnostics

    @staticmethod
    def _expand_subtree(tree: Dict[str, Any], selected_ids: Sequence[str], *, max_depth: int) -> List[str]:
        if not selected_ids or not isinstance(tree, dict):
            return []

        selected = set(selected_ids)
        out: List[str] = []

        def _walk(node: Dict[str, Any], active: bool, depth_from_selected: int) -> None:
            if not isinstance(node, dict):
                return
            sid = str(node.get("section_id") or "").strip()
            now_active = active or (sid in selected if sid else False)
            if now_active:
                if depth_from_selected <= max_depth and sid and sid not in out:
                    out.append(sid)
                if depth_from_selected >= max_depth:
                    return
            children = node.get("children") if isinstance(node.get("children"), list) else []
            for child in children:
                if isinstance(child, dict):
                    next_depth = depth_from_selected + 1 if now_active else 0
                    _walk(child, now_active, next_depth)

        _walk(tree, False, 0)
        return out

    @staticmethod
    def _build_summary(section_map: Dict[str, Dict[str, Any]], payload: Dict[str, Any]) -> str:
        primary = payload.get("primary_section_ids") or []
        supplementary = payload.get("supplementary_section_ids") or []
        subtree = payload.get("subtree_section_ids") or []
        entity_hints = payload.get("section_entity_hints") or {}
        node_types_map = payload.get("section_node_types") or {}
        seed_entities = payload.get("seed_entities") or []

        def _render(ids: Sequence[str]) -> List[str]:
            lines: List[str] = []
            for sid in ids:
                meta = section_map.get(sid) or {}
                path = meta.get("section_path") or meta.get("section_title") or sid
                page_start = meta.get("page_start")
                page_end = meta.get("page_end")
                page_hint = ""
                if isinstance(page_start, int) or isinstance(page_end, int):
                    page_hint = f" pages={page_start}-{page_end}"
                node_types = node_types_map.get(sid) or {}
                type_tags = []
                for key in ("table", "image"):
                    count = node_types.get(key)
                    if isinstance(count, int) and count > 0:
                        type_tags.append(f"{key}={count}")
                type_hint = f" [{', '.join(type_tags)}]" if type_tags else ""
                lines.append(f"- {sid}: {path}{page_hint}{type_hint}")
                hints = entity_hints.get(sid)
                if isinstance(hints, list) and hints:
                    max_hints = int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX)
                    preview = [str(h) for h in hints[:max_hints] if str(h or "").strip()]
                    if preview:
                        lines.append(f"  entities: {', '.join(preview)}")
            return lines

        lines = ["section.select result:"]
        if primary:
            lines.append("primary:")
            lines.extend(_render(primary))
        if supplementary:
            lines.append("supplementary:")
            lines.extend(_render(supplementary))
        if subtree:
            preview_limit = max(1, int(tool_defaults.SECTION_SELECT_SUBTREE_PREVIEW_MAX))
            preview = subtree[:preview_limit]
            lines.append(f"subtree_section_ids({len(subtree)}): {preview}")
            if len(subtree) > preview_limit:
                lines.append("subtree_section_ids: (truncated)")
        if payload.get("queue_ids") is not None:
            queue_ids = payload.get("queue_ids") or []
            lines.append(f"queue_size: {len(queue_ids)}")
        if payload.get("consumer_rounds") is not None:
            rounds = payload.get("consumer_rounds") or []
            lines.append(f"consumer_rounds: {len(rounds)}")
        if payload.get("enough_info") is not None:
            lines.append(f"enough_info: {payload.get('enough_info')}")
        if payload.get("explanation"):
            lines.append(f"explanation: {payload.get('explanation')}")
        if seed_entities:
            preview = [str(s) for s in seed_entities[: max(1, int(tool_defaults.SECTION_SELECT_ENTITY_HINTS_MAX))] if str(s or "").strip()]
            if preview:
                lines.append(f"seed_entities: {', '.join(preview)}")
        suggested_reads = payload.get("suggested_reads") or []
        if isinstance(suggested_reads, list) and suggested_reads:
            lines.append("next_steps:")
            for item in suggested_reads[: max(1, int(tool_defaults.SEARCH_SUGGESTED_READ_MAX))]:
                if not isinstance(item, dict):
                    continue
                args = item.get("args") if isinstance(item.get("args"), dict) else {}
                ps = args.get("page_start")
                pe = args.get("page_end")
                lines.append(f"- read.pages pages={ps}-{pe} (citeable)")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_evidence(request: ToolRunRequest, payload: Dict[str, Any], summary: str):
        return EvidenceChunk(
            chunk_id=derived_chunk_id(tool_name="section.select", plan_step=request.plan_step, content=summary, label="selection"),
            source="section.select",
            content=summary,
            kind=EVIDENCE_KIND_DERIVED,
            score=None,
            provenance={"selection": payload},
        )


__all__ = ["SectionSelectTool"]
