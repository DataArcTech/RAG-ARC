"""Graph chunk search channel and tool."""
import asyncio
import contextlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_GRAPH_CHUNK
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.entity_resolution import build_default_entity_resolver
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.graph_adapter.base import GraphDeepSearchAdapter
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher
from core.prompts.deepsearch import SEARCH_ENTITY_EXTRACT_PROMPT_EN
from core.utils.owner_guard import normalize_owner_id
from framework.thread_pool import get_thread_pool

from ...base import (
    GraphTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
    build_input_schema,
    call_llm_async,
    extract_json_from_text,
    safe_json_loads,
)
from ...governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from core.deepsearch.utils.llm_json import call_llm_json_with_retry
from .base import _ChannelResult, _SearchToolBase, resolve_explicit_file_scope, strip_file_scope_from_graph_context


class _GraphChunkChannel:
    def _graph_overrides(self: _SearchToolBase, extra: Mapping[str, Any]) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        allowlist = set(tool_defaults.SEARCH_GRAPH_SAFE_OVERRIDE_KEYS)
        if isinstance(extra.get("graph_overrides"), dict):
            for key, value in extra["graph_overrides"].items():
                if key in allowlist:
                    overrides[key] = value
        for key in allowlist:
            if key in extra:
                overrides[key] = extra[key]
        return overrides

    @staticmethod
    def _coerce_seed_items(items: Any) -> List[Tuple[str, str | None]]:
        out: List[Tuple[str, str | None]] = []
        if not isinstance(items, list):
            return out
        for item in items:
            if isinstance(item, str):
                token = item.strip()
                if token:
                    out.append((token, None))
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("entity_name") or "").strip()
                if name:
                    etype = str(item.get("type") or item.get("entity_type") or "").strip() or None
                    out.append((name, etype))
        return out

    async def _resolve_entity_ids(
        self: _SearchToolBase,
        *,
        request: ToolRunRequest,
        adapter: GraphDeepSearchAdapter,
        seeds: Sequence[Tuple[str, str | None]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {"resolved": 0, "candidates": 0}
        if not seeds:
            return [], diagnostics
        resolver = build_default_entity_resolver()
        resolved_ids: List[str] = []
        unresolved: List[Tuple[str, str | None]] = []
        for name, entity_type in seeds:
            diagnostics["candidates"] += 1
            if name.startswith("entity-"):
                resolved_ids.append(name)
                diagnostics["resolved"] += 1
                continue
            unresolved.append((name, entity_type))

        if not unresolved:
            return resolved_ids, diagnostics

        if not adapter_supports_cypher(adapter):
            diagnostics["reason"] = "cypher_unavailable"
            return resolved_ids, diagnostics

        for name, entity_type in unresolved:
            try:
                result = await resolver.resolve(
                    adapter=adapter,
                    access_scope=request.access_scope,
                    raw_entity=name,
                    entity_type_hint=entity_type or "",
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics.setdefault("errors", []).append(str(exc))
                continue
            if result.resolved and result.resolved_candidate is not None:
                resolved_ids.append(result.resolved_candidate.entity_id)
                diagnostics["resolved"] += 1
        return resolved_ids, diagnostics

    async def _extract_entities_with_llm(
        self: _SearchToolBase,
        *,
        query: str,
        limit: int,
    ) -> Tuple[List[Tuple[str, str | None]], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {"limit": limit}
        if limit <= 0:
            diagnostics["reason"] = "limit_disabled"
            return [], diagnostics
        if self.llm_connector is None:
            raise RuntimeError("graph_chunk entity extraction requires an LLM connector")
        messages = [
            {"role": "system", "content": SEARCH_ENTITY_EXTRACT_PROMPT_EN},
            {"role": "user", "content": f"Query: {query}"},
        ]
        kwargs: Dict[str, Any] = {
            "temperature": float(tool_defaults.SEARCH_ENTITY_EXTRACT_TEMPERATURE),
            "max_tokens": int(tool_defaults.SEARCH_ENTITY_EXTRACT_MAX_TOKENS),
        }
        low_cost = self._low_cost_model_name(self.llm_connector)
        if low_cost:
            kwargs["model"] = low_cost
            diagnostics["model"] = low_cost
        try:
            payload = await call_llm_json_with_retry(
                llm_connector=self.llm_connector,
                messages=messages,
                expected="object",
                temperature=float(kwargs["temperature"]),
                max_tokens=int(kwargs["max_tokens"]),
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics["error"] = str(exc)
            return [], diagnostics
        if not isinstance(payload, dict):
            diagnostics["error"] = "json_parse_failed"
            return [], diagnostics
        # Online term extraction: accept both {terms:[...]} (preferred) and legacy {entities:[...]} payloads.
        items = payload.get("terms")
        if not isinstance(items, list):
            items = payload.get("entities")
        if not isinstance(items, list):
            diagnostics["error"] = "missing_terms"
            return [], diagnostics

        results: List[Tuple[str, str | None]] = []
        for item in items:
            if len(results) >= int(limit):
                break
            if isinstance(item, str):
                token = item.strip()
                if token:
                    results.append((token, None))
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("name") or item.get("term") or "").strip()
                if not text:
                    continue
                etype = str(item.get("kind") or item.get("type") or "").strip() or None
                results.append((text, etype))
        diagnostics["extracted"] = len(results)
        return results, diagnostics

    def _graph_chunk_sync(
        self: _SearchToolBase,
        *,
        retriever: Any,
        query: str,
        owner_id: str,
        top_k: int,
        use_ppr: bool,
        enable_llm_rerank: bool,
        enable_entity_fallback: bool,
        entity_seed_top_k: int,
        seed_override: Optional[Sequence[str]],
        return_subgraph_info: bool,
        overrides: Mapping[str, Any],
    ) -> Tuple[List[Chunk], Dict[str, Any], bool]:
        diagnostics: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "use_ppr": use_ppr,
            "return_subgraph_info": bool(return_subgraph_info),
        }

        normalized_owner = normalize_owner_id(owner_id)
        if normalized_owner is None:
            diagnostics["reason"] = "owner_id_invalid"
            return [], diagnostics, False

        override_items = dict(overrides or {})

        @contextlib.contextmanager
        def _temp_overrides() -> Iterable[None]:
            cfg = getattr(retriever, "config", None)
            if cfg is None or not override_items:
                yield
                return
            original: Dict[str, Any] = {}
            for key, value in override_items.items():
                if hasattr(cfg, key):
                    original[key] = getattr(cfg, key)
                    setattr(cfg, key, value)
            try:
                yield
            finally:
                for key, value in original.items():
                    setattr(cfg, key, value)

        with _temp_overrides():
            build_maps = getattr(retriever, "_build_node_mappings", None)
            if callable(build_maps):
                build_maps(owner_id=normalized_owner)

            try:
                from core.utils.query_variants import generate_query_variants

                variant_queries = generate_query_variants(query, llm_connector=self.llm_connector)
            except Exception:  # noqa: BLE001
                variant_queries = [str(query or "").strip()]

            query_doc_scores = None
            dense_scores_fn = getattr(retriever, "_dense_passage_retrieval_scores", None)
            if callable(dense_scores_fn):
                for qv in variant_queries or [query]:
                    if not str(qv or "").strip():
                        continue
                    scores_v = dense_scores_fn(str(qv))
                    query_doc_scores = scores_v if query_doc_scores is None else np.maximum(query_doc_scores, scores_v)
                if query_doc_scores is None:
                    query_doc_scores = dense_scores_fn(query)
            else:
                query_doc_scores = None

            query_fact_scores, fact_ids = retriever._get_fact_scores_faiss(
                query,
                owner_id=normalized_owner,
                query_doc_scores=query_doc_scores,
            )
            diagnostics["fact_total"] = int(len(fact_ids or []))

            top_k_facts: List[Tuple] = []
            top_k_fact_indices: List[int] = []
            if query_fact_scores is not None and len(query_fact_scores) > 0:
                if enable_llm_rerank and getattr(retriever, "llm_client", None):
                    top_k_facts, top_k_fact_indices = retriever._rerank_facts(
                        query, query_fact_scores, fact_ids, owner_id=normalized_owner
                    )
                    diagnostics["fact_rerank"] = True
                else:
                    link_top_k = int(getattr(retriever.config, "fact_retrieval_top_k", 0) or 0)
                    link_top_k = max(1, link_top_k)
                    top_k_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                    top_k_facts = retriever._get_facts_by_indices(top_k_fact_indices, fact_ids, owner_id=normalized_owner)
                    diagnostics["fact_rerank"] = False

            diagnostics["fact_used"] = int(len(top_k_facts))

            seed_entity_ids: set[str] = set(seed_override or [])
            seed_source_counts = {"tool_args": len(seed_entity_ids), "facts": 0, "entity_nn": 0}

            if not seed_override and top_k_facts:
                fact_seeds = retriever._extract_entity_ids_from_facts(top_k_facts)
                seed_entity_ids.update(fact_seeds)
                seed_source_counts["facts"] = len(fact_seeds)

            entity_nn_enabled = bool(getattr(retriever.config, "seed_entities_from_entity_nn_enabled", False))
            if entity_nn_enabled and normalized_owner:
                try:
                    entity_nn_top_k = int(getattr(retriever.config, "seed_entities_from_entity_nn_top_k", 0) or 0)
                except Exception:
                    entity_nn_top_k = 0
                if entity_nn_top_k > 0:
                    try:
                        nn_candidates = retriever._seed_entities_from_entity_nn(
                            query=query, owner_id=normalized_owner, top_k=entity_nn_top_k
                        )
                    except Exception:  # noqa: BLE001
                        nn_candidates = []
                    added = 0
                    max_extra = int(getattr(retriever.config, "seed_entities_from_entity_nn_max_extra", 0) or 0)
                    for eid in nn_candidates:
                        if eid in seed_entity_ids:
                            continue
                        seed_entity_ids.add(eid)
                        added += 1
                        if max_extra > 0 and added >= max_extra:
                            break
                    if added:
                        seed_source_counts["entity_nn"] = added

            diagnostics["seed_sources"] = seed_source_counts

            if not seed_entity_ids:
                diagnostics["reason"] = "no_seed_entities"
                return [], diagnostics, bool(enable_entity_fallback)

            entity_relevance_scores = None
            if bool(getattr(retriever.config, "enable_pruning", False)):
                try:
                    entity_relevance_scores = retriever._compute_entity_relevance_scores(
                        seed_entity_ids,
                        top_k_facts,
                        query_fact_scores,
                        top_k_fact_indices,
                        owner_id=normalized_owner,
                    )
                except Exception:  # noqa: BLE001
                    entity_relevance_scores = None

            subgraph_nodes, subgraph_chunk_ids = retriever._expand_subgraph(
                seed_entity_ids,
                entity_relevance_scores=entity_relevance_scores,
                owner_id=normalized_owner,
            )
            diagnostics["subgraph_nodes"] = len(subgraph_nodes)
            diagnostics["subgraph_chunks"] = len(subgraph_chunk_ids)

            if not subgraph_chunk_ids:
                diagnostics["reason"] = "empty_subgraph"
                return [], diagnostics, False

            if not use_ppr:
                if query_doc_scores is None:
                    diagnostics["reason"] = "dense_scores_unavailable"
                    return [], diagnostics, False
                ranked: List[Tuple[str, float]] = []
                for idx, chunk_id in enumerate(getattr(retriever, "passage_node_keys", []) or []):
                    if chunk_id in subgraph_chunk_ids and idx < len(query_doc_scores):
                        ranked.append((chunk_id, float(query_doc_scores[idx])))
                ranked.sort(key=lambda item: item[1], reverse=True)
                selected = ranked[: max(1, int(top_k))] if ranked else []
                chunk_ids = [cid for cid, _score in selected]
                chunk_scores = [score for _cid, score in selected]
                chunks = retriever._convert_to_chunks(chunk_ids, chunk_scores, owner_id=normalized_owner)
                if return_subgraph_info and chunks:
                    # Minimal subgraph snapshot for visualization/tracing (not citeable evidence).
                    subgraph_info = {
                        "subgraph_nodes": list(subgraph_nodes),
                        "seed_entity_ids": list(seed_entity_ids),
                        "retrieved_chunk_ids": chunk_ids,
                        "query": query,
                        "mode": "dense_over_subgraph",
                    }
                    if chunks[0].metadata is None:
                        chunks[0].metadata = {}
                    chunks[0].metadata["_subgraph_info"] = subgraph_info
                diagnostics["selected"] = len(chunks)
                return chunks, diagnostics, False

            sorted_doc_ids, _sorted_doc_scores, ppr_scores_dict = retriever._graph_search_on_subgraph(
                query,
                query_fact_scores,
                top_k_facts,
                top_k_fact_indices,
                subgraph_nodes,
                owner_id=normalized_owner,
                query_doc_scores=query_doc_scores,
            )
            selected_chunk_ids, selected_chunk_scores, top_entity_id = retriever._select_top_entity_chunks(
                ppr_scores_dict=ppr_scores_dict,
                owner_id=normalized_owner,
                top_k=top_k,
                fallback_chunk_ids=sorted_doc_ids,
            )

            dense_score_map: Dict[str, float] = {}
            dense_mix_k = int(getattr(retriever.config, "dense_mix_in_top_k", 0) or 0)
            if dense_mix_k > 0 and query_doc_scores is not None:
                dense_sorted = np.argsort(query_doc_scores)[::-1]
                dense_ids: List[str] = []
                for idx in dense_sorted[: max(dense_mix_k * 3, dense_mix_k)]:
                    if idx < 0 or int(idx) >= len(retriever.passage_node_keys):
                        continue
                    chunk_id = retriever.passage_node_keys[int(idx)]
                    if not chunk_id or chunk_id in dense_score_map:
                        continue
                    dense_ids.append(chunk_id)
                    dense_score_map[chunk_id] = float(query_doc_scores[int(idx)])
                    if len(dense_ids) >= dense_mix_k:
                        break

                if dense_ids:
                    blended: List[str] = []
                    seen: set[str] = set()
                    for cid in dense_ids:
                        if cid in seen:
                            continue
                        seen.add(cid)
                        blended.append(cid)
                    for cid in selected_chunk_ids:
                        if cid in seen:
                            continue
                        seen.add(cid)
                        blended.append(cid)
                    blended = blended[: max(1, int(top_k))]
                    selected_chunk_ids = blended
                    selected_chunk_scores = [
                        float(ppr_scores_dict.get(cid, dense_score_map.get(cid, 0.0))) for cid in selected_chunk_ids
                    ]

            chunks = retriever._finalize_retrieval(
                query=query,
                owner_filter=normalized_owner,
                top_k=top_k,
                selected_chunk_ids=selected_chunk_ids,
                selected_chunk_scores=selected_chunk_scores,
                ppr_scores_dict=ppr_scores_dict,
                subgraph_nodes=subgraph_nodes,
                seed_entity_ids=seed_entity_ids,
                return_subgraph_info=bool(return_subgraph_info),
                top_entity_id=top_entity_id,
                dense_score_map=dense_score_map,
                dense_mix_k=dense_mix_k,
            )
            diagnostics["selected"] = len(chunks)
            return chunks, diagnostics, False

    async def _run_graph_chunk(
        self: _SearchToolBase,
        *,
        request: ToolRunRequest,
        query: str,
        top_k: int,
        file_scope,
    ) -> _ChannelResult:
        adapter = self._require_adapter(request.adapter)
        retriever = getattr(adapter, "retriever", None)
        if retriever is None:
            return _ChannelResult(
                channel="graph_chunk",
                evidences=[],
                diagnostics={"query": query, "reason": "graph_retriever_unavailable"},
                summary="graph_chunk search skipped: graph retriever unavailable.",
            )

        visibility = self._resolve_owner_visibility(request)
        if not visibility.enabled:
            return _ChannelResult(
                channel="graph_chunk",
                evidences=[],
                diagnostics={"query": query, "reason": "owner_id_missing", "owner_visibility": visibility.as_dict()},
                summary="graph_chunk search skipped: missing owner scope.",
            )

        override = request.extra.get("graph_top_k")
        effective_top_k = self._resolve_top_k(override, top_k)

        use_ppr = self._coerce_bool(request.extra.get("use_ppr"), tool_defaults.SEARCH_GRAPH_USE_PPR_DEFAULT)
        enable_llm_rerank = self._coerce_bool(
            request.extra.get("enable_llm_rerank"), tool_defaults.SEARCH_GRAPH_ENABLE_LLM_RERANK_DEFAULT
        )
        enable_entity_fallback = self._coerce_bool(
            request.extra.get("enable_entity_fallback"), tool_defaults.SEARCH_GRAPH_ENABLE_ENTITY_FALLBACK
        )
        entity_seed_top_k = self._resolve_top_k(
            request.extra.get("entity_seed_top_k"),
            tool_defaults.SEARCH_GRAPH_ENTITY_SEED_TOP_K,
        )

        overrides = self._graph_overrides(request.extra or {})

        seed_items = self._coerce_seed_items(request.extra.get("seed_entities"))
        resolved_seeds: List[str] = []
        seed_diag: Dict[str, Any] = {}
        if seed_items:
            resolved_seeds, seed_diag = await self._resolve_entity_ids(
                request=request,
                adapter=adapter,
                seeds=seed_items,
            )

        want_subgraph_info = bool(getattr(file_scope, "enabled", False))

        owner_ids = list(visibility.owner_ids_used)
        section_scope = self._resolve_section_scope(request.extra or {})
        section_dropped = 0

        # Single-owner path: keep the original LLM entity fallback semantics.
        if len(owner_ids) == 1:
            owner_id = owner_ids[0]
            async with adapter_locked(adapter):
                chunks, diagnostics, needs_llm = await get_thread_pool().run_blocking(
                    self._graph_chunk_sync,
                    retriever=retriever,
                    query=query,
                    owner_id=owner_id,
                    top_k=effective_top_k,
                    use_ppr=use_ppr,
                    enable_llm_rerank=enable_llm_rerank,
                    enable_entity_fallback=enable_entity_fallback,
                    entity_seed_top_k=entity_seed_top_k,
                    seed_override=resolved_seeds if resolved_seeds else None,
                    return_subgraph_info=want_subgraph_info,
                    overrides=overrides,
                )

            diagnostics["seed_entity_resolution"] = seed_diag

            llm_diag: Dict[str, Any] = {}
            if needs_llm and enable_entity_fallback and not resolved_seeds:
                if self.llm_connector is None and entity_seed_top_k > 0:
                    raise RuntimeError("graph_chunk requires LLM entity extraction but no LLM connector is available")
                llm_seeds, llm_diag = await self._extract_entities_with_llm(query=query, limit=entity_seed_top_k)
                if llm_seeds:
                    resolved_llm_ids, llm_resolve_diag = await self._resolve_entity_ids(
                        request=request,
                        adapter=adapter,
                        seeds=llm_seeds,
                    )
                    llm_diag["resolved"] = llm_resolve_diag
                    if resolved_llm_ids:
                        async with adapter_locked(adapter):
                            chunks, diagnostics, _ = await get_thread_pool().run_blocking(
                                self._graph_chunk_sync,
                                retriever=retriever,
                                query=query,
                                owner_id=owner_id,
                                top_k=effective_top_k,
                                use_ppr=use_ppr,
                                enable_llm_rerank=enable_llm_rerank,
                                enable_entity_fallback=enable_entity_fallback,
                                entity_seed_top_k=entity_seed_top_k,
                                seed_override=resolved_llm_ids,
                                return_subgraph_info=want_subgraph_info,
                                overrides=overrides,
                            )
                        diagnostics["llm_seed_used"] = True
                    else:
                        diagnostics["llm_seed_used"] = False
                else:
                    diagnostics["llm_seed_used"] = False
            diagnostics["llm_entity_extract"] = llm_diag

            subgraph_info = None
            if want_subgraph_info:
                for chunk in chunks or []:
                    meta = getattr(chunk, "metadata", None) or {}
                    if isinstance(meta, dict) and isinstance(meta.get("_subgraph_info"), dict) and meta["_subgraph_info"]:
                        subgraph_info = dict(meta["_subgraph_info"])
                        break

            chunks, dropped = self._apply_file_scope(chunks, file_scope)
            chunks, section_dropped = self._apply_section_scope(chunks, section_scope)

            if want_subgraph_info and subgraph_info and chunks:
                kept_chunk_ids = [str(getattr(c, "id", "") or "").strip() for c in chunks]
                kept_chunk_ids = [cid for cid in kept_chunk_ids if cid]
                if kept_chunk_ids:
                    subgraph_info["retrieved_chunk_ids"] = kept_chunk_ids
                if hasattr(file_scope, "as_dict"):
                    subgraph_info["file_scope"] = file_scope.as_dict()
                if chunks[0].metadata is None:
                    chunks[0].metadata = {}
                chunks[0].metadata["_subgraph_info"] = subgraph_info

            per_owner_diag: Dict[str, Any] = {owner_id: diagnostics}
            errors: List[str] = []
        else:
            # Multi-owner path: fan-out the base graph retrieval only (no implicit LLM entity fan-out).
            chunks = []
            per_owner_diag = {}
            errors = []
            for owner_id in owner_ids:
                async with adapter_locked(adapter):
                    part_chunks, diagnostics, needs_llm = await get_thread_pool().run_blocking(
                        self._graph_chunk_sync,
                        retriever=retriever,
                        query=query,
                        owner_id=owner_id,
                        top_k=effective_top_k,
                        use_ppr=use_ppr,
                        enable_llm_rerank=enable_llm_rerank,
                        enable_entity_fallback=enable_entity_fallback,
                        entity_seed_top_k=entity_seed_top_k,
                        seed_override=resolved_seeds if resolved_seeds else None,
                        return_subgraph_info=want_subgraph_info,
                        overrides=overrides,
                    )
                diagnostics["seed_entity_resolution"] = seed_diag
                if needs_llm and enable_entity_fallback and not resolved_seeds:
                    diagnostics["llm_seed_used"] = "skipped_multi_owner"
                per_owner_diag[owner_id] = diagnostics
                chunks.extend(part_chunks)

            subgraph_info = None
            if want_subgraph_info:
                for chunk in chunks or []:
                    meta = getattr(chunk, "metadata", None) or {}
                    if isinstance(meta, dict) and isinstance(meta.get("_subgraph_info"), dict) and meta["_subgraph_info"]:
                        subgraph_info = dict(meta["_subgraph_info"])
                        break

            chunks, dropped = self._apply_file_scope(chunks, file_scope)
            chunks, section_dropped = self._apply_section_scope(chunks, section_scope)

            if want_subgraph_info and subgraph_info and chunks:
                kept_chunk_ids = [str(getattr(c, "id", "") or "").strip() for c in chunks]
                kept_chunk_ids = [cid for cid in kept_chunk_ids if cid]
                if kept_chunk_ids:
                    subgraph_info["retrieved_chunk_ids"] = kept_chunk_ids
                if hasattr(file_scope, "as_dict"):
                    subgraph_info["file_scope"] = file_scope.as_dict()
                if chunks[0].metadata is None:
                    chunks[0].metadata = {}
                chunks[0].metadata["_subgraph_info"] = subgraph_info

        evidences: List[EvidenceChunk] = []
        results: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[:effective_top_k]):
            content = self._chunk_content(chunk)
            meta = self._chunk_meta(chunk)
            snippet = self._summary_window(content)
            chunk_id = self._chunk_id(chunk, "graph_chunk")
            score = self._chunk_score(chunk)
            file_name = self._chunk_file_name(meta)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source="graph_chunk",
                content=snippet,
                kind=EVIDENCE_KIND_DERIVED,
                score=score,
                provenance={
                    "channel": "graph_chunk",
                    "rank": idx,
                    "file_name": file_name,
                    "owner_id": getattr(chunk, "owner_id", None) or meta.get("owner_id"),
                    "evidence_class": EVIDENCE_CLASS_GRAPH_CHUNK,
                    "metadata": meta,
                },
            )
            evidences.append(evidence)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "file_name": file_name,
                    "summary": snippet,
                }
            )

        summary = (
            f"graph_chunk search returned {len(evidences)} chunks."
            if evidences
            else "graph_chunk search returned no chunks."
        )
        diagnostics_out = {
            "query": query,
            "top_k": effective_top_k,
            "file_scope_dropped": dropped,
            "section_scope": sorted(section_scope),
            "section_scope_dropped": section_dropped,
            "owner_visibility": visibility.as_dict(),
            "per_owner_diagnostics": per_owner_diag,
            "errors": errors,
            "results": results,
        }
        return _ChannelResult(channel="graph_chunk", evidences=evidences, diagnostics=diagnostics_out, summary=summary)


class SearchGraphChunkTool(_SearchToolBase, _GraphChunkChannel, GraphTool):
    """Graph-subgraph search tool (HippoRAG graph)."""

    descriptor = ToolDescriptor(
        name="search.scoped.graph",
        channel="graph",
        description="Graph-subgraph chunk retrieval without default PPR (optional) for fast localization.",
        speed="fast",
        cost="medium",
        strategy_tags=("search", "graph_chunk", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search.scoped.graph",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {"type": "string", "description": "Optional query override."},
                "file_id": {"type": "string", "description": "Restrict results to a specific file_id."},
                "file_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to file_ids."},
                "section_id": {"type": "string", "description": "Restrict results to a specific section_id."},
                "section_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict results to section_ids."},
                "owner_ids": {"type": "array", "items": {"type": "string"}, "description": "Owner ids to search (me/share)."},
                "top_k": {"type": "integer", "minimum": 0, "description": "Top-k results to return."},
                "graph_top_k": {"type": "integer", "minimum": 0, "description": "Alias of top_k."},
                "use_ppr": {"type": "boolean", "description": "Enable PPR (default off)."},
                "enable_llm_rerank": {"type": "boolean", "description": "Enable LLM fact rerank."},
                "enable_entity_fallback": {"type": "boolean", "description": "Enable entity fallback."},
                "entity_seed_top_k": {"type": "integer", "minimum": 0, "description": "Max entities from fallback."},
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional seed entities (entity_id or name).",
                },
                "graph_overrides": {
                    "type": "object",
                    "description": "Graph retrieval overrides (allowlisted safe keys only).",
                },
            }
        ),
        example_args={
            "question": "HippoRAG graph retrieval",
            "plan_step": "plan_01",
            "extra": {"top_k": 10, "use_ppr": False},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        extra = request.extra or {}
        file_scope, invalid, raw_ids = resolve_explicit_file_scope(extra)
        if invalid:
            return ToolResult(
                summary="search.scoped.graph skipped: invalid file_id(s) (expected UUID; use search.file to obtain file_id).",
                diagnostics={
                    "reason": "invalid_file_id_format",
                    "query": query,
                    "invalid_file_ids": invalid[:20],
                    "file_ids_filter_raw": raw_ids[:20],
                },
            )
        if file_scope is None or not getattr(file_scope, "file_ids", None):
            return ToolResult(
                summary="search.scoped.graph skipped: missing file_id/file_ids (call search.file first).",
                diagnostics={"reason": "missing_file_scope", "query": query},
            )
        top_k = self._resolve_top_k(extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        result = await self._run_graph_chunk(request=request, query=query, top_k=top_k, file_scope=file_scope)
        return ToolResult(summary=result.summary, evidences=result.evidences, diagnostics=result.diagnostics)


class SearchGlobalGraphTool(_SearchToolBase, _GraphChunkChannel, GraphTool):
    """Graph-subgraph global search tool (HippoRAG graph; explicitly ignores inherited file_scope)."""

    descriptor = ToolDescriptor(
        name="search.global.graph",
        channel="graph",
        description=(
            "Graph-subgraph chunk retrieval across all accessible documents. "
            "Warning: may introduce cross-document/company noise. Prefer search.scoped.graph."
        ),
        speed="fast",
        cost="medium",
        strategy_tags=("search", "graph_chunk", "global", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="F",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.search.global.graph",
        mcp_callable=True,
        input_schema=SearchGraphChunkTool.descriptor.input_schema,
        example_args={
            "question": "Find related chunks via graph signals globally",
            "plan_step": "plan_01",
            "extra": {"channels": ["graph"], "top_k": 10, "use_ppr": False},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        query = self._resolve_query(request)
        patched_ctx = strip_file_scope_from_graph_context(request.graph_context)
        extra = dict(request.extra or {})
        for key in ("file_id", "file_ids", "source_file_id", "source_file_ids", "filename_contains"):
            extra.pop(key, None)
        file_scope = None
        top_k = self._resolve_top_k(extra.get("top_k"), tool_defaults.SEARCH_DEFAULT_TOP_K)
        patched = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=extra,
            graph_context=patched_ctx,
            coverage_metrics=request.coverage_metrics,
        )
        result = await self._run_graph_chunk(request=patched, query=query, top_k=top_k, file_scope=file_scope)
        risk = "global_search_may_introduce_cross_doc_noise"
        diagnostics = {**dict(result.diagnostics or {}), "search_mode": "global", "risk": risk}
        summary = (result.summary or "graph search completed.").rstrip()
        summary = summary + f" NOTE: {risk}."
        return ToolResult(summary=summary, evidences=result.evidences, diagnostics=diagnostics)
