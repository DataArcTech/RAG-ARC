import logging
import re
import copy
from collections import defaultdict
from typing import Any, List, TYPE_CHECKING

from core.retrieval.base import BaseRetriever
from encapsulation.data_model.schema import Chunk
from core.file_management.atomic_units.markdown_table import extract_markdown_table_rows
from config.output_limits import (
    SEMANTIC_UNIT_MAX_MATCHED_SLICES,
    SEMANTIC_UNIT_MAX_MERGED_SLICE_CHARS,
    SEMANTIC_UNIT_MAX_MERGED_TOTAL_CHARS,
    TABLE_MAX_MERGED_ROWS,
    RAG_RETRIEVAL_OBSERVABILITY,
    RAG_RETRIEVAL_LOG_TOP_FILES,
)

if TYPE_CHECKING:
    from config.core.retrieval.multipath_config import MultiPathRetrieverConfig

logger = logging.getLogger(__name__)


class MultiPathRetriever(BaseRetriever):
    """
    MultiPath retriever, uses multiple retrievers and fuses results.

    Supported fusion methods:
    - rrf: Reciprocal Rank Fusion
    - weighted_sum: weighted sum
    - rank_fusion: rank-based fusion
    """
    
    def __init__(self, config: "MultiPathRetrieverConfig"):
        """Initialize MultiPathRetriever"""
        self.config = config
        self._index = None
        self._embedding = None
        self._init_fusion_method()

    def _init_fusion_method(self):
        """Initialize fusion method"""
        if self.config.fusion_method == "rrf":
            from core.utils.fusion import RRFusion
            self.config.fusion_instance = RRFusion(k=self.config.rrf_k, weights=self.config.weights)
        elif self.config.fusion_method == "weighted_sum":
            from core.utils.fusion import WeightedSumFusion
            weights = self.config.weights or [1.0] * len(self.config.retrievers)
            self.config.fusion_instance = WeightedSumFusion(weights=weights)
        elif self.config.fusion_method == "rank_fusion":
            from core.utils.fusion import RankFusion
            self.config.fusion_instance = RankFusion()
        else:
            from core.utils.fusion import RRFusion
            self.config.fusion_instance = RRFusion(k=self.config.rrf_k, weights=self.config.weights)

    def _get_relevant_chunks(self, query: str, **kwargs: Any) -> List[Chunk]:
        """Search relevant chunks and fuse results"""
        k = kwargs.get('k', self.config.search_kwargs.get('k', 5))
        bm25_query = str(kwargs.get("bm25_query") or "").strip() or None

        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")

        if not query.strip():
            return []

        # Optional query variants (e.g. Hans<->Hant) to improve recall on mixed-script corpora.
        def _variants(text: str) -> list[str]:
            try:
                from core.utils.query_variants import generate_query_variants

                return generate_query_variants(text)
            except Exception:
                return [text]

        all_results = []
        subgraph_info = None  # Store subgraph info from graph retriever
        routing_ratios = kwargs.get("retrieval_ratios")
        ratio_by_key: dict[str, float] = {}

        # Determine per-retriever quotas (coverage floor) based on routing ratios.
        # - If routing_ratios is absent, fall back to the configured fusion weights.
        # - If a retriever is effectively disabled (weight<=0), it contributes 0 quota and is not invoked.
        try:
            from core.utils.quota import allocate_quotas

            weights = list(self.config.weights) if self.config.weights else [1.0] * len(self.config.retrievers)
            # Ensure length matches retrievers.
            if len(weights) < len(self.config.retrievers):
                weights = weights + [1.0] * (len(self.config.retrievers) - len(weights))

            if isinstance(routing_ratios, dict):
                for key in ("dense", "bm25", "graph"):
                    val = routing_ratios.get(key)
                    if isinstance(val, (int, float)):
                        ratio_by_key[key] = float(val)
                    else:
                        try:
                            ratio_by_key[key] = float(val)
                        except Exception:  # noqa: BLE001
                            pass

            effective_ratios: list[float] = []
            for idx, cfg in enumerate(self.config.retrievers or []):
                t = str(getattr(cfg, "type", "") or "").strip()
                w = float(weights[idx]) if idx < len(weights) else 1.0
                if w <= 0:
                    effective_ratios.append(0.0)
                    continue
                if t == "dense":
                    effective_ratios.append(float(ratio_by_key.get("dense", w)))
                elif t == "tantivy_bm25":
                    effective_ratios.append(float(ratio_by_key.get("bm25", w)))
                elif t in {"pruned_hipporag_neo4j_retrieval", "pruned_hipporag_retrieval"}:
                    effective_ratios.append(float(ratio_by_key.get("graph", w)))
                else:
                    effective_ratios.append(float(w))

            quota = allocate_quotas(int(k), effective_ratios)
            quotas_by_idx = quota.quotas
            if RAG_RETRIEVAL_OBSERVABILITY:
                logger.info(
                    "multipath.quota owner_id=%s query=%r k_total=%s retrieval_ratios=%s quotas=%s weights=%s",
                    kwargs.get("owner_id"),
                    (query[:240] + "…") if len(query) > 240 else query,
                    int(k),
                    {k: ratio_by_key.get(k) for k in ("dense", "bm25", "graph") if k in ratio_by_key},
                    quotas_by_idx,
                    weights,
                )
        except Exception:  # noqa: BLE001
            quotas_by_idx = [0 for _ in (self.config.retrievers or [])]

        def _coerce_file_id(meta: Any) -> str | None:
            if not isinstance(meta, dict):
                return None
            for key in ("source_file_id", "sourceFileId", "file_id", "fileId", "document_id", "documentId"):
                token = str(meta.get(key) or "").strip()
                if token:
                    return token
            return None

        def _file_dist(chunks: list[Chunk], *, limit: int) -> list[tuple[str, int]]:
            from collections import Counter

            ctr: Counter[str] = Counter()
            for ch in chunks:
                meta = getattr(ch, "metadata", None) or {}
                fid = _coerce_file_id(meta)
                if fid:
                    ctr[fid] += 1
            return ctr.most_common(max(int(limit), 0))

        def _log_dist(stage: str, name: str, chunks: list[Chunk]) -> None:
            if not RAG_RETRIEVAL_OBSERVABILITY:
                return
            owner_scope = kwargs.get("owner_id")
            dist = _file_dist(chunks, limit=RAG_RETRIEVAL_LOG_TOP_FILES)
            logger.info(
                "multipath.%s query=%r owner_id=%s retriever=%s chunks=%d top_files=%s",
                stage,
                (query[:240] + "…") if len(query) > 240 else query,
                owner_scope,
                name,
                len(chunks),
                dist,
            )

        for idx, retriever in enumerate(self.config.built_retrievers or []):
            try:
                # Routing ratios can explicitly disable a backend by setting its ratio to 0.
                # This is LLM-driven (from rewrite_query_with_routing), not hardcoded.
                if isinstance(routing_ratios, dict):
                    try:
                        cfg_t = str(getattr(self.config.retrievers[idx], "type", "") or "").strip()
                    except Exception:  # noqa: BLE001
                        cfg_t = ""
                    disabled = False
                    if cfg_t == "dense" and float(ratio_by_key.get("dense", 1.0)) <= 0:
                        disabled = True
                    elif cfg_t == "tantivy_bm25" and float(ratio_by_key.get("bm25", 1.0)) <= 0:
                        disabled = True
                    elif cfg_t in {"pruned_hipporag_neo4j_retrieval", "pruned_hipporag_retrieval"} and float(ratio_by_key.get("graph", 1.0)) <= 0:
                        disabled = True
                    if disabled:
                        all_results.append([])
                        continue

                # Allow disabling a retriever via weight=0 (skip invocation entirely).
                if self.config.weights is not None and idx < len(self.config.weights):
                    try:
                        if float(self.config.weights[idx]) <= 0:
                            all_results.append([])
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                # Union results across variants per retriever.
                #
                # Important: keep the *best* (lowest) rank of each chunk across variants.
                # A naive append-based union would unfairly demote chunks that are only retrieved
                # by a later variant query (e.g. Simplified/Traditional), which then breaks RRF.

                def _chunk_key(chunk: Chunk) -> str:
                    cid = str(getattr(chunk, "id", "") or "").strip()
                    if cid:
                        return f"id:{cid}"
                    meta = getattr(chunk, "metadata", None) or {}
                    fid = _coerce_file_id(meta)
                    if fid:
                        return f"file:{fid}:{hash(getattr(chunk, 'content', '') or '')}"
                    return f"content:{hash(getattr(chunk, 'content', '') or '')}"

                best_by_key: dict[str, tuple[int, int, Chunk]] = {}
                seq = 0
                cfg_t = str(getattr(self.config.retrievers[idx], "type", "") or "").strip()
                per_query = bm25_query if (cfg_t == "tantivy_bm25" and bm25_query) else query
                for qv in _variants(per_query):
                    qv_chunks = retriever.invoke(qv, **kwargs) or []
                    for rank, chunk in enumerate(qv_chunks, 1):
                        seq += 1
                        key = _chunk_key(chunk)
                        prev = best_by_key.get(key)
                        if prev is None or int(rank) < int(prev[0]):
                            # (best_rank, first_seen_seq, chunk)
                            best_by_key[key] = (int(rank), int(prev[1] if prev else seq), chunk)

                chunks = [t[2] for t in sorted(best_by_key.values(), key=lambda x: (x[0], x[1]))]

                # Stash per-retriever top quota for later quota-guaranteed candidate assembly.
                # We do not trim `chunks` here because fusion still needs the full ranked list.
                if chunks:
                    q = 0
                    if idx < len(quotas_by_idx):
                        q = int(quotas_by_idx[idx] or 0)
                    if q > 0:
                        for ch in chunks[: min(q, len(chunks))]:
                            meta = getattr(ch, "metadata", None) or {}
                            meta.setdefault("_multipath_quota_source", str(getattr(self.config.retrievers[idx], "type", "") or type(retriever).__name__))
                            ch.metadata = meta

                # Check if this is a graph retriever with subgraph info
                if chunks and hasattr(chunks[0], 'metadata') and chunks[0].metadata:
                    if '_subgraph_info' in chunks[0].metadata:
                        subgraph_info = chunks[0].metadata.pop('_subgraph_info')
                        logger.info(f"Captured subgraph info from {type(retriever).__name__}")

                # Ensure each chunk has a score
                for chunk in chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    if 'score' not in chunk.metadata:
                        chunk.metadata['score'] = 1.0
                all_results.append(chunks)
                _log_dist("retriever", type(retriever).__name__, chunks)
                logger.debug(f"Retriever {type(retriever).__name__} returned {len(chunks)} results")
            except Exception as e:
                logger.error("Retriever %s failed: %s", type(retriever).__name__, e)
                logger.debug("Retriever %s traceback", type(retriever).__name__, exc_info=True)
                all_results.append([])

        if not all_results or all(len(results) == 0 for results in all_results):
            return []

        fused_chunks = self.config.fusion_instance.fuse(all_results, k)
        _log_dist("fused", type(self.config.fusion_instance).__name__, fused_chunks)

        # Ensure each retriever contributes at least its quota into the final candidate set.
        # This mitigates "one backend drowns out others" regressions on sparse-detail queries.
        try:
            from core.utils.fusion import _fusion_key as _fk  # stable key aligned with fusion

            seeds: list[Chunk] = []
            seen: set[str] = set()
            for idx, results in enumerate(all_results):
                q = int(quotas_by_idx[idx]) if idx < len(quotas_by_idx) else 0
                if q <= 0:
                    continue
                for ch in results[: min(q, len(results))]:
                    key = _fk(ch)
                    if key in seen:
                        continue
                    seen.add(key)
                    seeds.append(ch)

            if seeds:
                fused_rank = {_fk(ch): i for i, ch in enumerate(fused_chunks)}

                def _seed_sort_key(ch: Chunk) -> tuple[int, int]:
                    key = _fk(ch)
                    rank = fused_rank.get(key)
                    return (rank if rank is not None else 1_000_000_000, 0)

                seeds_sorted = sorted(seeds, key=_seed_sort_key)
                merged: list[Chunk] = []
                merged_seen: set[str] = set()
                for ch in seeds_sorted:
                    key = _fk(ch)
                    if key in merged_seen:
                        continue
                    merged_seen.add(key)
                    merged.append(ch)
                    if len(merged) >= int(k):
                        break
                if len(merged) < int(k):
                    for ch in fused_chunks:
                        key = _fk(ch)
                        if key in merged_seen:
                            continue
                        merged_seen.add(key)
                        merged.append(ch)
                        if len(merged) >= int(k):
                            break
                fused_chunks = merged
        except Exception:  # noqa: BLE001
            pass

        fused_chunks = self._postprocess_anchor_backfill(fused_chunks, owner_id=kwargs.get("owner_id"))

        # Attach subgraph info to first chunk if available
        if subgraph_info and fused_chunks:
            if fused_chunks[0].metadata is None:
                fused_chunks[0].metadata = {}
            fused_chunks[0].metadata['_subgraph_info'] = subgraph_info
            logger.info("Attached subgraph info to fused results")

        return fused_chunks

    def _select_backfill_index(self) -> Any | None:
        for retriever in self.config.built_retrievers or []:
            index = getattr(retriever, "index", None)
            if index is not None and hasattr(index, "get_by_ids"):
                return index
        return None

    @staticmethod
    def _chunk_score(chunk: Chunk) -> float | None:
        metadata = getattr(chunk, "metadata", None) or {}
        score = metadata.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None

    @staticmethod
    def _as_matched_slice(chunk: Chunk) -> dict:
        metadata = getattr(chunk, "metadata", None) or {}
        index_text = metadata.get("index_text")
        matched_content = chunk.content
        if isinstance(index_text, str) and index_text.strip():
            matched_content = index_text
        return {
            "id": getattr(chunk, "id", None),
            "content": matched_content,
            "score": metadata.get("score"),
            "metadata": copy.deepcopy(metadata),
        }

    @staticmethod
    def _slice_index_text(chunk: Chunk) -> str:
        metadata = getattr(chunk, "metadata", None) or {}
        index_text = metadata.get("index_text")
        if isinstance(index_text, str) and index_text.strip():
            return index_text
        return chunk.content or ""

    @staticmethod
    def _clone_chunk(chunk: Chunk) -> Chunk:
        metadata = getattr(chunk, "metadata", None) or {}
        return Chunk(
            id=getattr(chunk, "id", None),
            owner_id=getattr(chunk, "owner_id", None),
            domain=getattr(chunk, "domain", None),
            content=chunk.content,
            metadata=copy.deepcopy(metadata),
            graph=getattr(chunk, "graph", None),
        )

    @staticmethod
    def _table_rows_from_slice_content(content: str) -> list[str]:
        return extract_markdown_table_rows(content)

    @staticmethod
    def _truncate_text(text: str, max_chars: int | None) -> str:
        if max_chars is None:
            return text
        max_chars = max(int(max_chars), 0)
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n…"

    def _postprocess_anchor_backfill(self, chunks: List[Chunk], *, owner_id: Any = None) -> List[Chunk]:
        """
        Replace slice hits with their anchor chunk (by `anchor_chunk_id`) and attach matched slices.

        This keeps RAG prompt + evidence output consistent by ensuring downstream sees a single
        chunk per semantic unit, with `matched_slices` preserved in metadata for debugging/UX.
        """
        if not chunks:
            return chunks

        index = self._select_backfill_index()
        if index is None:
            return chunks

        placeholders: List[Chunk | tuple[str, str]] = []
        slice_groups: dict[str, list[Chunk]] = defaultdict(list)
        placed_anchor_ids: set[str] = set()

        def _chunk_role(chunk: Chunk) -> str:
            metadata = getattr(chunk, "metadata", None) or {}
            return str(metadata.get("chunk_role") or "").strip()

        def _build_anchor_output(anchor_chunk: Chunk | None, slices: list[Chunk], *, anchor_id: str) -> Chunk | None:
            def _slice_sort_key(chunk: Chunk) -> tuple[float, int]:
                metadata = getattr(chunk, "metadata", None) or {}
                score = self._chunk_score(chunk)
                tie_break = 1_000_000_000
                if isinstance(metadata.get("slice_index"), int):
                    tie_break = int(metadata["slice_index"])
                row_range = metadata.get("row_range") if isinstance(metadata.get("row_range"), dict) else None
                if row_range and isinstance(row_range.get("start"), int):
                    tie_break = int(row_range["start"])
                line_range = metadata.get("line_range") if isinstance(metadata.get("line_range"), dict) else None
                if line_range and isinstance(line_range.get("start"), int):
                    tie_break = int(line_range["start"])
                item_range = metadata.get("list_item_range") if isinstance(metadata.get("list_item_range"), dict) else None
                if item_range and isinstance(item_range.get("start"), int):
                    tie_break = int(item_range["start"])
                return (-(score if score is not None else 0.0), tie_break)

            if not slices:
                return anchor_chunk

            selected_slices = sorted(slices, key=_slice_sort_key)
            if SEMANTIC_UNIT_MAX_MATCHED_SLICES is not None:
                selected_slices = selected_slices[:SEMANTIC_UNIT_MAX_MATCHED_SLICES]

            slice_owner = getattr(slices[0], "owner_id", None)
            if anchor_chunk is None:
                fallback = self._clone_chunk(slices[0])
                if fallback.metadata is None:
                    fallback.metadata = {}
                fallback.metadata["anchor_missing"] = True
                fallback.metadata["matched_slices"] = [self._as_matched_slice(s) for s in selected_slices]
                fallback.metadata.setdefault("prompt_text", fallback.content)
                return fallback

            if slice_owner and getattr(anchor_chunk, "owner_id", None) and str(anchor_chunk.owner_id) != str(slice_owner):
                logger.warning("Anchor owner mismatch for %s; using slice fallback", anchor_id)
                fallback = self._clone_chunk(slices[0])
                if fallback.metadata is None:
                    fallback.metadata = {}
                fallback.metadata["anchor_owner_mismatch"] = True
                fallback.metadata["matched_slices"] = [self._as_matched_slice(s) for s in selected_slices]
                fallback.metadata.setdefault("prompt_text", fallback.content)
                return fallback

            anchor_out = self._clone_chunk(anchor_chunk)
            if anchor_out.metadata is None:
                anchor_out.metadata = {}
            anchor_out.metadata["matched_slices"] = [self._as_matched_slice(s) for s in selected_slices]

            slice_scores = [score for score in (self._chunk_score(s) for s in selected_slices) if score is not None]
            if slice_scores:
                anchor_score = self._chunk_score(anchor_out)
                anchor_out.metadata["score"] = (
                    max([anchor_score] + slice_scores) if anchor_score is not None else max(slice_scores)
                )

            anchor_is_summary = bool(anchor_out.metadata.get("anchor_is_summary"))
            semantic_unit_type = str(anchor_out.metadata.get("semantic_unit_type") or "")

            base_prompt = str(anchor_out.metadata.get("index_text") or anchor_out.content or "")

            if semantic_unit_type in {"table", "code", "list"}:
                total_budget = SEMANTIC_UNIT_MAX_MERGED_TOTAL_CHARS
                slice_budget = SEMANTIC_UNIT_MAX_MERGED_SLICE_CHARS

                merged: list[str] = []
                total_used = 0
                for slice_chunk in selected_slices:
                    snippet = self._slice_index_text(slice_chunk)
                    if not snippet.strip():
                        continue

                    # Do not split semantic-unit slices. Enforce budgets by selection only.
                    if slice_budget is not None and len(snippet) > int(slice_budget):
                        # Keep at least one intact slice when it's the only way to provide evidence.
                        if merged:
                            continue

                    if total_budget is not None and (total_used + len(snippet)) > int(total_budget):
                        break

                    merged.append(snippet)
                    total_used += len(snippet)

                if merged:
                    prompt = base_prompt.rstrip() + "\n\nMatched slices:\n" + "\n\n".join(merged)
                    anchor_out.metadata["prompt_text"] = prompt
                    if anchor_is_summary:
                        anchor_out.content = base_prompt.rstrip() + "\n\nMatched slices:\n" + "\n\n".join(merged)
                else:
                    anchor_out.metadata.setdefault("prompt_text", base_prompt)

            if semantic_unit_type not in {"table", "code", "list"}:
                anchor_out.metadata.setdefault("prompt_text", base_prompt)

            return anchor_out

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None) or {}
            role = _chunk_role(chunk)
            anchor_id = str(metadata.get("anchor_chunk_id") or "").strip()

            if role == "slice" and anchor_id:
                slice_groups[anchor_id].append(chunk)
                if anchor_id not in placed_anchor_ids:
                    placeholders.append(("anchor", anchor_id))
                    placed_anchor_ids.add(anchor_id)
                continue

            if role == "anchor" and getattr(chunk, "id", None):
                existing_anchor_id = str(getattr(chunk, "id") or "").strip()
                if not existing_anchor_id:
                    placeholders.append(chunk)
                    continue
                replaced = False
                for idx, item in enumerate(placeholders):
                    if isinstance(item, tuple) and item[0] == "anchor" and item[1] == existing_anchor_id:
                        placeholders[idx] = chunk
                        replaced = True
                        break
                if not replaced and existing_anchor_id not in placed_anchor_ids:
                    placeholders.append(chunk)
                    placed_anchor_ids.add(existing_anchor_id)
                continue

            placeholders.append(chunk)

        missing_anchor_ids: list[str] = []
        for item in placeholders:
            if isinstance(item, tuple) and item[0] == "anchor":
                missing_anchor_ids.append(item[1])

        fetched: dict[str, Chunk] = {}
        if missing_anchor_ids:
            try:
                for anchor_chunk in index.get_by_ids(missing_anchor_ids):
                    if anchor_chunk is None or not getattr(anchor_chunk, "id", None):
                        continue
                    fetched[str(anchor_chunk.id)] = anchor_chunk
            except Exception as e:
                logger.warning("Anchor backfill get_by_ids failed: %s", e)

        output: List[Chunk] = []
        for item in placeholders:
            if not isinstance(item, tuple):
                anchor_id = str(getattr(item, "id", None) or "").strip()
                if anchor_id and _chunk_role(item) == "anchor" and anchor_id in slice_groups:
                    enriched = _build_anchor_output(item, slice_groups.get(anchor_id, []), anchor_id=anchor_id)
                    if enriched is not None:
                        output.append(enriched)
                    continue

                output.append(item)
                continue

            _, anchor_id = item
            anchor_chunk = fetched.get(anchor_id)
            slices = slice_groups.get(anchor_id, [])
            if not slices:
                if anchor_chunk is not None:
                    output.append(anchor_chunk)
                continue
            enriched = _build_anchor_output(anchor_chunk, slices, anchor_id=anchor_id)
            if enriched is not None:
                output.append(enriched)

        return output


    @property
    def retrievers(self):
        """Expose built retrievers for external access"""
        return self.config.built_retrievers or []

    def get_multipath_info(self) -> dict:
        retrievers = self.config.built_retrievers or []
        return {
            "retriever_count": len(retrievers),
            "retriever_types": [type(r).__name__ for r in retrievers],
            "fusion_method": self.config.fusion_method,
            "search_kwargs": self.config.search_kwargs
        }
