"""Hybrid entity/name resolution for DeepSearch graph tools.

Design goals (see docs-proj/docs-proj/plans/2026-01-15-deepsearch-online-entity-resolution-plan.md):
- Shared capability across all graph tools (neighbors/path/intersection/...).
- Conservative auto-resolve: prefer returning candidates over mismatching.
- Performance-aware: tools call this only on 0-hit/ambiguous paths.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher
from core.utils.retrieval_helper import RetrievalHelper
from core.utils.text_processing import text_processing


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tokenize(normalized: str, *, min_len: int) -> List[str]:
    tokens = [t for t in str(normalized or "").split() if len(t) >= max(1, int(min_len))]
    # Keep token order stable but dedupe.
    seen: set[str] = set()
    out: List[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _token_f1(query_tokens: Sequence[str], cand_tokens: Sequence[str]) -> float:
    q = set(query_tokens)
    c = set(cand_tokens)
    if not q or not c:
        return 0.0
    inter = len(q.intersection(c))
    if inter <= 0:
        return 0.0
    precision = inter / max(1, len(c))
    recall = inter / max(1, len(q))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class EntityResolutionCandidate:
    entity_id: str
    entity_name: str
    entity_name_normalized: str
    entity_type: str
    entity_type_key: str
    strategy: str  # exact|alias|token|faiss
    hit_count: int = 0
    edge_count: int = 0
    mention_count: int = 0
    faiss_score: float | None = None
    score: float = 0.0
    score_breakdown: Mapping[str, float] | None = None


@dataclass(frozen=True)
class EntityResolutionResult:
    raw: str
    normalized: str
    entity_type_hint: str | None
    resolved: bool
    resolved_candidate: EntityResolutionCandidate | None
    candidates: Sequence[EntityResolutionCandidate]
    diagnostics: Mapping[str, Any]


class EntityResolver:
    """Resolve a noisy entity surface form into a canonical graph Entity node."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        candidate_limit: int = 12,
        min_token_len: int = 3,
        min_token_hits: int = 2,
        auto_score_min: float = 0.86,
        auto_score_margin: float = 0.06,
        enable_alias: bool = True,
        enable_token_overlap: bool = True,
        enable_embedding_fallback: bool = True,
        faiss_top_k: int = 16,
        faiss_min_similarity: float | None = None,
        validate_edges_first: bool = True,
        require_min_edge_count: int = 1,
        enable_chunk_validation: bool = True,
        require_min_mention_count: int = 1,
        score_weight_token_f1: float = 0.7,
        score_weight_char_ratio: float = 0.3,
        alias_score_bonus: float = 0.12,
    ) -> None:
        self.enabled = bool(enabled)
        self.candidate_limit = max(1, int(candidate_limit))
        self.min_token_len = max(1, int(min_token_len))
        self.min_token_hits = max(1, int(min_token_hits))
        self.auto_score_min = float(auto_score_min)
        self.auto_score_margin = float(auto_score_margin)
        self.enable_alias = bool(enable_alias)
        self.enable_token_overlap = bool(enable_token_overlap)
        self.enable_embedding_fallback = bool(enable_embedding_fallback)
        self.faiss_top_k = max(1, int(faiss_top_k))
        self.faiss_min_similarity = None if faiss_min_similarity is None else float(faiss_min_similarity)
        self.validate_edges_first = bool(validate_edges_first)
        self.require_min_edge_count = max(0, int(require_min_edge_count))
        self.enable_chunk_validation = bool(enable_chunk_validation)
        self.require_min_mention_count = max(0, int(require_min_mention_count))

        # Keep weights stable but normalize defensively.
        w1 = max(0.0, float(score_weight_token_f1))
        w2 = max(0.0, float(score_weight_char_ratio))
        total = w1 + w2
        if total <= 0:
            w1, w2, total = 0.7, 0.3, 1.0
        self.score_weight_token_f1 = w1 / total
        self.score_weight_char_ratio = w2 / total
        self.alias_score_bonus = max(0.0, float(alias_score_bonus))

    async def resolve(
        self,
        *,
        adapter: Any,
        access_scope: Optional[GraphAccessScope],
        raw_entity: str,
        entity_type_hint: str = "",
    ) -> EntityResolutionResult:
        raw = str(raw_entity or "").strip()
        normalized = text_processing(raw)

        if not self.enabled:
            return EntityResolutionResult(
                raw=raw,
                normalized=normalized,
                entity_type_hint=entity_type_hint or None,
                resolved=False,
                resolved_candidate=None,
                candidates=(),
                diagnostics={"disabled": True},
            )
        if not normalized:
            return EntityResolutionResult(
                raw=raw,
                normalized=normalized,
                entity_type_hint=entity_type_hint or None,
                resolved=False,
                resolved_candidate=None,
                candidates=(),
                diagnostics={"reason": "empty_entity"},
            )
        if adapter is None or not adapter_supports_cypher(adapter):
            return EntityResolutionResult(
                raw=raw,
                normalized=normalized,
                entity_type_hint=entity_type_hint or None,
                resolved=False,
                resolved_candidate=None,
                candidates=(),
                diagnostics={"reason": "cypher_unavailable"},
            )

        attempts: Dict[str, Any] = {"alias": False, "token_overlap": False, "faiss": False}
        candidates: List[EntityResolutionCandidate] = []

        # 1) Alias/canonical layer (high signal, cheap).
        if self.enable_alias:
            attempts["alias"] = True
            alias_cands = await self._candidates_from_alias_layer(
                adapter=adapter,
                access_scope=access_scope,
                normalized=normalized,
                entity_type_hint=entity_type_hint,
                limit=self.candidate_limit,
            )
            candidates = self._merge_candidates(candidates, alias_cands)

        # 2) Token overlap recall (robust to suffix/parentheses noise).
        if self.enable_token_overlap:
            attempts["token_overlap"] = True
            token_cands = await self._candidates_from_token_overlap(
                adapter=adapter,
                access_scope=access_scope,
                normalized=normalized,
                entity_type_hint=entity_type_hint,
                limit=self.candidate_limit,
            )
            candidates = self._merge_candidates(candidates, token_cands)

        # 3) Embedding fallback (rarely triggered; adapter-dependent).
        if self.enable_embedding_fallback and not candidates:
            attempts["faiss"] = True
            faiss_cands = await self._candidates_from_entity_faiss(
                adapter=adapter,
                access_scope=access_scope,
                raw=raw,
                normalized=normalized,
                entity_type_hint=entity_type_hint,
                limit=self.candidate_limit,
            )
            candidates = self._merge_candidates(candidates, faiss_cands)

        scored = self._score_candidates(normalized, candidates)
        scored = sorted(scored, key=lambda c: (c.score, c.hit_count, c.edge_count), reverse=True)[: self.candidate_limit]

        resolved_candidate: EntityResolutionCandidate | None = None
        if scored:
            resolved_candidate = await self._maybe_auto_resolve(
                adapter=adapter,
                access_scope=access_scope,
                normalized=normalized,
                entity_type_hint=entity_type_hint,
                scored=scored,
            )

        diagnostics = {
            "attempts": attempts,
            "policy": {
                "auto_score_min": self.auto_score_min,
                "auto_score_margin": self.auto_score_margin,
                "min_token_len": self.min_token_len,
                "min_token_hits": self.min_token_hits,
                "validate_edges_first": self.validate_edges_first,
                "require_min_edge_count": self.require_min_edge_count,
                "enable_chunk_validation": self.enable_chunk_validation,
                "require_min_mention_count": self.require_min_mention_count,
                "enable_embedding_fallback": self.enable_embedding_fallback,
                "faiss_top_k": self.faiss_top_k,
                "faiss_min_similarity": self.faiss_min_similarity,
            },
        }
        if resolved_candidate is not None:
            diagnostics["resolved_entity"] = {
                "entity_id": resolved_candidate.entity_id,
                "entity_name": resolved_candidate.entity_name,
                "entity_name_normalized": resolved_candidate.entity_name_normalized,
                "entity_type": resolved_candidate.entity_type,
                "entity_type_key": resolved_candidate.entity_type_key,
                "score": resolved_candidate.score,
                "strategy": resolved_candidate.strategy,
                "edge_count": resolved_candidate.edge_count,
                "mention_count": resolved_candidate.mention_count,
            }
        return EntityResolutionResult(
            raw=raw,
            normalized=normalized,
            entity_type_hint=entity_type_hint or None,
            resolved=resolved_candidate is not None,
            resolved_candidate=resolved_candidate,
            candidates=tuple(scored),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _merge_candidates(
        existing: Sequence[EntityResolutionCandidate],
        incoming: Sequence[EntityResolutionCandidate],
    ) -> List[EntityResolutionCandidate]:
        by_id: Dict[str, EntityResolutionCandidate] = {c.entity_id: c for c in existing if c.entity_id}
        for cand in incoming:
            if not cand.entity_id:
                continue
            prev = by_id.get(cand.entity_id)
            if prev is None:
                by_id[cand.entity_id] = cand
                continue
            # Prefer higher-signal strategy & counts, but keep fields stable.
            if cand.strategy == "alias" and prev.strategy != "alias":
                by_id[cand.entity_id] = cand
                continue
            if cand.hit_count > prev.hit_count:
                by_id[cand.entity_id] = cand
                continue
            if cand.edge_count > prev.edge_count:
                by_id[cand.entity_id] = cand
                continue
        return list(by_id.values())

    def _score_candidates(self, normalized_query: str, candidates: Sequence[EntityResolutionCandidate]) -> List[EntityResolutionCandidate]:
        query_tokens = _tokenize(normalized_query, min_len=self.min_token_len)
        out: List[EntityResolutionCandidate] = []
        for cand in candidates:
            cand_norm = str(cand.entity_name_normalized or "")
            cand_tokens = _tokenize(cand_norm, min_len=self.min_token_len)
            token_f1 = _token_f1(query_tokens, cand_tokens)
            char_ratio = SequenceMatcher(None, normalized_query, cand_norm).ratio() if cand_norm else 0.0
            score = self.score_weight_token_f1 * token_f1 + self.score_weight_char_ratio * char_ratio
            if cand.strategy == "alias":
                score = min(1.0, score + self.alias_score_bonus)
            breakdown = {"token_f1": token_f1, "char_ratio": char_ratio}
            out.append(
                EntityResolutionCandidate(
                    entity_id=cand.entity_id,
                    entity_name=cand.entity_name,
                    entity_name_normalized=cand.entity_name_normalized,
                    entity_type=cand.entity_type,
                    entity_type_key=cand.entity_type_key,
                    strategy=cand.strategy,
                    hit_count=cand.hit_count,
                    edge_count=cand.edge_count,
                    mention_count=cand.mention_count,
                    faiss_score=cand.faiss_score,
                    score=float(score),
                    score_breakdown=breakdown,
                )
            )
        return out

    async def _maybe_auto_resolve(
        self,
        *,
        adapter: Any,
        access_scope: Optional[GraphAccessScope],
        normalized: str,
        entity_type_hint: str,
        scored: Sequence[EntityResolutionCandidate],
    ) -> EntityResolutionCandidate | None:
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None

        if best.score < self.auto_score_min:
            return None
        if second is not None and (best.score - float(second.score)) < self.auto_score_margin:
            return None

        if self.validate_edges_first and self.require_min_edge_count > 0:
            if best.edge_count < self.require_min_edge_count:
                # Only pay chunk-validation cost for the best candidate.
                if self.enable_chunk_validation and self.require_min_mention_count > 0:
                    mentions = await self._mention_count(
                        adapter=adapter,
                        access_scope=access_scope,
                        entity_id=best.entity_id,
                    )
                    if mentions < self.require_min_mention_count:
                        return None
                    # Update mention_count to make the decision observable to the caller.
                    return EntityResolutionCandidate(
                        **{**best.__dict__, "mention_count": mentions}  # type: ignore[arg-type]
                    )
                return None
        return best

    async def _candidates_from_alias_layer(
        self,
        *,
        adapter: Any,
        access_scope: Optional[GraphAccessScope],
        normalized: str,
        entity_type_hint: str,
        limit: int,
    ) -> List[EntityResolutionCandidate]:
        entity_type = str(entity_type_hint or "").strip()
        cypher = """
        MATCH (a:EntityAlias)
        WHERE a.owner_id = $owner_id
          AND a.alias_text_normalized = $entity
        MATCH (a)-[:ALIAS_OF]->(c:EntityCanonical)<-[:HAS_CONCEPT]-(e:Entity)
        WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
          AND ($entity_type = '' OR e.entity_type = $entity_type)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(:Entity)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
        RETURN e.entity_id AS entity_id,
               e.entity_name AS entity_name,
               e.entity_name_normalized AS entity_name_normalized,
               e.entity_type AS entity_type,
               e.entity_type_key AS entity_type_key,
               999 AS hit_count,
               count(r) AS edge_count
        ORDER BY edge_count DESC, entity_name_normalized ASC
        LIMIT $limit
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"entity": normalized, "entity_type": entity_type, "limit": int(limit)},
                access_scope=access_scope,
            )
        out: List[EntityResolutionCandidate] = []
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            entity_id = str(row.get("entity_id") or "").strip()
            if not entity_id:
                continue
            out.append(
                EntityResolutionCandidate(
                    entity_id=entity_id,
                    entity_name=str(row.get("entity_name") or "").strip(),
                    entity_name_normalized=str(row.get("entity_name_normalized") or "").strip(),
                    entity_type=str(row.get("entity_type") or "").strip(),
                    entity_type_key=str(row.get("entity_type_key") or "").strip(),
                    strategy="alias",
                    hit_count=_safe_int(row.get("hit_count"), 0),
                    edge_count=_safe_int(row.get("edge_count"), 0),
                )
            )
        return out

    async def _candidates_from_token_overlap(
        self,
        *,
        adapter: Any,
        access_scope: Optional[GraphAccessScope],
        normalized: str,
        entity_type_hint: str,
        limit: int,
    ) -> List[EntityResolutionCandidate]:
        entity_type = str(entity_type_hint or "").strip()
        tokens = _tokenize(normalized, min_len=self.min_token_len)
        if not tokens:
            return []
        min_hits = max(1, self.min_token_hits)
        limit = max(1, int(limit))
        cypher = """
        UNWIND $tokens AS tok
        MATCH (e:Entity)
        WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
          AND ($entity_type = '' OR e.entity_type = $entity_type)
          AND e.entity_name_normalized CONTAINS tok
        WITH e, count(DISTINCT tok) AS hit_count
        WHERE hit_count >= $min_hits
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(:Entity)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
        RETURN e.entity_id AS entity_id,
               e.entity_name AS entity_name,
               e.entity_name_normalized AS entity_name_normalized,
               e.entity_type AS entity_type,
               e.entity_type_key AS entity_type_key,
               hit_count AS hit_count,
               count(r) AS edge_count
        ORDER BY hit_count DESC, edge_count DESC, entity_name_normalized ASC
        LIMIT $limit
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"tokens": tokens, "min_hits": int(min_hits), "limit": int(limit), "entity_type": entity_type},
                access_scope=access_scope,
            )
        out: List[EntityResolutionCandidate] = []
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            entity_id = str(row.get("entity_id") or "").strip()
            if not entity_id:
                continue
            out.append(
                EntityResolutionCandidate(
                    entity_id=entity_id,
                    entity_name=str(row.get("entity_name") or "").strip(),
                    entity_name_normalized=str(row.get("entity_name_normalized") or "").strip(),
                    entity_type=str(row.get("entity_type") or "").strip(),
                    entity_type_key=str(row.get("entity_type_key") or "").strip(),
                    strategy="token",
                    hit_count=_safe_int(row.get("hit_count"), 0),
                    edge_count=_safe_int(row.get("edge_count"), 0),
                )
            )
        return out

    async def _candidates_from_entity_faiss(
        self,
        *,
        adapter: Any,
        access_scope: Optional[GraphAccessScope],
        raw: str,
        normalized: str,
        entity_type_hint: str,
        limit: int,
    ) -> List[EntityResolutionCandidate]:
        # Best-effort: only works for HippoRAG adapters backed by PrunedHippoRAGNeo4jStore.
        retriever = getattr(adapter, "retriever", None)
        graph_store = getattr(retriever, "graph_store", None) if retriever is not None else None
        if graph_store is None:
            return []
        getter = getattr(graph_store, "get_entity_faiss_db", None)
        if not callable(getter):
            return []
        scope_id = access_scope.as_token() if access_scope is not None else None
        try:
            entity_db = getter(scope_id)
        except Exception:
            return []
        if entity_db is None or getattr(entity_db, "index", None) is None:
            return []

        embedder = getattr(graph_store, "embedding_model", None)
        if embedder is None or not hasattr(embedder, "embed"):
            return []

        try:
            embedding = embedder.embed([raw or normalized])[0]
        except Exception:
            return []

        # Use FAISS for recall, then re-rank by conservative string/edge scoring below.
        search_kwargs: Dict[str, Any] = {"k": int(self.faiss_top_k), "metric": "cosine"}
        if self.faiss_min_similarity is not None:
            search_kwargs["score_threshold"] = float(self.faiss_min_similarity)

        docs = RetrievalHelper.vector_search_with_faiss(entity_db, embedding, search_kwargs)
        entity_ids = [str(getattr(doc, "id", "") or "").strip() for doc, _ in docs if getattr(doc, "id", None)]
        if not entity_ids:
            return []

        entity_type = str(entity_type_hint or "").strip()
        cypher = """
        UNWIND $ids AS eid
        MATCH (e:Entity {entity_id: eid})
        WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
          AND ($entity_type = '' OR e.entity_type = $entity_type)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(:Entity)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
        RETURN e.entity_id AS entity_id,
               e.entity_name AS entity_name,
               e.entity_name_normalized AS entity_name_normalized,
               e.entity_type AS entity_type,
               e.entity_type_key AS entity_type_key,
               count(r) AS edge_count
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"ids": entity_ids, "entity_type": entity_type},
                access_scope=access_scope,
            )

        faiss_scores: Dict[str, float] = {}
        for doc, score in docs:
            doc_id = str(getattr(doc, "id", "") or "").strip()
            if doc_id:
                faiss_scores[doc_id] = float(score)

        out: List[EntityResolutionCandidate] = []
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            entity_id = str(row.get("entity_id") or "").strip()
            if not entity_id:
                continue
            out.append(
                EntityResolutionCandidate(
                    entity_id=entity_id,
                    entity_name=str(row.get("entity_name") or "").strip(),
                    entity_name_normalized=str(row.get("entity_name_normalized") or "").strip(),
                    entity_type=str(row.get("entity_type") or "").strip(),
                    entity_type_key=str(row.get("entity_type_key") or "").strip(),
                    strategy="faiss",
                    edge_count=_safe_int(row.get("edge_count"), 0),
                    faiss_score=faiss_scores.get(entity_id),
                )
            )
        return out[: max(1, int(limit))]

    async def _mention_count(
        self,
        *,
        adapter: Any,
        access_scope: Optional[GraphAccessScope],
        entity_id: str,
    ) -> int:
        eid = str(entity_id or "").strip()
        if not eid:
            return 0
        cypher = """
        MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity {entity_id: $entity_id})
        WHERE COALESCE(c.owner_id, $global_owner) = $owner_id
          AND COALESCE(m.owner_id, $global_owner) = $owner_id
        RETURN count(c) AS mentions
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, {"entity_id": eid}, access_scope=access_scope)
        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        if not isinstance(row0, Mapping):
            return 0
        return _safe_int(row0.get("mentions"), 0)

