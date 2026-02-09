"""KG long-term maintenance for PrunedHippoRAG Neo4j store.

This module implements L1 (background) maintenance:
- Disambiguate surface entities by clustering mention contexts (chunk embeddings).
- Create/update EntityIdentity nodes (cluster centers) with centroid embeddings.
- Assign mentions to identities via RESOLVED_TO edges (Graphiti-style reconciliation).
- Optionally align identities via SAME_AS edges using vector similarity + Gradient truncation.

Notes
-----
- This is deterministic-first; no LLM calls are performed here.
- All thresholds/budgets must come from config; when unspecified, we reuse existing graph knobs
  (e.g., synonymy_edge_sim_threshold) to avoid scattered magic numbers.
"""
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.knowledge_graph.maintenance.gradient_select import GradientSelect, truncate_indices
from core.knowledge_graph.maintenance.similarity import greedy_cluster_by_centroid, normalize_rows
from core.knowledge_graph.maintenance.time_window import TimeWindow
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


def _b64_f32(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return base64.b64encode(v.tobytes()).decode("ascii")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


@dataclass(frozen=True)
class _MentionRow:
    mention_id: str
    chunk_id: str
    surface_entity_id: str
    source_file_id: str
    source_version: str
    valid_from: str
    valid_to: str
    effective_date: str


class _PrunedHippoRAGNeo4jKGMaintenanceMixin:
    def _kg_owner_key(self, owner_id: object) -> str:
        return self._owner_key(str(owner_id))  # type: ignore[attr-defined]

    def _kg_maintenance_config(self):  # noqa: ANN001
        cfg = getattr(self, "config", None)
        return getattr(cfg, "kg_maintenance", None) if cfg is not None else None

    def _kg_l1_enabled(self) -> bool:
        cfg = self._kg_maintenance_config()
        return bool(getattr(cfg, "l1_enabled", False))

    def _kg_l1_min_cluster_similarity(self) -> float:
        cfg = self._kg_maintenance_config()
        override = getattr(cfg, "l1_disambiguation_min_cluster_similarity", None)
        if override is not None:
            value = _safe_float(override, None)
            if value is not None:
                return float(value)
        # Reuse existing synonymy threshold as the default similarity policy.
        return float(getattr(self, "synonymy_edge_sim_threshold", 0.0) or 0.0)  # type: ignore[attr-defined]

    def _kg_l1_min_mentions_to_split(self) -> int:
        cfg = self._kg_maintenance_config()
        return max(0, int(getattr(cfg, "l1_min_mentions_to_split", 0) or 0))

    def _kg_l1_alignment_params(self) -> Dict[str, Any]:
        cfg = self._kg_maintenance_config()
        topk_raw = getattr(cfg, "l1_alignment_nn_topk", None)
        if topk_raw is None:
            # Default to existing synonymy policy to avoid introducing new constants.
            topk = int(getattr(self, "synonymy_edge_topk", 0) or 0)  # type: ignore[attr-defined]
        else:
            topk = max(0, int(topk_raw or 0))
        min_score = getattr(cfg, "l1_alignment_min_score", None)
        if min_score is None:
            min_score_val = float(getattr(self, "synonymy_edge_sim_threshold", 0.0) or 0.0)  # type: ignore[attr-defined]
        else:
            min_score_val = float(_safe_float(min_score, 0.0) or 0.0)
        mink = max(0, int(getattr(cfg, "l1_alignment_gradient_mink", 0) or 0))
        g = float(getattr(cfg, "l1_alignment_gradient_g", 0.0) or 0.0)
        require_time_overlap = bool(getattr(cfg, "l1_require_time_overlap_for_merge", True))
        return {
            "topk": int(topk),
            "min_score": float(min_score_val),
            "gradient": GradientSelect(mink=mink, g=g),
            "use_gradient": bool(mink > 0 and g > 0),
            "require_time_overlap": bool(require_time_overlap),
        }

    def _kg_fetch_mentions_for_files(
        self,
        *,
        owner_key: str,
        file_ids: Sequence[str],
    ) -> list[_MentionRow]:
        ids = [str(x).strip() for x in (file_ids or []) if str(x or "").strip()]
        if not ids:
            return []
        query = """
        UNWIND $file_ids AS file_id
        MATCH (m:EntityMention {owner_id: $owner_id, source_file_id: file_id})
        RETURN m.mention_id AS mention_id,
               m.chunk_id AS chunk_id,
               m.surface_entity_id AS surface_entity_id,
               m.source_file_id AS source_file_id,
               coalesce(m.source_version, '') AS source_version,
               coalesce(m.valid_from, '') AS valid_from,
               coalesce(m.valid_to, '') AS valid_to,
               coalesce(m.effective_date, '') AS effective_date
        """
        rows = self._execute_query(query, {"file_ids": ids, "owner_id": owner_key})  # type: ignore[attr-defined]
        out: list[_MentionRow] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            mention_id = str(row.get("mention_id") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").strip()
            surface_entity_id = str(row.get("surface_entity_id") or "").strip()
            if not mention_id or not chunk_id or not surface_entity_id:
                continue
            out.append(
                _MentionRow(
                    mention_id=mention_id,
                    chunk_id=chunk_id,
                    surface_entity_id=surface_entity_id,
                    source_file_id=str(row.get("source_file_id") or "").strip(),
                    source_version=str(row.get("source_version") or "").strip(),
                    valid_from=str(row.get("valid_from") or "").strip(),
                    valid_to=str(row.get("valid_to") or "").strip(),
                    effective_date=str(row.get("effective_date") or "").strip(),
                )
            )
        return out

    def _kg_fetch_entity_type_keys(self, *, owner_key: str, entity_ids: Sequence[str]) -> Dict[str, str]:
        ids = [str(x).strip() for x in (entity_ids or []) if str(x or "").strip()]
        if not ids:
            return {}
        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:Entity {owner_id: $owner_id, entity_id: entity_id})
        RETURN e.entity_id AS entity_id, coalesce(e.entity_type_key, 'entity') AS entity_type_key
        """
        rows = self._execute_query(query, {"entity_ids": ids, "owner_id": owner_key})  # type: ignore[attr-defined]
        out: Dict[str, str] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            eid = str(row.get("entity_id") or "").strip()
            t = str(row.get("entity_type_key") or "").strip() or "entity"
            if eid:
                out[eid] = t
        return out

    def _kg_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        # Chunk embeddings are stored normalized (cosine). If missing, we skip for now.
        try:
            emb = getattr(self, "chunk_embeddings", {}).get(str(chunk_id))  # type: ignore[attr-defined]
        except Exception:
            emb = None
        if emb is None:
            return None
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return None
        return arr

    def _kg_l1_build_disambiguation_payloads(
        self,
        *,
        owner_key: str,
        by_surface: Dict[str, List[_MentionRow]],
        type_keys: Dict[str, str],
        min_sim: float,
        min_mentions_to_split: int,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[str],
        List[str],
        int,
    ]:
        """Build identity + assignment payloads deterministically (no DB writes).

        Returns:
            identities_payload, assignments_payload, mention_ids_in_scope, identity_ids, missing_embeddings_count
        """

        identities_payload: List[Dict[str, Any]] = []
        assignments_payload: List[Dict[str, Any]] = []
        mention_ids_in_scope: List[str] = []
        identity_ids: List[str] = []
        skipped_missing_embeddings = 0

        for surface_entity_id, rows in by_surface.items():
            # Collect embeddings aligned with rows.
            embs: List[np.ndarray] = []
            aligned_rows: List[_MentionRow] = []
            for r in rows:
                vec = self._kg_chunk_embedding(r.chunk_id)
                if vec is None:
                    skipped_missing_embeddings += 1
                    continue
                embs.append(vec)
                aligned_rows.append(r)
            if not aligned_rows:
                continue

            mention_ids_in_scope.extend([r.mention_id for r in aligned_rows])

            # Conservative: if we don't have enough mentions, keep single identity.
            if (min_mentions_to_split > 0) and (len(aligned_rows) < min_mentions_to_split):
                clusters = [list(range(len(aligned_rows)))]
            elif min_sim <= 0:
                clusters = [list(range(len(aligned_rows)))]
            else:
                clusters = [c.indices for c in greedy_cluster_by_centroid(np.stack(embs, axis=0), min_sim=min_sim)]

            # Build identity payloads + assignments.
            entity_type_key = type_keys.get(surface_entity_id) or "entity"
            # Normalize per-mention embeddings once (for confidence scoring).
            embs_norm = normalize_rows(np.stack(embs, axis=0))
            for cluster_indices in clusters:
                cluster_mentions = [aligned_rows[i] for i in cluster_indices]
                rep_mention_id = min(m.mention_id for m in cluster_mentions)
                identity_id = compute_mdhash_id(
                    f"{surface_entity_id}|{rep_mention_id}",
                    prefix="identity-",
                    owner_id=owner_key,
                )
                identity_ids.append(identity_id)

                cluster_embs = embs_norm[np.array(cluster_indices, dtype=np.int64)]
                centroid_sum = np.asarray(cluster_embs.sum(axis=0), dtype=np.float32)
                centroid = normalize_rows(centroid_sum)[0]

                # Identity time summary (best-effort; keep mention-level as source of truth).
                # Store as raw strings so Neo4j writes cannot fail due to date parsing.
                valid_from_min = ""
                valid_to_max = ""
                parsed_from: list[Tuple[str, Any]] = []
                parsed_to: list[Tuple[str, Any]] = []
                for m in cluster_mentions:
                    vf = m.valid_from or m.effective_date
                    vt = m.valid_to
                    tw = TimeWindow.from_strings(valid_from=vf or "", valid_to=vt or "")
                    if tw.valid_from is not None:
                        parsed_from.append((vf or "", tw.valid_from))
                    if tw.valid_to is not None:
                        parsed_to.append((vt or "", tw.valid_to))
                if parsed_from:
                    valid_from_min = min(parsed_from, key=lambda item: item[1])[0]
                if parsed_to:
                    valid_to_max = max(parsed_to, key=lambda item: item[1])[0]

                identities_payload.append(
                    {
                        "identity_id": identity_id,
                        "owner_id": owner_key,
                        "entity_type_key": entity_type_key,
                        "surface_entity_id": surface_entity_id,
                        "status": "active",
                        "centroid_dim": int(centroid.shape[0]),
                        "centroid_b64": _b64_f32(centroid),
                        "centroid_sum_b64": _b64_f32(centroid_sum),
                        "member_count": int(len(cluster_mentions)),
                        "valid_from_min": valid_from_min,
                        "valid_to_max": valid_to_max,
                        # Keep versions optional to avoid growing payload; can be added later via config.
                        "source_versions": "",
                    }
                )
                for i in cluster_indices:
                    m = aligned_rows[i]
                    confidence = float(np.dot(embs_norm[i], centroid))
                    assignments_payload.append(
                        {
                            "mention_id": m.mention_id,
                            "identity_id": identity_id,
                            "owner_id": owner_key,
                            "method": "embedding_greedy_cluster",
                            "confidence": confidence,
                        }
                    )

        return (
            identities_payload,
            assignments_payload,
            mention_ids_in_scope,
            identity_ids,
            int(skipped_missing_embeddings),
        )

    def run_kg_maintenance_l1_for_file_ids(
        self,
        *,
        owner_id: object,
        file_ids: Sequence[str],
        run_id: str | None = None,
    ) -> Dict[str, Any]:
        """Run L1 maintenance for a set of file_ids (owner-scoped)."""
        started = time.perf_counter()
        owner_key = self._kg_owner_key(owner_id)
        rid = str(run_id or f"kg-run-{uuid.uuid4().hex}")

        stats: Dict[str, Any] = {
            "success": False,
            "owner_id": owner_key,
            "file_ids": [str(x).strip() for x in (file_ids or []) if str(x or "").strip()],
            "run_id": rid,
            "counts": {},
            "timings_s": {},
        }

        if not self._kg_l1_enabled():
            stats["success"] = True
            stats["skipped"] = "l1_disabled"
            stats["timings_s"]["total"] = float(time.perf_counter() - started)
            return stats

        t0 = time.perf_counter()
        mention_rows = self._kg_fetch_mentions_for_files(owner_key=owner_key, file_ids=file_ids)
        stats["timings_s"]["fetch_mentions"] = float(time.perf_counter() - t0)
        stats["counts"]["mentions_total"] = int(len(mention_rows))

        if not mention_rows:
            stats["success"] = True
            stats["skipped"] = "no_mentions"
            stats["timings_s"]["total"] = float(time.perf_counter() - started)
            return stats

        # Group by surface entity.
        by_surface: Dict[str, List[_MentionRow]] = {}
        for row in mention_rows:
            by_surface.setdefault(row.surface_entity_id, []).append(row)
        stats["counts"]["surface_entities"] = int(len(by_surface))

        # Fetch type keys for surface entities.
        t0 = time.perf_counter()
        type_keys = self._kg_fetch_entity_type_keys(owner_key=owner_key, entity_ids=list(by_surface.keys()))
        stats["timings_s"]["fetch_entity_types"] = float(time.perf_counter() - t0)

        min_sim = float(self._kg_l1_min_cluster_similarity())
        min_mentions_to_split = int(self._kg_l1_min_mentions_to_split())

        t0 = time.perf_counter()
        (
            identities_payload,
            assignments_payload,
            mention_ids_in_scope,
            new_identity_ids,
            skipped_missing_embeddings,
        ) = self._kg_l1_build_disambiguation_payloads(
            owner_key=owner_key,
            by_surface=by_surface,
            type_keys=type_keys,
            min_sim=min_sim,
            min_mentions_to_split=min_mentions_to_split,
        )
        stats["timings_s"]["disambiguate_build_payloads"] = float(time.perf_counter() - t0)
        stats["counts"]["missing_chunk_embeddings"] = int(skipped_missing_embeddings)

        if not identities_payload or not assignments_payload:
            stats["success"] = True
            stats["skipped"] = "no_payloads"
            stats["timings_s"]["total"] = float(time.perf_counter() - started)
            return stats

        stats["counts"]["identities_upsert"] = int(len(identities_payload))
        stats["counts"]["assignments_upsert"] = int(len(assignments_payload))

        # Persist: identities + assignments (Graphiti-style supersede old edges, then write new).
        t0 = time.perf_counter()
        from encapsulation.database.graph_db.neo4j_entity_identity_cypher import (
            SUPERSEDE_RESOLVED_TO_FOR_MENTIONS_QUERY,
            UPSERT_ENTITY_IDENTITIES_QUERY,
            UPSERT_RESOLVED_TO_QUERY,
        )

        with self._driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            with session.begin_transaction() as tx:
                tx.run(UPSERT_ENTITY_IDENTITIES_QUERY, {"identities": identities_payload})
                tx.run(SUPERSEDE_RESOLVED_TO_FOR_MENTIONS_QUERY, {"mention_ids": mention_ids_in_scope, "run_id": rid})
                tx.run(UPSERT_RESOLVED_TO_QUERY, {"assignments": assignments_payload, "run_id": rid})
                tx.commit()

        stats["timings_s"]["persist_identities_assignments"] = float(time.perf_counter() - t0)

        # Alignment: SAME_AS edges between identities (optional).
        align_params = self._kg_l1_alignment_params()
        if align_params["topk"] > 0:
            t0 = time.perf_counter()
            align_stats = self._kg_align_identities_same_as(
                owner_key=owner_key,
                run_id=rid,
                focus_identity_ids=sorted(set(new_identity_ids)),
                topk=int(align_params["topk"]),
                min_score=float(align_params["min_score"]),
                gradient=align_params["gradient"],
                use_gradient=bool(align_params["use_gradient"]),
            )
            stats["timings_s"]["align_identities"] = float(time.perf_counter() - t0)
            stats["alignment"] = align_stats

        stats["success"] = True
        stats["timings_s"]["total"] = float(time.perf_counter() - started)
        return stats

    def run_kg_maintenance_l2_for_owner(
        self,
        *,
        owner_id: object,
        file_ids: Sequence[str] | None = None,
        run_id: str | None = None,
        rebuild_same_as: bool = False,
        backfill_missing_mentions: bool = False,
    ) -> Dict[str, Any]:
        """Run L2 repair for an owner (optionally scoped to file_ids).

        L2 goals
        - Repair drift after incremental uploads and deletions (stale identity aggregates, orphan nodes/edges).
        - Keep history (supersede edges instead of deleting).
        - Never cross owner boundaries (multi-tenant isolation).
        """

        started = time.perf_counter()
        owner_key = self._kg_owner_key(owner_id)
        rid = str(run_id or f"kg-l2-{uuid.uuid4().hex}")
        scope_file_ids = [str(x).strip() for x in (file_ids or []) if str(x or "").strip()]

        stats: Dict[str, Any] = {
            "success": False,
            "owner_id": owner_key,
            "file_ids": scope_file_ids,
            "run_id": rid,
            "counts": {},
            "timings_s": {},
        }

        # Optional: fill missing mention nodes for this scope before repair.
        if backfill_missing_mentions:
            t0 = time.perf_counter()
            chunk_query = """
            MATCH (c:Chunk {owner_id: $owner_id})
            WHERE $file_ids_is_empty OR c.source_file_id IN $file_ids
            RETURN c.chunk_id AS chunk_id
            """
            rows = self._execute_query(  # type: ignore[attr-defined]
                chunk_query,
                {"owner_id": owner_key, "file_ids": scope_file_ids, "file_ids_is_empty": bool(not scope_file_ids)},
            )
            chunk_ids = [str(r.get("chunk_id") or "").strip() for r in (rows or []) if isinstance(r, dict)]
            chunk_ids = [x for x in chunk_ids if x]
            stats["counts"]["backfill_chunk_ids"] = int(len(chunk_ids))
            if chunk_ids:
                self.materialize_entity_mentions_for_chunk_ids(owner_id=owner_key, chunk_ids=chunk_ids)  # type: ignore[attr-defined]
            stats["timings_s"]["backfill_missing_mentions"] = float(time.perf_counter() - t0)

        # Fetch active RESOLVED_TO edges in scope.
        t0 = time.perf_counter()
        query = """
        MATCH (m:EntityMention {owner_id: $owner_id})-[r:RESOLVED_TO]->(i:EntityIdentity {owner_id: $owner_id})
        WHERE r.valid_to IS NULL
          AND ($file_ids_is_empty OR m.source_file_id IN $file_ids)
        RETURN m.mention_id AS mention_id,
               m.chunk_id AS chunk_id,
               m.surface_entity_id AS surface_entity_id,
               coalesce(m.valid_from, '') AS valid_from,
               coalesce(m.valid_to, '') AS valid_to,
               coalesce(m.effective_date, '') AS effective_date,
               i.identity_id AS identity_id,
               coalesce(r.confidence, 0.0) AS confidence,
               r.valid_from AS edge_valid_from
        """
        rows = self._execute_query(  # type: ignore[attr-defined]
            query,
            {"owner_id": owner_key, "file_ids": scope_file_ids, "file_ids_is_empty": bool(not scope_file_ids)},
        )
        stats["timings_s"]["fetch_active_resolved_to"] = float(time.perf_counter() - t0)

        if not rows:
            # Still do orphan identity cleanup for global runs.
            from encapsulation.database.graph_db.neo4j_entity_identity_cypher import (
                MARK_ORPHAN_IDENTITIES_INACTIVE_QUERY,
                ZERO_ORPHAN_IDENTITIES_MEMBER_COUNT_QUERY,
                SUPERSEDE_SAME_AS_FOR_INACTIVE_IDENTITIES_QUERY,
            )

            t0 = time.perf_counter()
            marked = 0
            zeroed = 0
            superseded = 0
            with self._driver.session(database=self.database) as session:  # type: ignore[attr-defined]
                with session.begin_transaction() as tx:
                    rec = list(tx.run(MARK_ORPHAN_IDENTITIES_INACTIVE_QUERY, {"owner_id": owner_key}))
                    if rec:
                        try:
                            marked = int(rec[0].get("marked") or 0)
                        except Exception:
                            marked = 0
                    rec0 = list(tx.run(ZERO_ORPHAN_IDENTITIES_MEMBER_COUNT_QUERY, {"owner_id": owner_key}))
                    if rec0:
                        try:
                            zeroed = int(rec0[0].get("zeroed") or 0)
                        except Exception:
                            zeroed = 0
                    rec2 = list(tx.run(SUPERSEDE_SAME_AS_FOR_INACTIVE_IDENTITIES_QUERY, {"owner_id": owner_key, "run_id": rid}))
                    if rec2:
                        try:
                            superseded = int(rec2[0].get("superseded") or 0)
                        except Exception:
                            superseded = 0
                    tx.commit()
            stats["timings_s"]["cleanup_orphans_only"] = float(time.perf_counter() - t0)
            stats["counts"]["identities_marked_inactive"] = int(marked)
            stats["counts"]["identities_orphans_zeroed"] = int(zeroed)
            stats["counts"]["same_as_superseded_inactive"] = int(superseded)
            stats["success"] = True
            stats["timings_s"]["total"] = float(time.perf_counter() - started)
            return stats

        # Group by mention_id to detect multiple active assignments.
        by_mention: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            mid = str(row.get("mention_id") or "").strip()
            if not mid:
                continue
            by_mention.setdefault(mid, []).append(row)
        stats["counts"]["mentions_with_active_assignments"] = int(len(by_mention))

        # Supersede duplicate assignments (keep the best per mention).
        supersede_payload: List[Dict[str, Any]] = []
        kept_rows: List[Dict[str, Any]] = []
        for mid, items in by_mention.items():
            if len(items) == 1:
                kept_rows.append(items[0])
                continue
            # Prefer higher confidence; tie-break by more recent edge_valid_from when available; else stable by identity_id.
            def _key(it: Dict[str, Any]) -> Tuple[float, float, str]:
                conf = _safe_float(it.get("confidence"), 0.0) or 0.0
                evf = it.get("edge_valid_from")
                ts = 0.0
                try:
                    if evf is not None:
                        # Neo4j datetime -> string-ish; compare lexicographically is unsafe; keep 0.
                        ts = 0.0
                except Exception:
                    ts = 0.0
                return (float(conf), float(ts), str(it.get("identity_id") or ""))

            items_sorted = sorted(items, key=_key, reverse=True)
            kept = items_sorted[0]
            kept_rows.append(kept)
            for drop in items_sorted[1:]:
                supersede_payload.append(
                    {
                        "mention_id": str(drop.get("mention_id") or "").strip(),
                        "identity_id": str(drop.get("identity_id") or "").strip(),
                        "owner_id": owner_key,
                    }
                )

        if supersede_payload:
            from encapsulation.database.graph_db.neo4j_entity_identity_cypher import SUPERSEDE_RESOLVED_TO_FOR_ASSIGNMENTS_QUERY

            t0 = time.perf_counter()
            with self._driver.session(database=self.database) as session:  # type: ignore[attr-defined]
                with session.begin_transaction() as tx:
                    tx.run(SUPERSEDE_RESOLVED_TO_FOR_ASSIGNMENTS_QUERY, {"assignments": supersede_payload, "run_id": rid})
                    tx.commit()
            stats["timings_s"]["supersede_duplicate_assignments"] = float(time.perf_counter() - t0)
        stats["counts"]["resolved_to_superseded"] = int(len(supersede_payload))

        # Recompute identity aggregates from kept active assignments.
        by_identity: Dict[str, List[Dict[str, Any]]] = {}
        for row in kept_rows:
            iid = str(row.get("identity_id") or "").strip()
            if not iid:
                continue
            by_identity.setdefault(iid, []).append(row)
        stats["counts"]["identities_touched"] = int(len(by_identity))

        updates: List[Dict[str, Any]] = []
        missing_embeddings = 0
        for identity_id, items in by_identity.items():
            embs: List[np.ndarray] = []
            # time window summary
            parsed_from: list[Tuple[str, Any]] = []
            parsed_to: list[Tuple[str, Any]] = []
            for it in items:
                chunk_id = str(it.get("chunk_id") or "").strip()
                vec = self._kg_chunk_embedding(chunk_id)
                if vec is None:
                    missing_embeddings += 1
                else:
                    embs.append(vec)

                vf = str(it.get("valid_from") or "").strip() or str(it.get("effective_date") or "").strip()
                vt = str(it.get("valid_to") or "").strip()
                tw = TimeWindow.from_strings(valid_from=vf or "", valid_to=vt or "")
                if tw.valid_from is not None:
                    parsed_from.append((vf or "", tw.valid_from))
                if tw.valid_to is not None:
                    parsed_to.append((vt or "", tw.valid_to))

            member_count = int(len(items))
            valid_from_min = min(parsed_from, key=lambda t: t[1])[0] if parsed_from else ""
            valid_to_max = max(parsed_to, key=lambda t: t[1])[0] if parsed_to else ""

            if embs:
                mat = normalize_rows(np.stack(embs, axis=0))
                centroid_sum = np.asarray(mat.sum(axis=0), dtype=np.float32)
                centroid = normalize_rows(centroid_sum)[0]
                updates.append(
                    {
                        "identity_id": identity_id,
                        "owner_id": owner_key,
                        "centroid_dim": int(centroid.shape[0]),
                        "centroid_b64": _b64_f32(centroid),
                        "centroid_sum_b64": _b64_f32(centroid_sum),
                        "member_count": member_count,
                        "valid_from_min": valid_from_min,
                        "valid_to_max": valid_to_max,
                    }
                )
            else:
                # No embeddings available: still repair counts/time; keep centroid untouched.
                updates.append(
                    {
                        "identity_id": identity_id,
                        "owner_id": owner_key,
                        "centroid_dim": None,
                        "centroid_b64": None,
                        "centroid_sum_b64": None,
                        "member_count": member_count,
                        "valid_from_min": valid_from_min,
                        "valid_to_max": valid_to_max,
                    }
                )

        stats["counts"]["missing_chunk_embeddings"] = int(missing_embeddings)

        if updates:
            from encapsulation.database.graph_db.neo4j_entity_identity_cypher import UPDATE_ENTITY_IDENTITIES_AGGREGATES_QUERY

            # Split into two batches: with centroid and without (Cypher SET cannot set null dim to int reliably).
            with_centroid = [u for u in updates if u.get("centroid_dim") is not None]
            without_centroid = [u for u in updates if u.get("centroid_dim") is None]

            t0 = time.perf_counter()
            with self._driver.session(database=self.database) as session:  # type: ignore[attr-defined]
                with session.begin_transaction() as tx:
                    if with_centroid:
                        tx.run(UPDATE_ENTITY_IDENTITIES_AGGREGATES_QUERY, {"updates": with_centroid})
                    if without_centroid:
                        # Update only counts/time.
                        q2 = """
                        UNWIND $updates AS u
                        MATCH (i:EntityIdentity {identity_id: u.identity_id, owner_id: u.owner_id})
                        SET i.member_count = u.member_count,
                            i.valid_from_min = u.valid_from_min,
                            i.valid_to_max = u.valid_to_max,
                            i.updated_at = datetime()
                        """
                        tx.run(q2, {"updates": without_centroid})
                    tx.commit()
            stats["timings_s"]["update_identity_aggregates"] = float(time.perf_counter() - t0)
        stats["counts"]["identities_updated"] = int(len(updates))

        # Mark orphan identities inactive + supersede SAME_AS edges to inactive identities.
        from encapsulation.database.graph_db.neo4j_entity_identity_cypher import (
            MARK_ORPHAN_IDENTITIES_INACTIVE_QUERY,
            ZERO_ORPHAN_IDENTITIES_MEMBER_COUNT_QUERY,
            SUPERSEDE_SAME_AS_FOR_INACTIVE_IDENTITIES_QUERY,
            SUPERSEDE_ALL_SAME_AS_FOR_OWNER_QUERY,
        )

        t0 = time.perf_counter()
        marked = 0
        zeroed = 0
        superseded_inactive = 0
        superseded_all = 0
        with self._driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            with session.begin_transaction() as tx:
                rec = list(tx.run(MARK_ORPHAN_IDENTITIES_INACTIVE_QUERY, {"owner_id": owner_key}))
                if rec:
                    try:
                        marked = int(rec[0].get("marked") or 0)
                    except Exception:
                        marked = 0
                rec0 = list(tx.run(ZERO_ORPHAN_IDENTITIES_MEMBER_COUNT_QUERY, {"owner_id": owner_key}))
                if rec0:
                    try:
                        zeroed = int(rec0[0].get("zeroed") or 0)
                    except Exception:
                        zeroed = 0
                rec2 = list(tx.run(SUPERSEDE_SAME_AS_FOR_INACTIVE_IDENTITIES_QUERY, {"owner_id": owner_key, "run_id": rid}))
                if rec2:
                    try:
                        superseded_inactive = int(rec2[0].get("superseded") or 0)
                    except Exception:
                        superseded_inactive = 0
                if rebuild_same_as:
                    rec3 = list(tx.run(SUPERSEDE_ALL_SAME_AS_FOR_OWNER_QUERY, {"owner_id": owner_key, "run_id": rid}))
                    if rec3:
                        try:
                            superseded_all = int(rec3[0].get("superseded") or 0)
                        except Exception:
                            superseded_all = 0
                tx.commit()
        stats["timings_s"]["cleanup_orphans_and_same_as"] = float(time.perf_counter() - t0)
        stats["counts"]["identities_marked_inactive"] = int(marked)
        stats["counts"]["identities_orphans_zeroed"] = int(zeroed)
        stats["counts"]["same_as_superseded_inactive"] = int(superseded_inactive)
        stats["counts"]["same_as_superseded_all"] = int(superseded_all)

        # Rebuild SAME_AS edges (optional).
        if rebuild_same_as:
            t0 = time.perf_counter()
            align_params = self._kg_l1_alignment_params()
            topk = int(align_params["topk"])
            if topk > 0:
                # Focus all active identities for this owner.
                active_ids_rows = self._execute_query(  # type: ignore[attr-defined]
                    "MATCH (i:EntityIdentity {owner_id: $owner_id}) WHERE coalesce(i.status,'active')='active' RETURN i.identity_id AS identity_id",
                    {"owner_id": owner_key},
                )
                active_ids = [str(r.get("identity_id") or "").strip() for r in (active_ids_rows or []) if isinstance(r, dict)]
                active_ids = [x for x in active_ids if x]
                align_stats = self._kg_align_identities_same_as(
                    owner_key=owner_key,
                    run_id=rid,
                    focus_identity_ids=sorted(set(active_ids)),
                    topk=topk,
                    min_score=float(align_params["min_score"]),
                    gradient=align_params["gradient"],
                    use_gradient=bool(align_params["use_gradient"]),
                )
                stats["alignment"] = align_stats
            stats["timings_s"]["rebuild_same_as"] = float(time.perf_counter() - t0)

        stats["success"] = True
        stats["timings_s"]["total"] = float(time.perf_counter() - started)
        return stats

    def _kg_align_identities_same_as(
        self,
        *,
        owner_key: str,
        run_id: str,
        focus_identity_ids: Sequence[str],
        topk: int,
        min_score: float,
        gradient: GradientSelect,
        use_gradient: bool,
    ) -> Dict[str, Any]:
        """
        Align identities via SAME_AS edges using centroid cosine similarity.

        Implementation notes:
        - Builds an in-memory FAISS IP index over identity centroids for this owner.
        - Only emits SAME_AS edges; strong merges are intentionally not performed here.
        """
        focus = [str(x).strip() for x in (focus_identity_ids or []) if str(x or "").strip()]
        if not focus:
            return {"attempted": 0, "written": 0, "reason": "no_focus"}
        if topk <= 0:
            return {"attempted": 0, "written": 0, "reason": "disabled"}

        query = """
        MATCH (i:EntityIdentity {owner_id: $owner_id})
        WHERE coalesce(i.status, 'active') = 'active'
          AND i.centroid_b64 IS NOT NULL
          AND i.centroid_dim IS NOT NULL
        RETURN i.identity_id AS identity_id,
               i.surface_entity_id AS surface_entity_id,
               i.centroid_dim AS centroid_dim,
               i.centroid_b64 AS centroid_b64,
               coalesce(i.valid_from_min, '') AS valid_from_min,
               coalesce(i.valid_to_max, '') AS valid_to_max
        """
        rows = self._execute_query(query, {"owner_id": owner_key})  # type: ignore[attr-defined]
        if not rows:
            return {"attempted": 0, "written": 0, "reason": "no_identities"}

        # Decode vectors.
        ids: list[str] = []
        surface_by_id: Dict[str, str] = {}
        time_by_id: Dict[str, TimeWindow] = {}
        vecs: list[np.ndarray] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            iid = str(row.get("identity_id") or "").strip()
            if not iid:
                continue
            dim = _safe_int(row.get("centroid_dim"), None)
            b64 = str(row.get("centroid_b64") or "").strip()
            if not b64 or not dim:
                continue
            try:
                raw = base64.b64decode(b64.encode("ascii"))
                v = np.frombuffer(raw, dtype=np.float32)
                if int(v.shape[0]) != int(dim):
                    continue
            except Exception:
                continue
            ids.append(iid)
            surface_by_id[iid] = str(row.get("surface_entity_id") or "").strip()
            time_by_id[iid] = TimeWindow.from_strings(
                valid_from=str(row.get("valid_from_min") or "").strip(),
                valid_to=str(row.get("valid_to_max") or "").strip(),
            )
            vecs.append(v)

        if not ids:
            return {"attempted": 0, "written": 0, "reason": "no_vectors"}

        mat = normalize_rows(np.stack(vecs, axis=0))

        # Build FAISS IP index.
        import faiss

        index = faiss.IndexFlatIP(mat.shape[1])
        from core.utils.faiss_lock import FAISS_LOCK

        with FAISS_LOCK:
            index.add(mat.astype(np.float32))

        # Map identity_id -> row index.
        idx_by_id = {iid: i for i, iid in enumerate(ids)}
        focus_indices = [idx_by_id[i] for i in focus if i in idx_by_id]
        if not focus_indices:
            return {"attempted": 0, "written": 0, "reason": "focus_not_found"}

        # Candidate selection per focus identity.
        pairs: List[Dict[str, Any]] = []
        attempted = 0
        for fi in focus_indices:
            q = mat[fi : fi + 1]
            k = min(int(topk) + 1, int(index.ntotal))
            with FAISS_LOCK:
                scores, nbrs = index.search(q.astype(np.float32), k)
            cand_scores = [float(s) for s in scores.reshape(-1).tolist()]
            cand_idxs = [int(x) for x in nbrs.reshape(-1).tolist()]
            # Remove self.
            filtered: list[Tuple[int, float]] = []
            for ci, sc in zip(cand_idxs, cand_scores):
                if ci < 0 or ci == fi:
                    continue
                filtered.append((ci, float(sc)))
            if not filtered:
                continue
            attempted += 1

            # Sort by score desc.
            filtered.sort(key=lambda t: t[1], reverse=True)
            idxs = list(range(len(filtered)))
            scores_desc = [s for _i, s in filtered]
            if use_gradient:
                keep = truncate_indices(scores_desc, policy=gradient)
                keep_set = set(keep)
                idxs = [i for i in idxs if i in keep_set]
            for i in idxs:
                ci, sc = filtered[i]
                if sc < float(min_score):
                    continue
                left = ids[min(fi, ci)]
                right = ids[max(fi, ci)]
                if not left or not right or left == right:
                    continue
                # Avoid aligning identities that came from the same surface entity; those are disambiguation siblings.
                if surface_by_id.get(left) and surface_by_id.get(left) == surface_by_id.get(right):
                    continue
                tw_left = time_by_id.get(left) or TimeWindow.from_strings()
                tw_right = time_by_id.get(right) or TimeWindow.from_strings()
                overlap = tw_left.overlaps(tw_right)
                reason = "embedding_identity_similarity"
                if overlap is False:
                    reason = "embedding_identity_similarity_no_time_overlap"
                pairs.append(
                    {
                        "left_id": left,
                        "right_id": right,
                        "owner_id": owner_key,
                        "score": float(sc),
                        "method": "faiss_identity_centroid",
                        "time_overlap": None if overlap is None else bool(overlap),
                        "reason": reason,
                    }
                )

        if not pairs:
            return {"attempted": int(attempted), "written": 0, "reason": "no_pairs"}

        # Dedupe (left,right).
        dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for p in pairs:
            k = (p["left_id"], p["right_id"])
            best = dedup.get(k)
            if best is None or float(p.get("score") or 0.0) > float(best.get("score") or 0.0):
                dedup[k] = p
        payload = list(dedup.values())

        from encapsulation.database.graph_db.neo4j_entity_identity_cypher import UPSERT_SAME_AS_QUERY

        with self._driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            with session.begin_transaction() as tx:
                tx.run(UPSERT_SAME_AS_QUERY, {"pairs": payload, "run_id": run_id})
                tx.commit()

        return {"attempted": int(attempted), "written": int(len(payload))}
