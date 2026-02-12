"""Chunk+fact ingestion routines for PrunedHippoRAG Neo4j store (kept under 1000 LOC)."""

import os
import json
import logging
from typing import Any, Dict, List, Optional

from core.file_management.extractor.metadata_keys import BUSINESS_TIME_KEY, EXTRACTION_ERROR_KEY, SDF_KEY
from encapsulation.data_model.schema import Chunk
from encapsulation.database.graph_db.pruned_hipporag_neo4j_chunk_embeddings import _PrunedHippoRAGNeo4jChunkEmbeddingsMixin
from encapsulation.database.utils.fact_provenance import upsert_fact_occurrence
from encapsulation.database.utils.sdf_schema_payload import build_sdf_schema_payload
from encapsulation.database.utils.schema_layer_nodes import build_schema_layer_payload
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from core.knowledge_graph.schema import normalize_relation_token
from encapsulation.database.graph_db.pruned_hipporag_neo4j_chunk_upsert_cleanup import run_chunk_replace_cleanup
from config import pageindex as pageindex_cfg
from config.core.deepsearch import tool_defaults

logger = logging.getLogger(__name__)


def _coerce_fact_provenance_max_source_chunks(raw: Any, *, default: int = 50, max_value: int = 1000) -> int:
    """
    Coerce `fact_provenance_max_source_chunks` into a safe integer.

    Note:
    - `0` is a valid value meaning "disable source_chunk_ids storage", so do NOT use `or default`.
    """
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = int(default)
    return max(0, min(int(max_value), value))


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_node_type(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    mapping = getattr(tool_defaults, "SECTION_NODE_TYPE_MAP", {}) or {}
    if token and token in mapping:
        return str(mapping[token])
    default = getattr(tool_defaults, "SECTION_NODE_TYPE_DEFAULT", "page")
    return str(default)


def _coerce_str(value: Any) -> str:
    return str(value or "").strip()


def _trim_summary(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _tree_node_summary(metadata: Dict[str, Any], content: str) -> Optional[str]:
    summary = ""
    index_text = metadata.get("index_text")
    if isinstance(index_text, str) and index_text.strip():
        summary = index_text.strip()
    if not summary:
        caption = metadata.get("table_caption")
        if isinstance(caption, str) and caption.strip():
            summary = caption.strip()
    if not summary:
        alts = metadata.get("image_alts")
        if isinstance(alts, list):
            cleaned = [str(a).strip() for a in alts if str(a or "").strip()]
            if cleaned:
                summary = "; ".join(cleaned)
    if not summary:
        summary = str(content or "").strip()
    if not summary:
        return None
    max_chars = int(getattr(tool_defaults, "SECTION_SEARCH_SUMMARY_PREVIEW_CHARS", 160))
    max_chars = max(1, max_chars)
    return _trim_summary(summary, max_chars=max_chars)


def _select_domain_for_chunk(chunk: Chunk, metadata: Dict[str, Any], schema: Any) -> str:
    """
    Select a chunk domain for KG schema governance.

    Policy:
    - Prefer explicit chunk.domain (ingestion-provided)
    - Fallback to chunk.metadata["domain"] (optional upstream classifier)
    - Finally: use KG schema default_domain when available, otherwise "default"
    """
    schema_default = getattr(schema, "default_domain", None) if schema is not None else None
    fallback = str(schema_default).strip() if schema_default is not None else ""
    if not fallback:
        fallback = "default"

    domain_raw = chunk.domain or metadata.get("domain") or fallback
    domain = str(domain_raw).strip()
    return domain or fallback

class _PrunedHippoRAGNeo4jIndexingIngestMixin(_PrunedHippoRAGNeo4jChunkEmbeddingsMixin):
    def _init_faiss_indices(self):
        """
        Initialize FAISS indices for facts and entities.

        - Fact index: FAISS Flat (exact search) for fact retrieval
        - Entity index: FAISS HNSW (approximate search) for synonymy edge computation
        """
        from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig

        storage_path = self.config.storage_path
        index_name = self.config.index_name
        os.makedirs(storage_path, exist_ok=True)

        # Initialize fact index (FAISS Flat for exact search)
        fact_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='flat',
            metric='cosine',
            normalize_L2=True,
            index_path=os.path.join(storage_path, 'fact_index'),
            index_name=index_name,
        )
        self.fact_faiss_db = fact_config.build()

        # Load existing fact index if available
        fact_index_path = os.path.join(storage_path, 'fact_index')
        if os.path.exists(fact_index_path):
            try:
                self.fact_faiss_db.load_index(fact_index_path)
                logger.info(f"Loaded existing fact index: {self.fact_faiss_db.index.ntotal} facts")
            except Exception as e:
                logger.warning(f"Failed to load fact index: {e}")

        # Initialize entity index (FAISS HNSW for approximate search)
        entity_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='hnsw',
            metric='cosine',
            normalize_L2=True,
            m=self.config.hnsw_M,
            efConstruction=self.config.hnsw_ef_construction,
            efSearch=self.config.hnsw_ef_search,
            index_path=os.path.join(storage_path, 'entity_index'),
            index_name=index_name,
        )
        self.entity_faiss_db = entity_config.build()

        # Load existing entity index if available
        entity_index_path = os.path.join(storage_path, 'entity_index')
        if os.path.exists(entity_index_path):
            try:
                self.entity_faiss_db.load_index(entity_index_path)
                logger.info(f"Loaded existing entity index: {self.entity_faiss_db.index.ntotal} entities")
            except Exception as e:
                logger.warning(f"Failed to load entity index: {e}")

        logger.info("FAISS indices initialized (fact: Flat, entity: HNSW)")

    def _batch_add_chunks_and_graph_data(
        self,
        chunks: List[Chunk],
        *,
        enable_entity_mentions: bool | None = None,
        entity_mentions_remaining: int | None = None,
    ) -> tuple[List[str], Dict[str, Any]]:
        """
        Batch add chunks and their graph data to Neo4j (OPTIMIZED).

        This method collects all data from chunks and performs batch insertions
        using UNWIND, which is much faster than individual queries.

        Args:
            chunks: List of Chunk objects to add

        Returns:
            (new_entity_ids, stats) where stats includes optional L0 mention materialization metrics.
        """
        import time
        start_time = time.time()

        # Collect all data
        chunk_data = []
        section_data_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
        section_chunk_links: List[Dict[str, Any]] = []
        tree_node_data_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
        tree_node_section_links: List[Dict[str, Any]] = []
        tree_node_chunk_links: List[Dict[str, Any]] = []
        tree_node_parent_links: List[Dict[str, Any]] = []
        tree_node_section_link_keys: set[tuple[str, str, str]] = set()
        tree_node_chunk_link_keys: set[tuple[str, str, str]] = set()
        tree_node_parent_link_keys: set[tuple[str, str, str]] = set()
        entity_data: Dict[str, Dict[str, Any]] = {}  # entity_id -> entity payload
        mention_data = []
        entity_mention_data: List[Dict[str, Any]] = []
        fact_data_by_id: Dict[str, Dict[str, Any]] = {}
        schema_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        schema_links: List[Dict[str, Any]] = []
        sdf_event_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        sdf_has_subevent_edges: List[Dict[str, Any]] = []
        sdf_before_edges: List[Dict[str, Any]] = []
        sdf_chunk_event_links: List[Dict[str, Any]] = []
        new_entity_ids: List[str] = []

        l0_stats: Dict[str, Any] = {
            "entity_mentions": {
                "enabled": False,
                "attempted": 0,
                "written": 0,
                "skipped_reason": None,
                "elapsed_s": 0.0,
            }
        }

        # HippoRAG chunk-triples contract: keep only triples whose endpoints are extracted named entities.
        # (Precision-first; endpoints outside extracted entities are dropped.)

        stats_by_owner: Dict[str, Dict[str, int]] = {}
        cfg = getattr(self, "config", None)
        max_source_chunks = _coerce_fact_provenance_max_source_chunks(
            getattr(cfg, "fact_provenance_max_source_chunks", None),
            default=50,
            max_value=1000,
        )
        enable_schema_layers = bool(getattr(cfg, "enable_schema_layer_nodes", False))
        raw_schema_max_nodes = getattr(cfg, "schema_layer_max_nodes_per_chunk", None)

        # L0 mention materialization config (hot-path, must be optional and budgeted by caller).
        l0_cfg = getattr(getattr(cfg, "kg_maintenance", None), "l0", None) if cfg is not None else None
        l0_enabled_cfg = bool(getattr(l0_cfg, "enabled", False)) if l0_cfg is not None else False
        if enable_entity_mentions is None:
            enable_entity_mentions = l0_enabled_cfg
        enable_entity_mentions = bool(enable_entity_mentions)
        l0_stats["entity_mentions"]["enabled"] = bool(enable_entity_mentions)
        if enable_entity_mentions and entity_mentions_remaining is not None and int(entity_mentions_remaining) <= 0:
            enable_entity_mentions = False
            l0_stats["entity_mentions"]["enabled"] = False
            l0_stats["entity_mentions"]["skipped_reason"] = "quota_exhausted"

        # Source-version extraction keys for mention evidence.
        version_keys = list(getattr(l0_cfg, "source_version_metadata_keys", []) or []) if l0_cfg is not None else []

        def _extract_source_version(meta: Dict[str, Any]) -> str:
            for key in version_keys:
                val = str((meta or {}).get(key) or "").strip()
                if val:
                    return val
            return ""

        def _extract_business_time(meta: Dict[str, Any], *, chunk_obj: Chunk) -> Dict[str, Any]:
            business_time: Dict[str, Any] = {}
            try:
                if chunk_obj.graph and isinstance(getattr(chunk_obj.graph, "metadata", None), dict):
                    raw = chunk_obj.graph.metadata.get(BUSINESS_TIME_KEY)
                    if isinstance(raw, dict):
                        business_time = dict(raw)
            except Exception:
                business_time = {}
            if not business_time and isinstance((meta or {}).get(BUSINESS_TIME_KEY), dict):
                business_time = dict((meta or {}).get(BUSINESS_TIME_KEY) or {})
            return business_time
        schema_layer_max_nodes = int(raw_schema_max_nodes) if raw_schema_max_nodes is not None else 0
        enable_sdf_schema = bool(getattr(cfg, "enable_sdf_schema", False))
        sdf_max_events = int(getattr(cfg, "sdf_max_events_per_chunk", 0) or 0)
        sdf_max_relations = int(getattr(cfg, "sdf_max_relations_per_chunk", 0) or 0)
        sdf_max_source_chunks = _coerce_fact_provenance_max_source_chunks(
            getattr(cfg, "sdf_provenance_max_source_chunks", None),
            default=50,
            max_value=1000,
        )

        schema = getattr(self, "kg_schema", None)

        for chunk in chunks:
            # Prepare chunk data
            metadata = dict(chunk.metadata) if chunk.metadata else {}

            owner_source = chunk.owner_id or metadata.get('owner_id')
            owner_str = self._normalize_owner_id(owner_source)
            if owner_str:
                metadata['owner_id'] = owner_str
            db_owner_id = self._owner_key(owner_str)
            owner_stats = stats_by_owner.setdefault(
                db_owner_id,
                {
                    "chunks_total": 0,
                    "chunks_graph_empty": 0,
                    "chunks_extraction_failed": 0,
                    "triples_total": 0,
                    "triples_kept": 0,
                    "triples_kept_via_canonical_endpoints": 0,
                    "triples_dropped_endpoints": 0,
                    "triples_dropped_ambiguous_endpoints": 0,
                    "triples_dropped_canonical_ambiguous_endpoints": 0,
                    "triples_dropped_schema": 0,
                    "predicates_aliased": 0,
                    "predicates_kept": 0,
                    "predicates_collapsed": 0,
                    "predicates_rejected": 0,
                    "predicates_allowlist_rejected": 0,
                    "triples_kept_direction_insensitive": 0,
                },
            )
            owner_stats["chunks_total"] += 1

            # Extract source_file_id from metadata for independent storage
            source_file_id = metadata.get("source_file_id")
            page_start = _coerce_int(metadata.get("page_start"))
            page_end = _coerce_int(metadata.get("page_end"))

            # Persist business_time into Chunk.metadata so background maintenance can recover temporal fields
            # even when extractor-only fields are not available.
            try:
                if chunk.graph and isinstance(getattr(chunk.graph, "metadata", None), dict):
                    raw_bt = chunk.graph.metadata.get(BUSINESS_TIME_KEY)
                    if isinstance(raw_bt, dict) and BUSINESS_TIME_KEY not in metadata:
                        metadata[BUSINESS_TIME_KEY] = dict(raw_bt)
            except Exception:
                pass
            
            chunk_id = str(chunk.id or "").strip()
            if not chunk_id:
                raise ValueError("Chunk.id is required for Neo4j ingest (got empty/None).")

            chunk_data.append({
                'chunk_id': chunk_id,
                'content': chunk.content,
                'metadata': json.dumps(metadata) if metadata else '{}',
                'owner_id': db_owner_id,
                'source_file_id': source_file_id,  # Store as independent property for filtering
                'page_start': page_start,
                'page_end': page_end,
            })

            section_id = str(metadata.get("section_id") or "").strip()
            if section_id and source_file_id:
                section_key = (db_owner_id, section_id)
                section_path = str(metadata.get("section_path") or "").strip()
                section_title = str(metadata.get("section_title") or "").strip()
                if not section_title and section_path:
                    delimiter = pageindex_cfg.section_path_delimiter()
                    section_title = section_path.split(delimiter)[-1].strip() if delimiter in section_path else section_path
                section_level = _coerce_int(metadata.get("section_level"))
                section_parent_id = str(metadata.get("section_parent_id") or "").strip() or None
                page_start = _coerce_int(metadata.get("section_page_start"))
                if page_start is None:
                    page_start = _coerce_int(metadata.get("page_start"))
                page_end = _coerce_int(metadata.get("section_page_end"))
                if page_end is None:
                    page_end = _coerce_int(metadata.get("page_end"))

                existing = section_data_by_key.get(section_key)
                if existing is None:
                    section_data_by_key[section_key] = {
                        "section_id": section_id,
                        "owner_id": db_owner_id,
                        "source_file_id": source_file_id,
                        "section_path": section_path or None,
                        "section_title": section_title or None,
                        "section_level": section_level,
                        "section_parent_id": section_parent_id,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                else:
                    if not existing.get("section_path") and section_path:
                        existing["section_path"] = section_path
                    if not existing.get("section_title") and section_title:
                        existing["section_title"] = section_title
                    if existing.get("section_level") is None and section_level is not None:
                        existing["section_level"] = section_level
                    if not existing.get("section_parent_id") and section_parent_id:
                        existing["section_parent_id"] = section_parent_id
                    if page_start is not None:
                        existing_start = existing.get("page_start")
                        existing["page_start"] = page_start if existing_start is None else min(int(existing_start), page_start)
                    if page_end is not None:
                        existing_end = existing.get("page_end")
                        existing["page_end"] = page_end if existing_end is None else max(int(existing_end), page_end)

                section_chunk_links.append(
                    {
                        "section_id": section_id,
                        "chunk_id": chunk_id,
                        "owner_id": db_owner_id,
                    }
                )

            tree_node_id = _coerce_str(metadata.get("semantic_unit_id")) or chunk_id
            semantic_unit_type = _coerce_str(metadata.get("semantic_unit_type")) or "text"
            node_type = _normalize_node_type(semantic_unit_type)
            section_path = str(metadata.get("section_path") or "").strip()
            page_start = _coerce_int(metadata.get("page_start"))
            page_end = _coerce_int(metadata.get("page_end"))
            summary = _tree_node_summary(metadata, chunk.content or "")
            token_count = _coerce_int(metadata.get("token_count"))
            resource_urls: Optional[List[str]] = None
            urls = metadata.get("image_urls")
            if isinstance(urls, list):
                cleaned = [str(u).strip() for u in urls if str(u or "").strip()]
                if cleaned:
                    resource_urls = cleaned

            if tree_node_id and source_file_id:
                tree_key = (db_owner_id, tree_node_id)
                existing = tree_node_data_by_key.get(tree_key)
                if existing is None:
                    tree_node_data_by_key[tree_key] = {
                        "node_id": tree_node_id,
                        "owner_id": db_owner_id,
                        "source_file_id": source_file_id,
                        "node_type": node_type,
                        "semantic_unit_type": semantic_unit_type,
                        "section_id": section_id or None,
                        "section_path": section_path or None,
                        "page_start": page_start,
                        "page_end": page_end,
                        "summary": summary,
                        "resource_urls": resource_urls,
                        "resource_paths": resource_urls,
                        "token_count": token_count,
                    }
                else:
                    if not existing.get("node_type") or (
                        existing.get("node_type") == tool_defaults.SECTION_NODE_TYPE_DEFAULT
                        and node_type != tool_defaults.SECTION_NODE_TYPE_DEFAULT
                    ):
                        existing["node_type"] = node_type
                    if not existing.get("semantic_unit_type") and semantic_unit_type:
                        existing["semantic_unit_type"] = semantic_unit_type
                    if not existing.get("section_id") and section_id:
                        existing["section_id"] = section_id
                    if not existing.get("section_path") and section_path:
                        existing["section_path"] = section_path
                    if page_start is not None:
                        prev = existing.get("page_start")
                        existing["page_start"] = page_start if prev is None else min(int(prev), page_start)
                    if page_end is not None:
                        prev = existing.get("page_end")
                        existing["page_end"] = page_end if prev is None else max(int(prev), page_end)
                    if summary and not existing.get("summary"):
                        existing["summary"] = summary
                    if resource_urls and not existing.get("resource_urls"):
                        existing["resource_urls"] = resource_urls
                    if resource_urls and not existing.get("resource_paths"):
                        existing["resource_paths"] = resource_urls
                    if token_count is not None:
                        existing["token_count"] = int(existing.get("token_count") or 0) + token_count

                chunk_link_key = (db_owner_id, tree_node_id, chunk_id)
                if chunk_link_key not in tree_node_chunk_link_keys:
                    tree_node_chunk_link_keys.add(chunk_link_key)
                    tree_node_chunk_links.append(
                        {"node_id": tree_node_id, "chunk_id": chunk_id, "owner_id": db_owner_id}
                    )

                if section_id:
                    section_link_key = (db_owner_id, section_id, tree_node_id)
                    if section_link_key not in tree_node_section_link_keys:
                        tree_node_section_link_keys.add(section_link_key)
                        tree_node_section_links.append(
                            {"section_id": section_id, "node_id": tree_node_id, "owner_id": db_owner_id}
                        )

                parent_unit_id = _coerce_str(metadata.get("parent_unit_id"))
                if parent_unit_id and parent_unit_id != tree_node_id:
                    parent_link_key = (db_owner_id, parent_unit_id, tree_node_id)
                    if parent_link_key not in tree_node_parent_link_keys:
                        tree_node_parent_link_keys.add(parent_link_key)
                        tree_node_parent_links.append(
                            {"parent_id": parent_unit_id, "node_id": tree_node_id, "owner_id": db_owner_id}
                        )

            if enable_schema_layers:
                mindmap = None
                if isinstance(metadata.get("mindmap"), dict):
                    mindmap = metadata.get("mindmap")
                nodes_raw = mindmap.get("nodes") if isinstance(mindmap, dict) else None
                schema_nodes, schema_occurrences = build_schema_layer_payload(
                    mindmap_nodes=nodes_raw if isinstance(nodes_raw, list) else None,
                    chunk_id=chunk_id,
                    owner_id=owner_str,
                    db_owner_id=db_owner_id,
                    max_nodes=schema_layer_max_nodes,
                )
                for node in schema_nodes:
                    schema_nodes_by_id.setdefault(str(node.get("schema_id")), node)
                schema_links.extend(schema_occurrences)

            if enable_sdf_schema:
                sdf = metadata.get(SDF_KEY) if isinstance(metadata.get(SDF_KEY), dict) else None
                sdf_nodes, sdf_sub_edges, sdf_before, sdf_links = build_sdf_schema_payload(
                    sdf=sdf,
                    chunk_id=chunk_id,
                    db_owner_id=db_owner_id,
                    max_events=sdf_max_events,
                    max_relations=sdf_max_relations,
                    max_source_chunks=sdf_max_source_chunks,
                )
                for node in sdf_nodes:
                    sdf_event_nodes_by_id.setdefault(str(node.get("sdf_event_id")), node)
                sdf_has_subevent_edges.extend(sdf_sub_edges)
                sdf_before_edges.extend(sdf_before)
                sdf_chunk_event_links.extend(sdf_links)

            # Process graph data
            if chunk.graph and chunk.graph.is_empty():
                if chunk.graph.metadata.get(EXTRACTION_ERROR_KEY):
                    owner_stats["chunks_extraction_failed"] += 1
                else:
                    owner_stats["chunks_graph_empty"] += 1
            if chunk.graph and not chunk.graph.is_empty():
                domain = _select_domain_for_chunk(chunk, metadata, schema)
                domain_schema = schema.for_domain(domain) if schema else None
                schema_version = getattr(schema, "version", None) or "unmanaged"

                # Build entity mapping from NER outputs.
                # - `entity_name_normalized` is used for stable matching + hashing.
                # - `entity_name` keeps the display value for explainability/exports.
                entity_name_to_type_keys: dict[str, set[str]] = {}
                entity_key_to_type_display: dict[tuple[str, str], str] = {}
                entity_key_to_display: dict[tuple[str, str], str] = {}
                entity_key_to_canonical: dict[tuple[str, str], str] = {}
                for entity_dict in chunk.graph.entities:
                    raw_name = entity_dict.get("entity_name")
                    if not raw_name:
                        continue
                    display_name = str(raw_name).strip()
                    if not display_name:
                        continue

                    normalized_name = text_processing(display_name)
                    if not normalized_name:
                        continue

                    entity_type_display = str(entity_dict.get("entity_type", "Entity") or "Entity").strip() or "Entity"
                    entity_type_key = text_processing(entity_type_display) or "entity"

                    entity_name_to_type_keys.setdefault(normalized_name, set()).add(entity_type_key)
                    entity_key = (normalized_name, entity_type_key)
                    entity_key_to_type_display.setdefault(entity_key, entity_type_display)
                    entity_key_to_display.setdefault(entity_key, display_name)

                    canonical = None
                    if domain_schema is not None:
                        canonical = domain_schema.canonicalize_entity_name(display_name)
                    canonical_name = canonical or normalized_name
                    entity_key_to_canonical[entity_key] = canonical_name

                # Collect entity nodes + chunk mentions from NER outputs (not only triple endpoints).
                # This keeps the entity layer usable even when relation extraction is sparse or schema-rejected.
                mention_keys: set[tuple[str, str]] = set()
                business_time = _extract_business_time(metadata, chunk_obj=chunk)
                chunk_valid_from = business_time.get("valid_from") or business_time.get("effective_date")
                chunk_effective_date = business_time.get("effective_date") or business_time.get("valid_from")
                chunk_valid_to = business_time.get("valid_to")
                source_version = _extract_source_version(metadata)
                for (normalized_name, entity_type_key), entity_type_display in entity_key_to_type_display.items():
                    entity_id = compute_mdhash_id(
                        f"{normalized_name}|{entity_type_key}",
                        prefix="entity-",
                        owner_id=owner_str,
                    )
                    display_name = entity_key_to_display.get((normalized_name, entity_type_key)) or normalized_name
                    if entity_id not in entity_data:
                        canonical_name = entity_key_to_canonical.get((normalized_name, entity_type_key)) or normalized_name
                        entity_data[entity_id] = {
                            "entity_id": entity_id,
                            "entity_name": display_name,
                            "entity_name_normalized": normalized_name,
                            "entity_canonical_name": canonical_name,
                            # Canonical key keeps type to avoid collapsing same-name different-type entities in COUNT DISTINCT.
                            "entity_canonical_key": f"{canonical_name}|{entity_type_key}",
                            "entity_type": entity_type_display or "Entity",
                            "entity_type_key": entity_type_key,
                            "owner_id": db_owner_id,
                        }
                        mention_key = (chunk_id, entity_id)
                    if mention_key not in mention_keys:
                        mention_keys.add(mention_key)
                        mention_record = {"chunk_id": chunk_id, "entity_id": entity_id, "owner_id": db_owner_id}
                        mention_data.append(mention_record)
                        if enable_entity_mentions:
                            # Mention evidence is keyed per (chunk, surface_entity). Keep mention_id stable and owner-scoped.
                            mention_id = compute_mdhash_id(
                                f"{chunk_id}|{entity_id}",
                                prefix="mention-",
                                owner_id=db_owner_id,
                            )
                            payload = {
                                "mention_id": mention_id,
                                "chunk_id": chunk_id,
                                "surface_entity_id": entity_id,
                                "owner_id": db_owner_id,
                                "source_file_id": str(source_file_id or "").strip(),
                                "source_version": source_version,
                                "valid_from": str(chunk_valid_from or "").strip(),
                                "valid_to": str(chunk_valid_to or "").strip(),
                                "effective_date": str(chunk_effective_date or "").strip(),
                            }
                            entity_mention_data.append(payload)

                # Process and normalize relation triples (schema-governed predicate normalization)
                canonical_to_entity_keys: dict[str, set[tuple[str, str]]] = {}
                for entity_key, canonical_name in entity_key_to_canonical.items():
                    token = str(canonical_name or "").strip()
                    if not token:
                        continue
                    canonical_to_entity_keys.setdefault(token, set()).add(entity_key)

                enable_endpoint_fallback = bool(getattr(cfg, "enable_endpoint_canonical_fallback", True))

                def _resolve_endpoint_key(raw: Any) -> tuple[tuple[str, str] | None, bool, bool]:
                    """
                    Resolve a triple endpoint into an (entity_name_normalized, entity_type_key) key.

                    Returns:
                    - entity_key (normalized_name, type_key) when unambiguously resolved
                    - used_canonical_fallback: True when resolution required schema canonicalization fallback
                    - canonical_ambiguous: True when fallback matched >1 candidates (precision-first drop)
                    """

                    display = str(raw or "").strip()
                    if not display:
                        return None, False, False

                    normalized = text_processing(display)
                    if normalized and normalized in entity_name_to_type_keys:
                        types = entity_name_to_type_keys.get(normalized) or set()
                        if len(types) != 1:
                            return None, False, False
                        return (normalized, next(iter(types))), False, False

                    if not enable_endpoint_fallback or domain_schema is None:
                        return None, False, False

                    canonical = domain_schema.canonicalize_entity_name(display) or normalized
                    canonical = str(canonical or "").strip()
                    if not canonical:
                        return None, True, False
                    candidates = canonical_to_entity_keys.get(canonical) or set()
                    if len(candidates) != 1:
                        return None, True, bool(candidates)
                    return next(iter(candidates)), True, False

                # Triples may optionally include per-edge temporal bounds + paraphrased fact text.
                # Shape:
                #   (head_id, head_display, predicate, tail_id, tail_display, direction_sensitive, valid_from?, valid_to?, fact?)
                processed_triples: list[
                    tuple[str, str, str, str, str, bool, Optional[str], Optional[str], Optional[str]]
                ] = []
                for relation in chunk.graph.relations:
                    if len(relation) >= 3:
                        head_key, head_used_canonical, head_canonical_ambiguous = _resolve_endpoint_key(relation[0])
                        tail_key, tail_used_canonical, tail_canonical_ambiguous = _resolve_endpoint_key(relation[2])
                        owner_stats["triples_total"] += 1

                        # Enforce HippoRAG chunk-triples endpoint constraint (precision-first):
                        # triples must use extracted named entities as endpoints.
                        if head_key is None or tail_key is None:
                            if head_canonical_ambiguous or tail_canonical_ambiguous:
                                owner_stats["triples_dropped_canonical_ambiguous_endpoints"] += 1
                            else:
                                owner_stats["triples_dropped_endpoints"] += 1
                            continue

                        head, head_type_key = head_key
                        tail, tail_type_key = tail_key

                        if not head or not tail:
                            owner_stats["triples_dropped_endpoints"] += 1
                            continue
                        head_id = compute_mdhash_id(f"{head}|{head_type_key}", prefix="entity-", owner_id=owner_str)
                        tail_id = compute_mdhash_id(f"{tail}|{tail_type_key}", prefix="entity-", owner_id=owner_str)
                        head_display = entity_key_to_display.get((head, head_type_key)) or head
                        tail_display = entity_key_to_display.get((tail, tail_type_key)) or tail
                        if head_used_canonical or tail_used_canonical:
                            owner_stats["triples_kept_via_canonical_endpoints"] += 1

                        raw_predicate = relation[1]
                        if domain_schema is not None:
                            normalization = domain_schema.normalize_predicate_with_meta(str(raw_predicate))
                            normalized_predicate = normalization.canonical_predicate
                            strategy = (normalization.normalization_strategy or "").strip().lower()
                            if strategy == "alias":
                                owner_stats["predicates_aliased"] += 1
                            elif strategy == "keep":
                                owner_stats["predicates_kept"] += 1
                            elif strategy == "collapse":
                                owner_stats["predicates_collapsed"] += 1
                            elif strategy == "reject":
                                owner_stats["predicates_rejected"] += 1
                            if normalization.allowlist_rejected:
                                owner_stats["predicates_allowlist_rejected"] += 1
                            direction_sensitive = bool(domain_schema.is_direction_sensitive_from_normalization(normalization))
                        else:
                            # Keep predicate token shape stable even without schema governance.
                            normalized_predicate = normalize_relation_token(str(raw_predicate))
                            direction_sensitive = True
                        if not normalized_predicate:
                            owner_stats["triples_dropped_schema"] += 1
                            continue

                        if not direction_sensitive:
                            owner_stats["triples_kept_direction_insensitive"] += 1
                        edge_valid_from = str(relation[3]).strip() if len(relation) > 3 and relation[3] is not None else None
                        edge_valid_to = str(relation[4]).strip() if len(relation) > 4 and relation[4] is not None else None
                        edge_fact = str(relation[5]).strip() if len(relation) > 5 and relation[5] is not None else None
                        processed_triples.append(
                            (
                                head_id,
                                head_display,
                                normalized_predicate,
                                tail_id,
                                tail_display,
                                direction_sensitive,
                                edge_valid_from or None,
                                edge_valid_to or None,
                                edge_fact or None,
                            )
                        )
                        owner_stats["triples_kept"] += 1

                # Collect fact data (aggregated with provenance)
                business_time = {}
                if chunk.graph and isinstance(getattr(chunk.graph, "metadata", None), dict):
                    raw = chunk.graph.metadata.get(BUSINESS_TIME_KEY)
                    if isinstance(raw, dict):
                        business_time = dict(raw)
                if not business_time and isinstance(metadata.get(BUSINESS_TIME_KEY), dict):
                    business_time = dict(metadata.get(BUSINESS_TIME_KEY) or {})
                chunk_valid_from = business_time.get("valid_from") or business_time.get("effective_date")
                chunk_effective_date = business_time.get("effective_date") or business_time.get("valid_from")
                chunk_valid_to = business_time.get("valid_to")
                for (
                    head_id,
                    head_name,
                    relation_type,
                    tail_id,
                    tail_name,
                    direction_sensitive,
                    edge_valid_from,
                    edge_valid_to,
                    edge_fact,
                ) in processed_triples:
                    valid_from = edge_valid_from or (str(chunk_valid_from).strip() if chunk_valid_from else None)
                    valid_to = edge_valid_to or (str(chunk_valid_to).strip() if chunk_valid_to else None)
                    effective_date = edge_valid_from or (str(chunk_effective_date).strip() if chunk_effective_date else None)
                    upsert_fact_occurrence(
                        fact_data_by_id,
                        head_id=head_id,
                        head_name=head_name,
                        relation_type=relation_type,
                        tail_id=tail_id,
                        tail_name=tail_name,
                        chunk_id=chunk_id,
                        owner_id=owner_str,
                        db_owner_id=db_owner_id,
                        schema_version=schema_version,
                        domain=domain,
                        direction_sensitive=direction_sensitive,
                        max_source_chunks=max_source_chunks,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        effective_date=effective_date,
                        fact=edge_fact,
                    )

        # Prepare entity list for batch insertion
        entity_list = list(entity_data.values())
        fact_data = list(fact_data_by_id.values())

        # Phase 2: queryable canonicalization layer (canonical keys + aliases).
        canonical_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        canonical_links: List[Dict[str, Any]] = []
        alias_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        alias_links: List[Dict[str, Any]] = []
        for entity in entity_list:
            owner_key = str(entity.get("owner_id") or "").strip()
            entity_id = str(entity.get("entity_id") or "").strip()
            canonical_key = str(entity.get("entity_canonical_key") or "").strip()
            canonical_name = str(entity.get("entity_canonical_name") or "").strip()
            entity_type_key = str(entity.get("entity_type_key") or "").strip() or "entity"
            alias_text_normalized = str(entity.get("entity_name_normalized") or "").strip()
            alias_text_display = str(entity.get("entity_name") or "").strip()

            if not owner_key or not entity_id or not canonical_key or not canonical_name:
                continue

            canonical_id = compute_mdhash_id(canonical_key, prefix="canonical-", owner_id=owner_key)
            canonical_nodes_by_id.setdefault(
                canonical_id,
                {
                    "canonical_id": canonical_id,
                    "canonical_key": canonical_key,
                    "canonical_name": canonical_name,
                    "entity_type_key": entity_type_key,
                    "owner_id": owner_key,
                },
            )
            canonical_links.append({"entity_id": entity_id, "canonical_id": canonical_id, "owner_id": owner_key})

            # Only create alias nodes when the surface form differs from the canonicalized key.
            if alias_text_normalized and alias_text_normalized != canonical_name:
                alias_id = compute_mdhash_id(f"{alias_text_normalized}|{canonical_id}", prefix="alias-", owner_id=owner_key)
                alias_nodes_by_id.setdefault(
                    alias_id,
                    {
                        "alias_id": alias_id,
                        "alias_text_normalized": alias_text_normalized,
                        "alias_text": alias_text_display or alias_text_normalized,
                        "owner_id": owner_key,
                    },
                )
                alias_links.append({"alias_id": alias_id, "canonical_id": canonical_id, "owner_id": owner_key})

        canonical_nodes = list(canonical_nodes_by_id.values())
        alias_nodes = list(alias_nodes_by_id.values())
        schema_node_list = list(schema_nodes_by_id.values())
        sdf_event_node_list = list(sdf_event_nodes_by_id.values())

        kg_ingest_meta_payloads: List[Dict[str, Any]] = []
        for owner_id, counters in stats_by_owner.items():
            total = int(counters.get("triples_total", 0))
            dropped_endpoints = int(counters.get("triples_dropped_endpoints", 0))
            dropped_ambiguous = int(counters.get("triples_dropped_ambiguous_endpoints", 0))
            drop_ratio = (float(dropped_endpoints) / float(total)) if total else 0.0
            payload = {
                "owner_id": owner_id,
                "chunks_total": int(counters.get("chunks_total", 0)),
                "chunks_graph_empty": int(counters.get("chunks_graph_empty", 0)),
                "chunks_extraction_failed": int(counters.get("chunks_extraction_failed", 0)),
                "triples_total": total,
                "triples_kept": int(counters.get("triples_kept", 0)),
                "triples_kept_via_canonical_endpoints": int(counters.get("triples_kept_via_canonical_endpoints", 0)),
                "triples_dropped_endpoints": dropped_endpoints,
                "triples_dropped_ambiguous_endpoints": dropped_ambiguous,
                "triples_dropped_canonical_ambiguous_endpoints": int(
                    counters.get("triples_dropped_canonical_ambiguous_endpoints", 0)
                ),
                "triples_dropped_schema": int(counters.get("triples_dropped_schema", 0)),
                "predicates_aliased": int(counters.get("predicates_aliased", 0)),
                "predicates_kept": int(counters.get("predicates_kept", 0)),
                "predicates_collapsed": int(counters.get("predicates_collapsed", 0)),
                "predicates_rejected": int(counters.get("predicates_rejected", 0)),
                "predicates_allowlist_rejected": int(counters.get("predicates_allowlist_rejected", 0)),
                "triples_kept_direction_insensitive": int(counters.get("triples_kept_direction_insensitive", 0)),
                "endpoint_drop_ratio": float(drop_ratio),
                "fact_provenance_max_source_chunks": int(max_source_chunks),
            }
            kg_ingest_meta_payloads.append(payload)
            logger.info(
                "KG ingest stats (owner=%s): total=%s kept=%s dropped_endpoints=%s dropped_schema=%s endpoint_drop_ratio=%.4f",
                owner_id,
                payload["triples_total"],
                payload["triples_kept"],
                payload["triples_dropped_endpoints"],
                payload["triples_dropped_schema"],
                payload["endpoint_drop_ratio"],
            )

        if section_data_by_key:
            delimiter = pageindex_cfg.section_path_delimiter()
            path_index: Dict[tuple[str, str, str], str] = {}
            for record in section_data_by_key.values():
                owner_id = str(record.get("owner_id") or "").strip()
                source_file_id = str(record.get("source_file_id") or "").strip()
                section_path = str(record.get("section_path") or "").strip()
                section_id = str(record.get("section_id") or "").strip()
                if owner_id and source_file_id and section_path and section_id:
                    path_index[(owner_id, source_file_id, section_path)] = section_id

            for record in section_data_by_key.values():
                if record.get("section_parent_id"):
                    continue
                section_path = str(record.get("section_path") or "").strip()
                if not section_path or delimiter not in section_path:
                    continue
                parent_path = delimiter.join([seg for seg in section_path.split(delimiter)[:-1] if seg])
                if not parent_path:
                    continue
                owner_id = str(record.get("owner_id") or "").strip()
                source_file_id = str(record.get("source_file_id") or "").strip()
                parent_id = path_index.get((owner_id, source_file_id, parent_path))
                if parent_id:
                    record["section_parent_id"] = parent_id

        logger.info(
            f"Batch data prepared: {len(chunk_data)} chunks, {len(entity_list)} entities, "
            f"{len(mention_data)} mentions, {len(fact_data)} facts"
        )

        # Batch insert using single transaction
        with self._driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                # For chunk upserts (same chunk_id re-index), remove stale evidence first so the graph remains stable.
                # This obeys "incremental + reconciliation" philosophy (like Graphiti), and avoids append-only drift.
                if (str(getattr(cfg, "chunk_upsert_policy", "append") or "").strip().lower() == "replace") and chunk_data:
                    try:
                        chunk_keys = [{"chunk_id": c["chunk_id"], "owner_id": c["owner_id"]} for c in chunk_data if c.get("chunk_id") and c.get("owner_id")]
                        run_chunk_replace_cleanup(tx, chunk_keys=chunk_keys)
                    except Exception as exc:  # noqa: BLE001
                        # No silent fallback: failing cleanup makes update semantics undefined, so surface clearly.
                        raise RuntimeError(f"Chunk upsert cleanup failed (chunk_upsert_policy=replace): {exc}") from exc

                # 1. Batch insert chunks
                if chunk_data:
                    chunk_query = """
                    UNWIND $chunks AS chunk
                    MERGE (c:Chunk {chunk_id: chunk.chunk_id})
                    SET c.content = chunk.content,
                        c.metadata = chunk.metadata,
                        c.owner_id = chunk.owner_id,
                        c.source_file_id = chunk.source_file_id,
                        c.page_start = COALESCE(chunk.page_start, c.page_start),
                        c.page_end = COALESCE(chunk.page_end, c.page_end),
                        c.updated_at = datetime(),
                        c.created_at = COALESCE(c.created_at, datetime())
                    """
                    tx.run(chunk_query, {'chunks': chunk_data})
                    logger.info(f"  Batch inserted {len(chunk_data)} chunks")

                if section_data_by_key:
                    section_query = """
                    UNWIND $sections AS section
                    MERGE (s:Section {section_id: section.section_id, owner_id: section.owner_id})
                    SET s.source_file_id = COALESCE(section.source_file_id, s.source_file_id),
                        s.section_path = COALESCE(section.section_path, s.section_path),
                        s.section_title = COALESCE(section.section_title, s.section_title),
                        s.section_level = COALESCE(section.section_level, s.section_level),
                        s.section_parent_id = COALESCE(section.section_parent_id, s.section_parent_id),
                        s.page_start = COALESCE(section.page_start, s.page_start),
                        s.page_end = COALESCE(section.page_end, s.page_end),
                        s.updated_at = datetime(),
                        s.created_at = COALESCE(s.created_at, datetime())
                    """
                    tx.run(section_query, {"sections": list(section_data_by_key.values())})
                    logger.info("  Upserted %s Section nodes", len(section_data_by_key))

                if section_data_by_key:
                    parent_links = [
                        {"owner_id": s.get("owner_id"), "section_id": s.get("section_id"), "parent_id": s.get("section_parent_id")}
                        for s in section_data_by_key.values()
                        if s.get("section_parent_id")
                    ]
                    if parent_links:
                        parent_query = """
                        UNWIND $links AS link
                        MATCH (c:Section {section_id: link.section_id, owner_id: link.owner_id})
                        MATCH (p:Section {section_id: link.parent_id, owner_id: link.owner_id})
                        MERGE (p)-[r:PARENT_OF {section_id: link.section_id, parent_id: link.parent_id}]->(c)
                        SET r.updated_at = datetime(),
                            r.created_at = COALESCE(r.created_at, datetime())
                        """
                        tx.run(parent_query, {"links": parent_links})
                        logger.info("  Upserted %s PARENT_OF relationships", len(parent_links))

                if section_chunk_links:
                    chunk_link_query = """
                    UNWIND $links AS link
                    MATCH (s:Section {section_id: link.section_id, owner_id: link.owner_id})
                    MATCH (c:Chunk {chunk_id: link.chunk_id})
                    WHERE COALESCE(c.owner_id, link.owner_id) = link.owner_id
                    MERGE (s)-[r:HAS_CHUNK {section_id: link.section_id, chunk_id: link.chunk_id}]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(chunk_link_query, {"links": section_chunk_links})
                    logger.info("  Upserted %s HAS_CHUNK relationships", len(section_chunk_links))

                if tree_node_data_by_key:
                    tree_node_query = """
                    UNWIND $nodes AS node
                    MERGE (t:TreeNode {node_id: node.node_id, owner_id: node.owner_id})
                    SET t.source_file_id = COALESCE(node.source_file_id, t.source_file_id),
                        t.section_id = COALESCE(node.section_id, t.section_id),
                        t.section_path = COALESCE(node.section_path, t.section_path),
                        t.node_type = COALESCE(node.node_type, t.node_type),
                        t.semantic_unit_type = COALESCE(node.semantic_unit_type, t.semantic_unit_type),
                        t.page_start = CASE WHEN node.page_start IS NULL THEN t.page_start ELSE node.page_start END,
                        t.page_end = CASE WHEN node.page_end IS NULL THEN t.page_end ELSE node.page_end END,
                        t.summary = CASE WHEN node.summary IS NULL OR node.summary = '' THEN t.summary ELSE node.summary END,
                        t.resource_urls = CASE
                            WHEN node.resource_urls IS NULL OR size(node.resource_urls) = 0 THEN t.resource_urls
                            ELSE node.resource_urls
                        END,
                        t.resource_paths = CASE
                            WHEN node.resource_paths IS NULL OR size(node.resource_paths) = 0 THEN t.resource_paths
                            ELSE node.resource_paths
                        END,
                        t.token_count = COALESCE(node.token_count, t.token_count),
                        t.updated_at = datetime(),
                        t.created_at = COALESCE(t.created_at, datetime())
                    """
                    tx.run(tree_node_query, {"nodes": list(tree_node_data_by_key.values())})
                    logger.info("  Upserted %s TreeNode nodes", len(tree_node_data_by_key))

                if tree_node_section_links:
                    section_tree_link_query = """
                    UNWIND $links AS link
                    MATCH (s:Section {section_id: link.section_id, owner_id: link.owner_id})
                    MATCH (t:TreeNode {node_id: link.node_id, owner_id: link.owner_id})
                    MERGE (s)-[r:HAS_CHILD {section_id: link.section_id, node_id: link.node_id}]->(t)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(section_tree_link_query, {"links": tree_node_section_links})
                    logger.info("  Upserted %s Section->TreeNode HAS_CHILD relationships", len(tree_node_section_links))

                if tree_node_parent_links:
                    node_parent_query = """
                    UNWIND $links AS link
                    MATCH (p:TreeNode {node_id: link.parent_id, owner_id: link.owner_id})
                    MATCH (c:TreeNode {node_id: link.node_id, owner_id: link.owner_id})
                    MERGE (p)-[r:HAS_CHILD {parent_id: link.parent_id, node_id: link.node_id}]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(node_parent_query, {"links": tree_node_parent_links})
                    logger.info("  Upserted %s TreeNode HAS_CHILD relationships", len(tree_node_parent_links))

                if tree_node_chunk_links:
                    tree_node_chunk_query = """
                    UNWIND $links AS link
                    MATCH (t:TreeNode {node_id: link.node_id, owner_id: link.owner_id})
                    MATCH (c:Chunk {chunk_id: link.chunk_id})
                    WHERE COALESCE(c.owner_id, link.owner_id) = link.owner_id
                    MERGE (t)-[r:HAS_CHUNK {node_id: link.node_id, chunk_id: link.chunk_id}]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(tree_node_chunk_query, {"links": tree_node_chunk_links})
                    logger.info("  Upserted %s TreeNode HAS_CHUNK relationships", len(tree_node_chunk_links))

                # 2. Batch insert entities and track new ones
                if entity_list:
                    entity_query = """
                    UNWIND $entities AS entity
                    MERGE (e:Entity {entity_id: entity.entity_id})
                    ON CREATE SET e.entity_name = entity.entity_name,
                                  e.entity_text = entity.entity_name,
                                  e.entity_name_normalized = entity.entity_name_normalized,
                                  e.entity_canonical_name = entity.entity_canonical_name,
                                  e.entity_canonical_key = entity.entity_canonical_key,
                                  e.entity_type = entity.entity_type,
                                  e.entity_type_key = entity.entity_type_key,
                                  e.node_type = 'entity',
                                  e.attributes = '{}',
                                  e.owner_id = entity.owner_id,
                                  e.created_at = datetime(),
                                  e.updated_at = datetime(),
                                  e.is_new = true
                    ON MATCH SET e.entity_name = entity.entity_name,
                                 e.entity_text = entity.entity_name,
                                 e.entity_name_normalized = entity.entity_name_normalized,
                                 e.entity_canonical_name = entity.entity_canonical_name,
                                 e.entity_canonical_key = entity.entity_canonical_key,
                                 e.entity_type = entity.entity_type,
                                 e.entity_type_key = entity.entity_type_key,
                                 e.owner_id = entity.owner_id,
                                 e.updated_at = datetime(),
                                 e.is_new = false
                    RETURN e.entity_id AS entity_id, e.is_new AS is_new
                    """
                    result = tx.run(entity_query, {'entities': entity_list})
                    for record in result:
                        if record['is_new']:
                            new_entity_ids.append(record['entity_id'])
                    logger.info(f"  Batch inserted {len(entity_list)} entities ({len(new_entity_ids)} new)")

                # 2.5 Canonicalization layer: canonical key nodes + canonical/alias relationships.
                if canonical_nodes:
                    canonical_query = """
                    UNWIND $canonicals AS c
                    // NOTE: Do NOT include extra labels (e.g. `:Concept`) in the MERGE pattern.
                    // Older DBs may already contain (:EntityCanonical {canonical_id}) nodes without the extra label.
                    // `MERGE (n:EntityCanonical:Concept {canonical_id})` would then attempt to create a new node and
                    // violate the unique constraint on :EntityCanonical(canonical_id).
                    MERGE (n:EntityCanonical {canonical_id: c.canonical_id})
                    ON CREATE SET n.owner_id = c.owner_id,
                                  n.canonical_key = c.canonical_key,
                                  n.canonical_name = c.canonical_name,
                                  n.entity_type_key = c.entity_type_key,
                                  n.created_at = datetime(),
                                  n.updated_at = datetime()
                    ON MATCH SET  n.owner_id = c.owner_id,
                                  n.canonical_key = c.canonical_key,
                                  n.canonical_name = c.canonical_name,
                                  n.entity_type_key = c.entity_type_key,
                                  n.updated_at = datetime()
                    SET n:Concept
                    """
                    tx.run(canonical_query, {"canonicals": canonical_nodes})
                    logger.info("  Upserted %s EntityCanonical nodes", len(canonical_nodes))

                if canonical_links:
                    canonical_rel_query = """
                    UNWIND $links AS link
                    MATCH (e:Entity {entity_id: link.entity_id, owner_id: link.owner_id})
                    MATCH (c:EntityCanonical {canonical_id: link.canonical_id})
                    MERGE (e)-[r:CANONICAL_OF]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(canonical_rel_query, {"links": canonical_links})
                    logger.info("  Upserted %s CANONICAL_OF relationships", len(canonical_links))

                    # Concept-layer alias: expose the canonicalization mapping as a concept edge for downstream tooling.
                    concept_rel_query = """
                    UNWIND $links AS link
                    MATCH (e:Entity {entity_id: link.entity_id, owner_id: link.owner_id})
                    MATCH (c:EntityCanonical {canonical_id: link.canonical_id})
                    MERGE (e)-[r:HAS_CONCEPT]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(concept_rel_query, {"links": canonical_links})
                    logger.info("  Upserted %s HAS_CONCEPT relationships", len(canonical_links))

                if alias_nodes:
                    alias_query = """
                    UNWIND $aliases AS a
                    MERGE (n:EntityAlias {alias_id: a.alias_id})
                    ON CREATE SET n.owner_id = a.owner_id,
                                  n.alias_text_normalized = a.alias_text_normalized,
                                  n.alias_text = a.alias_text,
                                  n.created_at = datetime(),
                                  n.updated_at = datetime()
                    ON MATCH SET  n.owner_id = a.owner_id,
                                  n.alias_text_normalized = a.alias_text_normalized,
                                  n.alias_text = a.alias_text,
                                  n.updated_at = datetime()
                    """
                    tx.run(alias_query, {"aliases": alias_nodes})
                    logger.info("  Upserted %s EntityAlias nodes", len(alias_nodes))

                if alias_links:
                    alias_rel_query = """
                    UNWIND $links AS link
                    MATCH (a:EntityAlias {alias_id: link.alias_id, owner_id: link.owner_id})
                    MATCH (c:EntityCanonical {canonical_id: link.canonical_id})
                    MERGE (a)-[r:ALIAS_OF]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(alias_rel_query, {"links": alias_links})
                    logger.info("  Upserted %s ALIAS_OF relationships", len(alias_links))

                # 2.6 Schema layer nodes derived from mindmap (Concept/Process/Instance scaffolding).
                if schema_node_list:
                    schema_query = """
                    UNWIND $nodes AS n
                    MERGE (s:SchemaNode {schema_id: n.schema_id})
                    ON CREATE SET s.owner_id = n.owner_id,
                                  s.layer = n.layer,
                                  s.text = n.text,
                                  s.text_normalized = n.text_normalized,
                                  s.created_at = datetime(),
                                  s.updated_at = datetime()
                    ON MATCH SET  s.owner_id = n.owner_id,
                                  s.layer = n.layer,
                                  s.text = n.text,
                                  s.text_normalized = n.text_normalized,
                                  s.updated_at = datetime()
                    """
                    tx.run(schema_query, {"nodes": schema_node_list})
                    logger.info("  Upserted %s SchemaNode nodes", len(schema_node_list))

                if schema_links:
                    schema_link_query = """
                    UNWIND $links AS link
                    MATCH (c:Chunk {chunk_id: link.chunk_id, owner_id: link.owner_id})
                    MATCH (s:SchemaNode {schema_id: link.schema_id})
                    MERGE (c)-[r:HAS_SCHEMA_NODE {chunk_id: link.chunk_id, schema_id: link.schema_id}]->(s)
                    SET r.owner_id = link.owner_id,
                        r.level = link.level,
                        r.layer = link.layer,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(schema_link_query, {"links": schema_links})
                    logger.info("  Upserted %s HAS_SCHEMA_NODE relationships", len(schema_links))

                # 2.7 Process schema (SDFEvent + SDF_BEFORE/HAS_SUBEVENT).
                if sdf_event_node_list:
                    sdf_event_query = """
                    UNWIND $events AS e
                    MERGE (n:SDFEvent {sdf_event_id: e.sdf_event_id})
                    WITH n, e,
                         CASE
                             WHEN n.source_chunk_ids IS NULL THEN e.source_chunk_ids
                             ELSE n.source_chunk_ids + [cid IN e.source_chunk_ids WHERE NOT cid IN n.source_chunk_ids]
                         END AS merged_chunk_ids
                    SET n.owner_id = e.owner_id,
                        n.doc_namespace = e.doc_namespace,
                        n.name = e.name,
                        n.name_normalized = e.name_normalized,
                        n.description = COALESCE(e.description, n.description),
                        n.children_gate = COALESCE(e.children_gate, n.children_gate),
                        n.effective_date = COALESCE(e.effective_date, n.effective_date),
                        n.valid_from = COALESCE(e.valid_from, n.valid_from),
                        n.valid_to = COALESCE(e.valid_to, n.valid_to),
                        n.scope = COALESCE(e.scope, n.scope),
                        n.priority = COALESCE(e.priority, n.priority),
                        n.attributes_json = COALESCE(e.attributes_json, n.attributes_json),
                        n.occurrences = COALESCE(n.occurrences, 0) + COALESCE(toInteger(e.occurrences), 1),
                        n.source_chunk_ids = merged_chunk_ids[..$max_source_chunks],
                        n.source_chunk_ids_truncated = COALESCE(n.source_chunk_ids_truncated, false)
                            OR COALESCE(e.source_chunk_ids_truncated, false)
                            OR ($max_source_chunks > 0 AND size(merged_chunk_ids) > $max_source_chunks),
                        n.updated_at = datetime(),
                        n.created_at = COALESCE(n.created_at, datetime())
                    """
                    tx.run(sdf_event_query, {"events": sdf_event_node_list, "max_source_chunks": int(sdf_max_source_chunks)})
                    logger.info("  Upserted %s SDFEvent nodes", len(sdf_event_node_list))

                if sdf_has_subevent_edges:
                    sdf_child_query = """
                    UNWIND $edges AS e
                    MATCH (p:SDFEvent {sdf_event_id: e.parent_id})
                    MATCH (c:SDFEvent {sdf_event_id: e.child_id})
                    MERGE (p)-[r:SDF_HAS_SUBEVENT {parent_id: e.parent_id, child_id: e.child_id}]->(c)
                    WITH r, e,
                         CASE
                             WHEN r.source_chunk_ids IS NULL THEN e.source_chunk_ids
                             ELSE r.source_chunk_ids + [cid IN e.source_chunk_ids WHERE NOT cid IN r.source_chunk_ids]
                         END AS merged_chunk_ids
                    SET r.owner_id = e.owner_id,
                        r.doc_namespace = e.doc_namespace,
                        r.importance = COALESCE(e.importance, r.importance),
                        r.occurrences = COALESCE(r.occurrences, 0) + COALESCE(toInteger(e.occurrences), 1),
                        r.source_chunk_ids = merged_chunk_ids[..$max_source_chunks],
                        r.source_chunk_ids_truncated = COALESCE(r.source_chunk_ids_truncated, false)
                            OR COALESCE(e.source_chunk_ids_truncated, false)
                            OR ($max_source_chunks > 0 AND size(merged_chunk_ids) > $max_source_chunks),
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(sdf_child_query, {"edges": sdf_has_subevent_edges, "max_source_chunks": int(sdf_max_source_chunks)})
                    logger.info("  Upserted %s SDF_HAS_SUBEVENT relationships", len(sdf_has_subevent_edges))

                if sdf_before_edges:
                    sdf_before_query = """
                    UNWIND $edges AS e
                    MATCH (s:SDFEvent {sdf_event_id: e.subject_id})
                    MATCH (t:SDFEvent {sdf_event_id: e.object_id})
                    MERGE (s)-[r:SDF_BEFORE {subject_id: e.subject_id, object_id: e.object_id}]->(t)
                    WITH r, e,
                         CASE
                             WHEN r.source_chunk_ids IS NULL THEN e.source_chunk_ids
                             ELSE r.source_chunk_ids + [cid IN e.source_chunk_ids WHERE NOT cid IN r.source_chunk_ids]
                         END AS merged_chunk_ids
                    SET r.owner_id = e.owner_id,
                        r.doc_namespace = e.doc_namespace,
                        r.occurrences = COALESCE(r.occurrences, 0) + COALESCE(toInteger(e.occurrences), 1),
                        r.source_chunk_ids = merged_chunk_ids[..$max_source_chunks],
                        r.source_chunk_ids_truncated = COALESCE(r.source_chunk_ids_truncated, false)
                            OR COALESCE(e.source_chunk_ids_truncated, false)
                            OR ($max_source_chunks > 0 AND size(merged_chunk_ids) > $max_source_chunks),
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(sdf_before_query, {"edges": sdf_before_edges, "max_source_chunks": int(sdf_max_source_chunks)})
                    logger.info("  Upserted %s SDF_BEFORE relationships", len(sdf_before_edges))

                if sdf_chunk_event_links:
                    sdf_chunk_link_query = """
                    UNWIND $links AS link
                    MATCH (c:Chunk {chunk_id: link.chunk_id, owner_id: link.owner_id})
                    MATCH (e:SDFEvent {sdf_event_id: link.sdf_event_id})
                    MERGE (c)-[r:HAS_SDF_EVENT {chunk_id: link.chunk_id, sdf_event_id: link.sdf_event_id}]->(e)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(sdf_chunk_link_query, {"links": sdf_chunk_event_links})
                    logger.info("  Upserted %s HAS_SDF_EVENT relationships", len(sdf_chunk_event_links))

                # 3. Batch create chunk-entity relationships
                if mention_data:
                    mention_query = """
                    UNWIND $mentions AS m
                    MATCH (c:Chunk {chunk_id: m.chunk_id, owner_id: m.owner_id})
                    MATCH (e:Entity {entity_id: m.entity_id, owner_id: m.owner_id})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.weight = COALESCE(r.weight, 0.0) + 1.0,
                        r.owner_id = m.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(mention_query, {'mentions': mention_data})
                    logger.info(f"  Batch created {len(mention_data)} MENTIONS relationships")

                # 3.5 L0: materialize entity mentions (occurrence evidence) for KG maintenance.
                if entity_mention_data:
                    # Optional cap: only materialize up to remaining quota (if provided by caller).
                    capped = entity_mention_data
                    if entity_mentions_remaining is not None:
                        try:
                            remaining_int = int(entity_mentions_remaining)
                        except Exception:
                            remaining_int = None
                        if remaining_int is not None and remaining_int >= 0:
                            capped = entity_mention_data[:remaining_int]

                    from encapsulation.database.graph_db.neo4j_entity_mention_cypher import UPSERT_ENTITY_MENTIONS_QUERY

                    t0 = time.perf_counter()
                    tx.run(UPSERT_ENTITY_MENTIONS_QUERY, {"entity_mentions": capped})
                    elapsed = float(time.perf_counter() - t0)
                    l0_stats["entity_mentions"]["attempted"] = int(len(entity_mention_data))
                    l0_stats["entity_mentions"]["written"] = int(len(capped))
                    l0_stats["entity_mentions"]["elapsed_s"] = float(elapsed)
                    logger.info("  L0: materialized %s EntityMention nodes (attempted=%s, elapsed=%.3fs)", len(capped), len(entity_mention_data), elapsed)
                elif enable_entity_mentions:
                    l0_stats["entity_mentions"]["attempted"] = 0
                    l0_stats["entity_mentions"]["written"] = 0
                    if l0_stats["entity_mentions"]["skipped_reason"] is None:
                        l0_stats["entity_mentions"]["skipped_reason"] = "no_mentions"

                # 4. Batch create fact relationships
                if fact_data:
                    fact_query = """
                    UNWIND $facts AS f
                    MATCH (e1:Entity {entity_id: f.head_id, owner_id: f.owner_id})
                    MATCH (e2:Entity {entity_id: f.tail_id, owner_id: f.owner_id})
                    MERGE (e1)-[r:RELATES_TO {fact_id: f.fact_id}]->(e2)
                    WITH e1, e2, r, f,
                         CASE
                             WHEN r.source_chunk_ids IS NULL THEN f.source_chunk_ids
                             ELSE r.source_chunk_ids + [cid IN f.source_chunk_ids WHERE NOT cid IN r.source_chunk_ids]
                         END AS merged_chunk_ids
                    SET r.head = f.head_name,
                        r.predicate = f.relation_type,
                        r.tail = f.tail_name,
                        r.text = f.fact_text,
                        r.owner_id = f.owner_id,
                        r.schema_version = f.schema_version,
                        r.domain = f.domain,
                        r.valid_from = CASE WHEN f.valid_from IS NULL OR f.valid_from = '' THEN r.valid_from ELSE datetime(f.valid_from) END,
                        r.valid_to = CASE WHEN f.valid_to IS NULL OR f.valid_to = '' THEN r.valid_to ELSE datetime(f.valid_to) END,
                        r.effective_date = CASE WHEN f.effective_date IS NULL OR f.effective_date = '' THEN r.effective_date ELSE datetime(f.effective_date) END,
                        r.occurrences = COALESCE(r.occurrences, 0) + COALESCE(toInteger(f.occurrences), 1),
                        r.source_chunk_ids = merged_chunk_ids[..$max_source_chunks],
                        r.source_chunk_ids_truncated = COALESCE(r.source_chunk_ids_truncated, false)
                            OR COALESCE(f.source_chunk_ids_truncated, false)
                            OR size(merged_chunk_ids) > $max_source_chunks,
                        r.weight = COALESCE(r.weight, 0.0) + COALESCE(toFloat(f.occurrences), 1.0),
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(fact_query, {"facts": fact_data, "max_source_chunks": int(max_source_chunks)})
                    logger.info(f"  Batch created {len(fact_data)} RELATES_TO relationships")

                # 5. Persist per-owner ingest stats (avoids process-global state and survives concurrency).
                if kg_ingest_meta_payloads:
                    meta_query = """
                    UNWIND $metas AS meta
                    MERGE (m:KGIngestMeta {owner_id: meta.owner_id})
                    SET m.chunks_total = meta.chunks_total,
                        m.chunks_graph_empty = meta.chunks_graph_empty,
                        m.chunks_extraction_failed = meta.chunks_extraction_failed,
                        m.triples_total = meta.triples_total,
                        m.triples_kept = meta.triples_kept,
                        m.triples_kept_via_canonical_endpoints = meta.triples_kept_via_canonical_endpoints,
                        m.triples_dropped_endpoints = meta.triples_dropped_endpoints,
                        m.triples_dropped_ambiguous_endpoints = meta.triples_dropped_ambiguous_endpoints,
                        m.triples_dropped_canonical_ambiguous_endpoints = meta.triples_dropped_canonical_ambiguous_endpoints,
                        m.triples_dropped_schema = meta.triples_dropped_schema,
                        m.predicates_aliased = meta.predicates_aliased,
                        m.predicates_kept = meta.predicates_kept,
                        m.predicates_collapsed = meta.predicates_collapsed,
                        m.predicates_rejected = meta.predicates_rejected,
                        m.predicates_allowlist_rejected = meta.predicates_allowlist_rejected,
                        m.triples_kept_direction_insensitive = meta.triples_kept_direction_insensitive,
                        m.endpoint_drop_ratio = meta.endpoint_drop_ratio,
                        m.fact_provenance_max_source_chunks = meta.fact_provenance_max_source_chunks,
                        m.updated_at = datetime(),
                        m.created_at = COALESCE(m.created_at, datetime())
                    """
                    tx.run(meta_query, {"metas": kg_ingest_meta_payloads})
                    logger.info("  Updated %s KGIngestMeta rows", len(kg_ingest_meta_payloads))

                tx.commit()

        elapsed = time.time() - start_time
        logger.info(f"Batch insertion completed in {elapsed:.2f}s")

        return new_entity_ids, l0_stats
