import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.knowledge_graph.heuristic_entity import iter_candidate_phrases, normalize_entity_key
from core.utils.stopwords import get_stopwords
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.retrieval.graph_neighborhood_config import GraphNeighborhoodRetrieverConfig

logger = logging.getLogger(__name__)


def _iter_edge_attr_dicts(edge_data: Any) -> list[dict]:
    if not edge_data:
        return []
    if isinstance(edge_data, dict) and "relation_type" in edge_data:
        return [edge_data]
    if isinstance(edge_data, dict):
        values = list(edge_data.values())
        return [v for v in values if isinstance(v, dict)]
    return []


@dataclass(frozen=True)
class _ScoredChunk:
    chunk_id: str
    score: float


class GraphNeighborhoodRetriever:
    """Heuristic graph neighborhood retriever for benchmark/offline mode.

    Requires a NetworkX-like graph store (with chunk/entity nodes + MENTIONS edges).
    """

    def __init__(self, config: "GraphNeighborhoodRetrieverConfig"):
        self.config = config
        self.graph_store = config.graph_store_config.build()
        self.graph = getattr(self.graph_store, "graph", None)
        if self.graph is None:
            raise ValueError("GraphNeighborhoodRetriever requires a graph_store with `.graph` (NetworkX).")

        self._stopwords = get_stopwords(getattr(config, "stopwords_language", "en"))
        extra = getattr(config, "extra_stopwords", None) or []
        self._stopwords = self._stopwords.union({normalize_entity_key(x) for x in extra if str(x).strip()})

        self._entity_index = self._build_entity_index()
        self._entity_index_keys = list(self._entity_index.keys())

    def _build_entity_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") != "Entity":
                continue
            name = str(node_data.get("entity_name") or "").strip()
            if not name:
                continue
            key = normalize_entity_key(name)
            if not key:
                continue
            index.setdefault(key, []).append(str(node_id))
        return index

    def retrieve_with_trace(self, query: str, *, k: int = 10) -> tuple[list[Chunk], dict]:
        max_words = int(self.config.max_entity_words)
        min_chars = int(self.config.min_entity_len_chars)
        max_seeds = int(self.config.max_seed_entities)
        max_neighbors_per_seed = int(self.config.max_neighbors_per_seed)
        max_chunks_per_entity = int(self.config.max_chunks_per_entity)
        seed_weight = float(self.config.seed_chunk_weight)
        neighbor_weight = float(self.config.neighbor_chunk_weight)
        max_mentions = int(self.config.max_mentions_per_seed_entity)
        seed_text_match_boost = float(self.config.seed_text_match_boost)
        traversal_mode = str(self.config.neighbor_traversal_mode or "both").strip().lower()
        allowed_relation_types = self.config.allowed_relation_types
        allowed_relation_types_set = {str(x).strip() for x in (allowed_relation_types or []) if str(x).strip()} or None

        phrases = iter_candidate_phrases(query or "", max_words=max_words)
        seed_keys: list[str] = []
        for phrase in phrases:
            key = normalize_entity_key(phrase)
            if not key or len(key) < min_chars:
                continue
            if key in self._stopwords:
                continue
            if key not in seed_keys:
                seed_keys.append(key)
            if len(seed_keys) >= max_seeds:
                break

        trace: dict = {"seed_entities": seed_keys}
        if not seed_keys:
            trace["fallback_reason"] = "no_seed_entities"
            return [], trace

        seed_node_ids: list[str] = []
        for key in seed_keys:
            seed_node_ids.extend(self._entity_index.get(key, []))

        if not seed_node_ids and bool(self.config.enable_fuzzy_entity_match):
            max_fuzzy = int(self.config.max_fuzzy_matches_per_seed)
            fuzzy_keys: list[str] = []
            for key in seed_keys:
                hits = 0
                for cand in self._entity_index_keys:
                    if key in cand or cand in key:
                        if cand not in fuzzy_keys:
                            fuzzy_keys.append(cand)
                            seed_node_ids.extend(self._entity_index.get(cand, []))
                            hits += 1
                            if hits >= max_fuzzy:
                                break
                if len(fuzzy_keys) >= max_fuzzy:
                    break
            if fuzzy_keys:
                trace["fuzzy_entity_keys"] = fuzzy_keys

        base_scores: dict[str, float] = {}
        seed_match_counts: dict[str, int] = {}

        def mention_chunk_count(entity_node_id: str, *, cap: int | None = None) -> int:
            count = 0
            for chunk_node_id in self.graph.predecessors(entity_node_id):
                edge_data = self.graph.get_edge_data(chunk_node_id, entity_node_id)
                if any(ed.get("relation_type") == "MENTIONS" for ed in _iter_edge_attr_dicts(edge_data)):
                    count += 1
                    if cap is not None and count >= cap:
                        break
            return count

        def seed_match_count_for_chunk_id(chunk_id: str) -> int:
            cached = seed_match_counts.get(chunk_id)
            if cached is not None:
                return cached
            chunk_node_id = f"chunk_{chunk_id}"
            content = ""
            if self.graph.has_node(chunk_node_id):
                content = str(self.graph.nodes[chunk_node_id].get("content") or "")
            content_norm = normalize_entity_key(content)
            count = 0
            for key in seed_keys:
                if key and key in content_norm:
                    count += 1
            seed_match_counts[chunk_id] = count
            return count

        def add_chunks_for_entity(entity_node_id: str, weight: float) -> None:
            count = 0
            for chunk_node_id in self.graph.predecessors(entity_node_id):
                edge_data = self.graph.get_edge_data(chunk_node_id, entity_node_id)
                if not any(ed.get("relation_type") == "MENTIONS" for ed in _iter_edge_attr_dicts(edge_data)):
                    continue
                chunk_id = str(self.graph.nodes[chunk_node_id].get("id_") or "").strip()
                if not chunk_id:
                    continue
                base_scores[chunk_id] = base_scores.get(chunk_id, 0.0) + weight
                if seed_text_match_boost > 0:
                    seed_match_count_for_chunk_id(chunk_id)
                count += 1
                if count >= max_chunks_per_entity:
                    break

        for node_id in seed_node_ids:
            if max_mentions > 0:
                mentions = mention_chunk_count(node_id, cap=max_mentions + 1)
                if mentions > max_mentions:
                    trace.setdefault("skipped_seed_entities", []).append(
                        {
                            "entity_node_id": node_id,
                            "reason": "too_many_mentions",
                            "mention_chunk_count": mentions,
                            "threshold": max_mentions,
                        }
                    )
                    continue
            add_chunks_for_entity(node_id, seed_weight)

            if not bool(self.config.enable_neighbor_expansion):
                continue

            neighbor_ids: list[str] = []
            neighbor_candidates: list[str] = []
            if traversal_mode in {"both", "out"}:
                neighbor_candidates.extend([str(x) for x in self.graph.successors(node_id)])
            if traversal_mode in {"both", "in"}:
                neighbor_candidates.extend([str(x) for x in self.graph.predecessors(node_id)])

            seen_neighbors: set[str] = set()
            for other in neighbor_candidates:
                if other in seen_neighbors:
                    continue
                seen_neighbors.add(other)
                if self.graph.nodes[other].get("node_type") != "Entity":
                    continue

                edge_datas: list[Any] = []
                if traversal_mode in {"both", "out"}:
                    edge_datas.append(self.graph.get_edge_data(node_id, other))
                if traversal_mode in {"both", "in"}:
                    edge_datas.append(self.graph.get_edge_data(other, node_id))

                attrs: list[dict] = []
                for edge_data in edge_datas:
                    attrs.extend(_iter_edge_attr_dicts(edge_data))

                rel_types = {str(a.get("relation_type") or "").strip() for a in attrs if str(a.get("relation_type") or "").strip()}
                if not rel_types:
                    continue
                if "MENTIONS" in rel_types:
                    continue
                if allowed_relation_types_set is not None and not (rel_types & allowed_relation_types_set):
                    continue

                neighbor_ids.append(str(other))
                if len(neighbor_ids) >= max_neighbors_per_seed:
                    break

            for neighbor_id in neighbor_ids:
                add_chunks_for_entity(neighbor_id, neighbor_weight)

        if not base_scores:
            trace["fallback_reason"] = "no_chunks_from_seeds"
            return [], trace

        if seed_text_match_boost > 0 and seed_match_counts:
            chunk_scores = {cid: (score + seed_text_match_boost * float(seed_match_counts.get(cid, 0))) for cid, score in base_scores.items()}
        else:
            chunk_scores = dict(base_scores)

        scored = sorted(
            (_ScoredChunk(chunk_id=cid, score=score) for cid, score in chunk_scores.items()),
            key=lambda item: (item.score, item.chunk_id),
            reverse=True,
        )
        top_ids = [item.chunk_id for item in scored[: int(k)]]
        chunks = list(self.graph_store.get_by_ids(top_ids) or [])
        chunk_by_id = {c.id: c for c in chunks if getattr(c, "id", None)}

        out: list[Chunk] = []
        for cid in top_ids:
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                continue
            meta = dict(getattr(chunk, "metadata", None) or {})
            meta.update(
                {
                    "retrieval_path": "graph_neighborhood",
                    "graph_neighborhood_score": float(chunk_scores.get(cid, 0.0)),
                    "seed_entities": seed_keys,
                }
            )
            chunk.metadata = meta
            out.append(chunk)
        return out, trace

    def invoke(self, query: str, *, k: int = 10) -> list[Chunk]:
        chunks, _ = self.retrieve_with_trace(query, k=k)
        return chunks
