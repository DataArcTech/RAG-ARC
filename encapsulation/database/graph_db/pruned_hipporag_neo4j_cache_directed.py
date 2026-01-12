import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import igraph as ig

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jDirectedCacheMixin:
    @staticmethod
    def _neighbor_pair(item: Any) -> Optional[Tuple[str, float]]:
        if not isinstance(item, tuple) or len(item) < 2:
            return None
        node_id = str(item[0] or "").strip()
        if not node_id:
            return None
        try:
            weight = float(item[1])
        except (TypeError, ValueError):
            weight = 1.0
        return node_id, weight

    @staticmethod
    def _neighbor_triple(item: Any) -> Optional[Tuple[str, float, str]]:
        pair = _PrunedHippoRAGNeo4jDirectedCacheMixin._neighbor_pair(item)
        if pair is None:
            return None
        node_id, weight = pair
        relation_type = ""
        if isinstance(item, tuple) and len(item) >= 3 and item[2] is not None:
            relation_type = str(item[2])
        return node_id, weight, relation_type

    @staticmethod
    def _should_add_reverse_edge_for_predicate(
        predicate: str,
        *,
        direction_policy: str,
        directed_relations: Set[str],
        direction_insensitive_relations: Set[str],
    ) -> bool:
        """Decide whether to add a reverse edge for an Entity→Entity fact edge."""

        policy = str(direction_policy or "whitelist").strip().lower()
        pred = str(predicate or "").strip()
        if not pred:
            return False
        if policy == "blacklist":
            return pred in direction_insensitive_relations
        return pred not in directed_relations

    def _directed_fact_cache_for_owner(self, owner_key: str) -> Dict[str, List[Tuple[str, float, str]]]:
        cache = getattr(self, "_directed_fact_cache", None)
        if cache is None or not isinstance(cache, dict):
            cache = {}
            setattr(self, "_directed_fact_cache", cache)
        owner_cache = cache.get(owner_key)
        if owner_cache is None or not isinstance(owner_cache, dict):
            owner_cache = {}
            cache[owner_key] = owner_cache
        return owner_cache

    def _load_directed_fact_cache(self, *, owner_id: Optional[Any], force_reload: bool = False) -> None:
        """Load directed Entity→Entity fact edges (RELATES_TO) into memory for direction-aware PPR."""

        with self.read_lock():
            if not (getattr(self, "_cache_loaded", False) and getattr(self, "_graph_cache", None)):
                self._load_graph_cache()

        owner_key = self._owner_key(owner_id) if owner_id is not None else None
        if owner_key is None:
            return

        loaded_key = getattr(self, "_directed_fact_cache_loaded_key", None)
        if loaded_key == owner_key and not force_reload:
            return

        from core.knowledge_graph.schema import normalize_relation_token

        global_owner = getattr(self, "OWNER_GLOBAL_KEY", "__GLOBAL__")
        params: Dict[str, Any] = {"global_owner": global_owner, "owner_id": owner_key}
        rows = self._execute_query(
            """
            MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
            WHERE COALESCE(e1.owner_id, $global_owner) = $owner_id
              AND COALESCE(e2.owner_id, $global_owner) = $owner_id
              AND COALESCE(r.owner_id, $global_owner) = $owner_id
            RETURN e1.entity_id AS source_id,
                   e2.entity_id AS target_id,
                   COALESCE(r.weight, 1.0) AS weight,
                   r.predicate AS predicate
            """,
            params,
        )

        owner_cache: Dict[str, List[Tuple[str, float, str]]] = {}
        edge_count = 0
        for record in rows or []:
            src = str(record.get("source_id") or "").strip()
            dst = str(record.get("target_id") or "").strip()
            if not src or not dst:
                continue
            pred = str(record.get("predicate") or "").strip()
            pred_norm = normalize_relation_token(pred) if pred else ""
            if not pred_norm:
                continue
            try:
                w = float(record.get("weight") or 1.0)
            except (TypeError, ValueError):
                w = 1.0
            owner_cache.setdefault(src, []).append((dst, w, pred_norm))
            edge_count += 1

        cache = self._directed_fact_cache_for_owner(owner_key)
        cache.clear()
        cache.update(owner_cache)
        setattr(self, "_directed_fact_cache_loaded_key", owner_key)
        logger.info("Loaded directed fact cache for owner %s: edges=%d nodes=%d", owner_key, edge_count, len(cache))

    def extract_subgraph_from_cache_for_ppr_directed(
        self,
        start_node_ids: List[str],
        *,
        owner_id: Optional[Any] = None,
        max_hops: int = 2,
        max_nodes: int = 2000,
        directed_policy: str = "whitelist",
        directed_relations: Optional[Set[str]] = None,
        direction_insensitive_relations: Optional[Set[str]] = None,
    ) -> Tuple[ig.Graph, Dict[str, int], Dict[int, str]]:
        """Extract a directed subgraph for push-PPR based on the directed cache."""

        self._load_directed_fact_cache(owner_id=owner_id, force_reload=False)
        owner_key = self._owner_key(owner_id) if owner_id is not None else None
        if owner_key is None:
            return ig.Graph(directed=True), {}, {}
        cache = self._directed_fact_cache_for_owner(owner_key)
        if not cache:
            return ig.Graph(directed=True), {}, {}

        directed_relations = directed_relations or set()
        direction_insensitive_relations = direction_insensitive_relations or set()
        max_hops = max(0, int(max_hops))
        max_nodes = max(1, int(max_nodes))

        visited: set[str] = set()
        frontier: list[str] = [str(node).strip() for node in start_node_ids if str(node).strip()]
        edges: list[tuple[str, str, float]] = []

        for _ in range(max_hops + 1):
            if not frontier or len(visited) >= max_nodes:
                break
            next_frontier: list[str] = []
            for src in frontier:
                if src in visited:
                    continue
                visited.add(src)
                for dst, w, pred in cache.get(src, []) or []:
                    dst_token = str(dst).strip()
                    if not dst_token:
                        continue
                    edges.append((src, dst_token, float(w)))
                    if self._should_add_reverse_edge_for_predicate(
                        pred,
                        direction_policy=directed_policy,
                        directed_relations=directed_relations,
                        direction_insensitive_relations=direction_insensitive_relations,
                    ):
                        edges.append((dst_token, src, float(w)))
                    if dst_token not in visited:
                        next_frontier.append(dst_token)
                        if len(visited) + len(next_frontier) >= max_nodes:
                            break
                if len(visited) + len(next_frontier) >= max_nodes:
                    break
            frontier = next_frontier

        node_index: Dict[str, int] = {}
        index_node: Dict[int, str] = {}
        ordered: list[str] = []
        for src, dst, _w in edges:
            if src not in node_index:
                node_index[src] = len(ordered)
                ordered.append(src)
            if dst not in node_index:
                node_index[dst] = len(ordered)
                ordered.append(dst)
        for idx, node in enumerate(ordered):
            index_node[idx] = node

        graph = ig.Graph(directed=True)
        graph.add_vertices(len(ordered))
        if edges:
            graph.add_edges([(node_index[src], node_index[dst]) for src, dst, _w in edges])
            graph.es["weight"] = [w for _src, _dst, w in edges]
        return graph, node_index, index_node

