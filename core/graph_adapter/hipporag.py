"""HippoRAG-backed GraphDeepSearchAdapter implementation."""
import asyncio
import json
import logging
import os
import re
from collections import deque
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from encapsulation.data_model.schema import Chunk
from encapsulation.database.utils.graph_export_utils import GraphExporter
from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j

from config.output_limits import DEEPSEARCH_GRAPH_EXPORT_MAX_EDGES
from core.graph_adapter.base import (
    GraphAccessScope,
    GraphAdapterCapability,
    GraphAdapterMetadata,
    GraphDeepSearchAdapter,
)
from core.graph_adapter.registry import register_adapter
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.utils.file_scope import FileScope, chunk_in_scope
from core.prompts import build_file_scope_xlang_rewrite_prompt

logger = logging.getLogger(__name__)


class HippoRAGGraphAdapter(GraphDeepSearchAdapter):
    """Graph adapter that wraps the Pruned HippoRAG retriever implementations."""

    supports_concurrent_calls = True

    def __init__(
        self,
        retriever,
        *,
        default_top_k: int = 5,
        summary_max_chunks: int = 3,
        summary_char_limit: int = 320,
        semantic_score_threshold: float = 0.05,
        adapter_version: str = "hipporag-adapter.v1",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.retriever = retriever
        self.default_top_k = max(1, default_top_k)
        self.summary_max_chunks = max(1, summary_max_chunks)
        self.summary_char_limit = max(120, summary_char_limit)
        self.semantic_score_threshold = max(0.0, min(1.0, semantic_score_threshold))
        self.adapter_version = adapter_version
        self.extra_metadata = extra_metadata or {}

    async def prepare(self, question: str, *, access_scope: Optional[GraphAccessScope] = None) -> None:
        """HippoRAG adapters do not require heavy warmup."""

        scope = self._scope_token(access_scope)
        logger.debug("HippoRAG adapter prepared for scope=%s question=%s", scope, question)

    async def aquery_subgraph(
        self,
        query: str,
        *,
        channel: str = "graph",
        access_scope: Optional[GraphAccessScope] = None,
        query_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run HippoRAG retrieval and export the resulting subgraph."""

        scope_token = self._scope_token(access_scope)
        if scope_token is None:
            logger.warning("HippoRAG adapter requires an access scope; returning empty payload for query %s", query)
            return self._empty_payload(query)

        options = dict(query_options) if isinstance(query_options, Mapping) else {}
        return await asyncio.to_thread(self._aquery_subgraph_sync, query, channel, scope_token, options)

    async def context_filter(
        self,
        data: Mapping[str, Any],
        *,
        filter_type: str = "semantic",
        access_scope: Optional[GraphAccessScope] = None,
    ) -> Mapping[str, Any]:
        """Apply very lightweight filtering until the dedicated GraphReasoning loop takes over."""

        if not data:
            return data
        nodes = list(data.get("nodes", []))
        edges = list(data.get("edges", []))
        metadata = dict(data.get("metadata", {}))
        ppr_scores: Dict[str, float] = metadata.get("node_ppr_scores", {})

        if filter_type == "semantic" and ppr_scores:
            threshold = self.semantic_score_threshold
            keep_ids = {node_id for node_id, score in ppr_scores.items() if float(score) >= threshold}
            nodes = [node for node in nodes if str(node.get("id")) in keep_ids]
            edges = [
                edge
                for edge in edges
                if str(edge.get("source")) in keep_ids and str(edge.get("target")) in keep_ids
            ]
        elif filter_type == "relational":
            edges = [edge for edge in edges if edge.get("relation")]

        filtered = dict(data)
        filtered["nodes"] = nodes
        filtered["edges"] = edges
        filtered["metadata"] = metadata
        return filtered

    async def summarize(
        self,
        channel: str,
        data: Mapping[str, Any],
        *,
        access_scope: Optional[GraphAccessScope] = None,
    ) -> str:
        """Produce a deterministic textual summary using the retrieved chunks."""

        chunks = data.get("chunks", []) if isinstance(data, Mapping) else []
        if not chunks:
            return "No supporting graph evidence was retrieved."
        lines: List[str] = []
        for idx, chunk in enumerate(chunks[: self.summary_max_chunks], start=1):
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            snippet = content[: self.summary_char_limit]
            if len(content) > self.summary_char_limit:
                snippet = snippet.rstrip() + "..."
            lines.append(f"{idx}. {snippet}")
        return "\n".join(lines) if lines else "Graph evidence chunks are empty."

    async def chain_traverse(
        self,
        strategy: Mapping[str, Any],
        *,
        access_scope: Optional[GraphAccessScope] = None,
    ) -> Mapping[str, Any]:
        """Run lightweight traversal routines for DeepSearch fast tools.

        Contract:
        - Always returns at least {strategy, hops, visited, scope}
        - For strategy == "ppr_prefetch": may include "paths": [{path_id, nodes, score}]
        - For strategy == "bridge_lookup": may include "bridges": [{head, relation, tail, score}]
        """

        if not isinstance(strategy, Mapping):
            scope_token = self._scope_token(access_scope)
            return {"strategy": "ppr_chain", "hops": 0, "visited": [], "scope": scope_token}

        scope_token = self._scope_token(access_scope)
        if scope_token is None:
            return {"strategy": str(strategy.get("strategy") or "ppr_chain"), "hops": 0, "visited": [], "scope": None}

        strategy_name = str(strategy.get("strategy") or "ppr_chain").strip() or "ppr_chain"
        seeds: List[str] = []
        seed_entities = strategy.get("seed_entities") or []
        if isinstance(seed_entities, list):
            seeds = [str(item).strip() for item in seed_entities if str(item).strip()]

        if not seeds:
            seed_chunk = str(strategy.get("seed_chunk") or strategy.get("seed_chunk_id") or "").strip()
            if seed_chunk:
                derived = await asyncio.to_thread(self._seed_entities_from_chunk_sync, seed_chunk, scope_token)
                if derived:
                    seeds = derived

        max_depth = strategy.get("max_depth")
        try:
            max_depth_int = int(max_depth) if max_depth is not None else 1
        except (TypeError, ValueError):
            max_depth_int = 1
        max_depth_int = max(1, max_depth_int)

        visited = seeds[: max_depth_int] if seeds else []
        base = {"strategy": strategy_name, "hops": max_depth_int, "visited": visited, "scope": scope_token}

        if strategy_name not in {"ppr_prefetch", "bridge_lookup"}:
            return base

        question = str(strategy.get("question") or "").strip()
        query = question or (" ; ".join(seeds) if seeds else "")
        if not query:
            return base

        payload = await self.aquery_subgraph(
            query,
            channel="graph",
            access_scope=access_scope,
            query_options={"export_subgraph": True},
        )
        edges = payload.get("edges") if isinstance(payload, dict) else None
        if not isinstance(edges, list) or not edges:
            return base

        entity_edges = self._entity_relation_edges(edges)
        if not entity_edges:
            return base

        if strategy_name == "bridge_lookup":
            bridges = self._extract_bridges(entity_edges, seeds=seeds)
            merged = dict(base)
            merged["bridges"] = bridges
            merged["hops"] = min(max_depth_int, 3) if max_depth_int else 1
            return merged

        max_paths = strategy.get("max_paths")
        try:
            max_paths_int = int(max_paths) if max_paths is not None else 3
        except (TypeError, ValueError):
            max_paths_int = 3
        max_paths_int = max(1, min(max_paths_int, 6))
        paths = self._prefetch_paths(entity_edges, seeds=seeds, max_depth=max_depth_int, max_paths=max_paths_int)
        merged = dict(base)
        merged["paths"] = paths
        return merged

    def _seed_entities_from_chunk_sync(self, chunk_id: str, owner_id: str) -> List[str]:
        graph_store = getattr(self.retriever, "graph_store", None)
        if graph_store is None or not hasattr(graph_store, "_execute_query"):
            return []
        query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        WHERE c.owner_id = $owner_id
        MATCH (c)-[:MENTIONS]->(e:Entity)
        WHERE e.owner_id = $owner_id
        RETURN e.entity_name AS name
        ORDER BY name
        LIMIT 12
        """
        try:
            rows = graph_store._execute_query(query, {"chunk_id": str(chunk_id), "owner_id": str(owner_id)})
        except Exception:
            return []
        names: List[str] = []
        seen: set[str] = set()
        for row in rows or []:
            name = str((row or {}).get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def metadata(self) -> GraphAdapterMetadata:
        """Publish capability metadata for observability dashboards."""

        retrieval_capability = GraphAdapterCapability(
            name="hipporag_retrieval",
            modes=("semantic", "relational"),
            metrics={"default_top_k": self.default_top_k},
        )
        chain_capability = GraphAdapterCapability(
            name="chain_of_exploration",
            modes=("ppr_chain", "ppr_prefetch", "bridge_lookup"),
            metrics={"chain_depth": self.summary_max_chunks},
        )
        concurrency_capability = GraphAdapterCapability(
            name="concurrency",
            modes=("thread",),
            metrics={"concurrency_safe": True, "model": "threadpool+rwlock"},
        )
        return GraphAdapterMetadata(
            adapter_name="hipporag",
            graph_type=self.extra_metadata.get("graph_type", "hipporag"),
            version=self.adapter_version,
            owner=self.extra_metadata.get("owner"),
            capabilities=(retrieval_capability, chain_capability, concurrency_capability),
            domain_tags=tuple(self.extra_metadata.get("domain_tags", [])),
            config_fingerprint=self.extra_metadata.get("config_fingerprint"),
        )

    @staticmethod
    def _entity_relation_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            relation = str(edge.get("relation") or "").strip()
            if not relation or relation == "mentions":
                continue
            head = str(edge.get("source") or "").strip()
            tail = str(edge.get("target") or "").strip()
            if not head or not tail:
                continue
            relations.append({"head": head, "relation": relation, "tail": tail})
        return relations

    @staticmethod
    def _extract_bridges(relations: List[Dict[str, str]], *, seeds: List[str]) -> List[Dict[str, Any]]:
        if not seeds:
            return []
        seed_set = {seed for seed in seeds if seed}
        seed_lower = {seed.lower() for seed in seed_set}
        bridges: List[Dict[str, Any]] = []
        for rel in relations:
            head = rel["head"]
            tail = rel["tail"]
            if head in seed_set or tail in seed_set or head.lower() in seed_lower or tail.lower() in seed_lower:
                bridges.append({"head": head, "relation": rel["relation"], "tail": tail, "score": 1.0})
                if len(bridges) >= 12:
                    break
        return bridges

    @staticmethod
    def _prefetch_paths(
        relations: List[Dict[str, str]],
        *,
        seeds: List[str],
        max_depth: int,
        max_paths: int,
    ) -> List[Dict[str, Any]]:
        if len(seeds) < 2:
            return []

        adjacency: Dict[str, List[str]] = {}
        for rel in relations:
            head = rel["head"]
            tail = rel["tail"]
            adjacency.setdefault(head, []).append(tail)
            adjacency.setdefault(tail, []).append(head)

        paths: List[Dict[str, Any]] = []
        for source, target in combinations(seeds[:6], 2):
            candidate = HippoRAGGraphAdapter._shortest_path(adjacency, source, target, max_depth=max_depth)
            if not candidate:
                continue
            content = " -> ".join(candidate)
            path_id = hashed_chunk_id(source=f"{source}->{target}", content=content, prefix="path")
            score = 1.0 / max(1, (len(candidate) - 1))
            paths.append({"path_id": path_id, "nodes": candidate, "score": score})
            if len(paths) >= max_paths:
                break
        return paths

    @staticmethod
    def _shortest_path(adjacency: Dict[str, List[str]], source: str, target: str, *, max_depth: int) -> List[str]:
        if source == target:
            return [source]
        if source not in adjacency or target not in adjacency:
            return []

        seen: set[str] = {source}
        queue: deque[tuple[str, List[str]]] = deque([(source, [source])])
        max_hops = max(1, int(max_depth))

        while queue:
            node, path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for nxt in adjacency.get(node, []):
                if nxt in seen:
                    continue
                next_path = path + [nxt]
                if nxt == target:
                    return next_path
                seen.add(nxt)
                queue.append((nxt, next_path))
        return []

    def _aquery_subgraph_sync(
        self,
        query: str,
        channel: str,
        scope_token: str,
        query_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunks, subgraph_info, scope_diag = self._retrieve_chunks_sync(query, scope_token, query_options=query_options)
        return self._build_graph_payload(
            query,
            channel,
            chunks,
            scope_token,
            subgraph_info_override=subgraph_info,
            scope_diagnostics=scope_diag,
            query_options=query_options,
        )

    def _retrieve_chunks_sync(
        self,
        query: str,
        owner_token: str,
        *,
        query_options: Optional[Mapping[str, Any]] = None,
    ) -> tuple[List[Chunk], Optional[Dict[str, Any]], Dict[str, Any]]:
        """Execute synchronous HippoRAG retrieval (invoked in a worker thread)."""

        top_k_raw = None
        if isinstance(query_options, Mapping):
            top_k_raw = query_options.get("top_k")
        try:
            top_k = int(top_k_raw) if top_k_raw is not None else int(self.default_top_k)
        except (TypeError, ValueError):
            top_k = int(self.default_top_k)
        top_k = max(1, top_k)

        file_scope = self._coerce_file_scope(query_options)
        requested_k = top_k
        fetch_k = requested_k
        if file_scope.enabled:
            # Over-fetch to compensate for post-filtering; keep a small cap.
            fetch_k = min(max(requested_k * 5, requested_k), 100)

        kwargs = {
            "k": fetch_k,
            "owner_id": owner_token,
            "return_subgraph_info": True,
        }
        raw_chunks: List[Chunk] = self.retriever.invoke(query, **kwargs)
        subgraph_info = self._extract_subgraph_info(raw_chunks)
        if not file_scope.enabled:
            return raw_chunks[:requested_k], subgraph_info, {"file_scope_applied": False}

        filtered: List[Chunk] = []
        dropped = 0
        for chunk in raw_chunks:
            meta = getattr(chunk, "metadata", None) or {}
            if chunk_in_scope(chunk_metadata=meta, scope=file_scope):
                filtered.append(chunk)
            else:
                dropped += 1
        # When file_scope is enabled, global retrieval + post-filtering can return 0 results,
        # especially for cross-language queries. In that case, attempt a single query rewrite
        # using the retriever's LLM client (when available) to bridge the language gap.
        xlang_diag: Dict[str, Any] = {}
        if not filtered[:requested_k] and self._file_scope_xlang_retry_enabled():
            rewritten = self._maybe_rewrite_query_for_scope(query)
            if rewritten and rewritten != query:
                xlang_diag["xlang_retry"] = True
                xlang_diag["rewritten_query_preview"] = rewritten[:200]
                try:
                    raw_retry: List[Chunk] = self.retriever.invoke(rewritten, **kwargs)
                    retry_info = self._extract_subgraph_info(raw_retry)
                    retry_filtered: List[Chunk] = []
                    retry_dropped = 0
                    for chunk in raw_retry:
                        meta = getattr(chunk, "metadata", None) or {}
                        if chunk_in_scope(chunk_metadata=meta, scope=file_scope):
                            retry_filtered.append(chunk)
                        else:
                            retry_dropped += 1
                    if retry_filtered:
                        return (
                            retry_filtered[:requested_k],
                            retry_info,
                            {
                                "file_scope_applied": True,
                                "file_scope": file_scope.as_dict(),
                                "requested_top_k": requested_k,
                                "fetch_top_k": fetch_k,
                                "kept_in_scope": len(retry_filtered[:requested_k]),
                                "dropped_out_of_scope": retry_dropped,
                                "input_chunk_count": len(raw_retry),
                                **xlang_diag,
                            },
                        )
                    xlang_diag["xlang_retry_no_hits"] = True
                except Exception as exc:
                    xlang_diag["xlang_retry_error"] = str(exc)
        return (
            filtered[:requested_k],
            subgraph_info,
            {
                "file_scope_applied": True,
                "file_scope": file_scope.as_dict(),
                "requested_top_k": requested_k,
                "fetch_top_k": fetch_k,
                "kept_in_scope": len(filtered[:requested_k]),
                "dropped_out_of_scope": dropped,
                "input_chunk_count": len(raw_chunks),
                **xlang_diag,
            },
        )

    def _build_graph_payload(
        self,
        query: str,
        channel: str,
        chunks: List[Chunk],
        scope_token: str,
        *,
        subgraph_info_override: Optional[Dict[str, Any]] = None,
        scope_diagnostics: Optional[Dict[str, Any]] = None,
        query_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert retriever outputs into a normalized structure consumed by DeepSearch."""

        chunk_entries = [self._chunk_to_dict(chunk) for chunk in chunks]
        subgraph_info = subgraph_info_override if subgraph_info_override is not None else self._extract_subgraph_info(chunks)
        if isinstance(subgraph_info, dict) and scope_token:
            subgraph_info.setdefault("owner_scope", scope_token)
        # Exporting the full subgraph (nodes/edges) can be expensive on Neo4j. Default to disabled unless explicitly
        # requested by callers that need visualization/triple context (e.g. graph_adapter.query traversal).
        export_subgraph = False
        if isinstance(query_options, Mapping) and "export_subgraph" in query_options:
            export_subgraph = bool(query_options.get("export_subgraph"))
        exported = (
            self._export_subgraph(subgraph_info)
            if (subgraph_info and export_subgraph)
            else {"nodes": [], "edges": [], "chunks": [], "metadata": {}}
        )

        metadata = {
            "adapter": "hipporag",
            "channel": channel,
            "owner_scope": scope_token,
            "subgraph_info": subgraph_info or {},
            "graph_export_metadata": exported.get("metadata", {}),
            "node_ppr_scores": (subgraph_info or {}).get("node_ppr_scores", {}),
        }
        if scope_diagnostics:
            metadata["scope_diagnostics"] = scope_diagnostics
        metadata["export_subgraph"] = export_subgraph

        return {
            "query": query,
            "channel": channel,
            "chunks": chunk_entries,
            "nodes": exported.get("nodes", []),
            "edges": exported.get("edges", []),
            "seed_entities": (subgraph_info or {}).get("seed_entity_ids", []),
            "metadata": metadata,
        }

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
        """Convert Chunk dataclass into a serializable dict."""

        chunk_id = getattr(chunk, "id", None) or chunk.metadata.get("chunk_id") if chunk.metadata else None
        return {
            "id": chunk_id,
            "content": getattr(chunk, "content", ""),
            "score": (chunk.metadata or {}).get("score"),
            "metadata": chunk.metadata or {},
        }

    @staticmethod
    def _extract_subgraph_info(chunks: List[Chunk]) -> Optional[Dict[str, Any]]:
        """Locate subgraph metadata stored by the retriever."""

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None)
            if metadata and "_subgraph_info" in metadata:
                return metadata["_subgraph_info"]
        return None

    @staticmethod
    def _coerce_file_scope(query_options: Optional[Mapping[str, Any]]) -> FileScope:
        if not isinstance(query_options, Mapping):
            return FileScope()
        raw = query_options.get("file_scope")
        if isinstance(raw, Mapping):
            return FileScope(
                file_ids=frozenset(str(x).strip() for x in (raw.get("file_ids") or []) if str(x).strip()),
                filename_contains=tuple(str(x).strip() for x in (raw.get("filename_contains") or []) if str(x).strip()),
                source=str(raw.get("source") or "adapter_query_options").strip() or "adapter_query_options",
            )
        return FileScope()

    @staticmethod
    def _file_scope_xlang_retry_enabled() -> bool:
        raw = os.getenv("DEEPSEARCH_FILE_SCOPE_XLANG_RETRY", "1")
        return str(raw or "").strip().lower() not in {"0", "false", "no", "off"}

    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    _ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")

    def _maybe_rewrite_query_for_scope(self, query: str) -> str | None:
        llm = getattr(self.retriever, "llm_client", None)
        chat = getattr(llm, "chat", None)
        if not callable(chat):
            return None

        text = str(query or "").strip()
        if not text:
            return None

        cjk = len(self._CJK_RE.findall(text))
        alpha = len(self._ASCII_ALPHA_RE.findall(text))
        total = max(1, len(text))
        cjk_ratio = cjk / total
        alpha_ratio = alpha / total

        direction: str | None = None
        if alpha_ratio >= 0.25 and cjk_ratio < 0.05:
            direction = "to_zh"
        elif cjk_ratio >= 0.15 and alpha_ratio < 0.08:
            direction = "to_en"
        else:
            return None

        prompt = build_file_scope_xlang_rewrite_prompt(query=text)
        try:
            raw = chat([{"role": "user", "content": prompt}])
        except Exception:
            return None

        payload = self._safe_json_loads(str(raw or ""))
        if not isinstance(payload, dict):
            return None

        zh_hans = str(payload.get("zh_hans") or "").strip()
        zh_hant = str(payload.get("zh_hant") or "").strip()
        en = str(payload.get("en") or "").strip()

        additions: List[str] = []
        if direction == "to_zh":
            if zh_hans:
                additions.append(zh_hans)
            if zh_hant and zh_hant != zh_hans:
                additions.append(zh_hant)
        elif direction == "to_en" and en:
            additions.append(en)

        if not additions:
            return None
        joined = "\n".join(additions)
        return f"{text}\n\n{joined}"

    @staticmethod
    def _safe_json_loads(raw: str) -> Any | None:
        text = str(raw or "").strip()
        if not text:
            return None
        # Remove ```json fences when present.
        if text.startswith("```"):
            # Strip the leading fence.
            if text.lower().startswith("```json"):
                text = text[len("```json") :].strip()
            else:
                text = text[len("```") :].strip()
            # Strip the trailing fence.
            if text.endswith("```"):
                text = text[: -len("```")].strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _export_subgraph(self, subgraph_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use the existing graph exporters to obtain nodes/edges for visualization."""

        graph_store = getattr(self.retriever, "graph_store", None)
        if graph_store is None:
            return {"nodes": [], "edges": [], "chunks": [], "metadata": {}}

        node_ids = subgraph_info.get("subgraph_nodes") or []
        seed_entities = set(subgraph_info.get("seed_entity_ids") or [])
        retrieved_chunk_ids = subgraph_info.get("retrieved_chunk_ids") or []
        node_scores = subgraph_info.get("node_ppr_scores") or {}

        try:
            read_lock = getattr(graph_store, "read_lock", None)
            lock_ctx = graph_store.read_lock() if callable(read_lock) else nullcontext()
            with lock_ctx:
                if graph_store.__class__.__name__ == "PrunedHippoRAGNeo4jStore":
                    return GraphExporterNeo4j.export_subgraph(
                        graph_store=graph_store,
                        subgraph_node_ids=set(str(node) for node in node_ids),
                        seed_entity_ids=set(str(entity) for entity in seed_entities),
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        node_ppr_scores=node_scores,
                        owner_id=subgraph_info.get("owner_id") or subgraph_info.get("owner_scope"),
                        owner_scope_label=subgraph_info.get("owner_scope"),
                        max_edges=DEEPSEARCH_GRAPH_EXPORT_MAX_EDGES,
                    )
                return GraphExporter.export_subgraph(
                    graph_store=graph_store,
                    subgraph_node_indices=set(node_ids),
                    seed_entity_ids=set(str(entity) for entity in seed_entities),
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    node_ppr_scores=node_scores,
                )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to export HippoRAG subgraph: %s", exc)
            return {"nodes": [], "edges": [], "chunks": [], "metadata": {}}

    @staticmethod
    def _scope_token(access_scope: Optional[GraphAccessScope]) -> Optional[str]:
        """Extract the opaque scope token used by HippoRAG retrievers."""

        if access_scope is None:
            return None
        token = access_scope.as_token()
        return token.strip() if isinstance(token, str) else token

    @staticmethod
    def _empty_payload(query: str) -> Dict[str, Any]:
        """Return a consistent empty payload for callers that need structured data."""

        return {
            "query": query,
            "channel": "graph",
            "chunks": [],
            "nodes": [],
            "edges": [],
            "seed_entities": [],
            "metadata": {"adapter": "hipporag"},
        }


def _build_retriever_from_payload(payload: Any):
    """Normalize retriever configurations provided via config dictionaries or paths."""

    if payload is None:
        raise ValueError("retriever_config is required for the HippoRAG adapter")

    if hasattr(payload, "retrieve"):
        return payload

    if hasattr(payload, "build") and callable(payload.build):
        return payload.build()

    if isinstance(payload, str):
        path = Path(payload)
        if not path.exists():
            raise ValueError(f"retriever config path {payload} does not exist")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if isinstance(payload, dict):
        cfg_type = payload.get("type")
        if cfg_type == "pruned_hipporag_retrieval":
            from config.core.retrieval.pruned_hipporag_config import PrunedHippoRAGRetrievalConfig

            return PrunedHippoRAGRetrievalConfig.model_validate(payload).build()
        if cfg_type == "pruned_hipporag_neo4j_retrieval":
            from config.core.retrieval.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jRetrievalConfig

            return PrunedHippoRAGNeo4jRetrievalConfig.model_validate(payload).build()

    raise ValueError("Unsupported retriever configuration for HippoRAG adapter")


def _hipporag_factory(**kwargs: Any) -> HippoRAGGraphAdapter:
    """Factory registered in the adapter registry."""

    retriever = kwargs.get("retriever")
    if retriever is None:
        retriever_config = kwargs.get("retriever_config")
        retriever = _build_retriever_from_payload(retriever_config)

    return HippoRAGGraphAdapter(
        retriever=retriever,
        default_top_k=int(kwargs.get("default_top_k", 5)),
        summary_max_chunks=int(kwargs.get("summary_max_chunks", 3)),
        summary_char_limit=int(kwargs.get("summary_char_limit", 320)),
        semantic_score_threshold=float(kwargs.get("semantic_score_threshold", 0.05)),
        adapter_version=str(kwargs.get("version", "hipporag-adapter.v1")),
        extra_metadata=kwargs.get("metadata"),
    )


register_adapter("hipporag", _hipporag_factory)

__all__ = ["HippoRAGGraphAdapter"]
