"""Think-on-Graph inspired beam search tool."""
import json
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.deepsearch.utils.file_scope import resolve_file_scope

from ..base import (
    GraphTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
    build_input_schema,
    call_llm_async,
    extract_json_from_text,
    safe_json_loads,
)
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_CHAIN_TRAVERSE, REQUIRES_LLM, SCOPE_OWNER
from config.core.deepsearch.tool_defaults import (
    BEAM_SEARCH_DEFAULT_BEAM_SIZE,
    BEAM_SEARCH_DEFAULT_MAX_DEPTH,
    BEAM_SEARCH_DEFAULT_SEED_ENTITY_TOP_K,
    BEAM_SEARCH_DEFAULT_TEMPERATURE,
    BEAM_SEARCH_SEED_EXTRACT_MAX_TOKENS,
    BEAM_SEARCH_SEED_EXTRACT_TEMPERATURE,
)
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from core.prompts.deepsearch import SEARCH_ENTITY_EXTRACT_PROMPT_EN
from core.deepsearch.utils.llm_json import call_llm_json_with_retry
from core.prompts.deepsearch.heavy_tools import BEAM_SEARCH_RERANK_SYSTEM_PROMPT_V1_EN


class BeamSearchTool(GraphTool):
    """Executes ToG-style beam search to discover promising graph reasoning paths."""

    descriptor = ToolDescriptor(
        name="graph.beam_search",
        channel="graph",
        description="Think-on-Graph style beam search that enumerates candidate KG paths, scores them, and "
        "returns ranked path summaries plus supporting chunks before committing to longer reasoning chains.",
        speed="slow",
        cost="high",
        strategy_tags=("beam_search", "tog", "graph_reasoning", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_CHAIN_TRAVERSE, REQUIRES_LLM),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.explore.beam_search",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional seed entities for initializing the beam.",
                },
                "seed_entity_top_k": {
                    "type": "integer",
                    "description": "Max entities to extract with LLM when no seeds are provided.",
                    "minimum": 0,
                },
                "beam_size": {"type": "integer", "description": "Override default beam size."},
                "max_depth": {"type": "integer", "description": "Override default reasoning depth."},
            }
        ),
        example_args={
            "question": "Explain the chain of command that connects OpenAI to Microsoft.",
            "plan_step": "plan_beam",
            "extra": {"seed_entities": ["OpenAI", "Microsoft"], "beam_size": 3, "max_depth": 3},
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        beam_size: int = BEAM_SEARCH_DEFAULT_BEAM_SIZE,
        max_depth: int = BEAM_SEARCH_DEFAULT_MAX_DEPTH,
        temperature: float = BEAM_SEARCH_DEFAULT_TEMPERATURE,
        seed_entity_top_k: int = BEAM_SEARCH_DEFAULT_SEED_ENTITY_TOP_K,
    ):
        if llm_connector is None:
            raise ValueError("BeamSearchTool requires an LLM connector (no fallback ranking).")
        self.llm_connector = llm_connector
        self.beam_size = max(1, beam_size)
        self.max_depth = max(1, max_depth)
        self.temperature = temperature
        self.seed_entity_top_k = max(0, int(seed_entity_top_k))

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        beam_size = int(request.extra.get("beam_size") or self.beam_size)
        max_depth = int(request.extra.get("max_depth") or self.max_depth)
        seeds = await self._seed_entities(request)
        file_scope = resolve_file_scope(
            extra=request.extra,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
            question=request.question,
        )

        query_options: Dict[str, Any] = {"export_subgraph": True}
        if file_scope.enabled:
            # File-scoped beam search: use the adapter's file_scope filtering to avoid drifting to unrelated files.
            query_options["file_scope"] = file_scope.as_dict()

        async with adapter_locked(adapter):
            traversal = await adapter.chain_traverse(
                {
                    "strategy": "beam_search",
                    "question": request.question,
                    "beam_size": beam_size,
                    "max_depth": max_depth,
                    "seed_entities": seeds,
                    "query_options": query_options,
                },
                access_scope=request.access_scope,
            )
        paths = self._normalize_paths(traversal.get("paths"))
        if not paths:
            paths = self._paths_from_chunks(traversal, seeds)
        if not paths:
            diagnostics = {
                "beam_size": beam_size,
                "max_depth": max_depth,
                "file_scope": file_scope.as_dict() if file_scope.enabled else None,
                "path_count": 0,
                "selected_paths": 0,
                "used_llm_rerank": False,
                "strategy": traversal.get("strategy") or "beam_search",
                "note": "beam_search returned no paths; degraded to empty result",
            }
            return ToolResult(
                summary="Beam search executed but no promising paths emerged.",
                diagnostics=diagnostics,
                think_notes=[],
            )

        ranked_paths = await self._rank_paths(request, paths, beam_size)
        evidences = self._paths_to_evidences(ranked_paths, tool_name=self.descriptor.name, plan_step=request.plan_step)
        diagnostics = {
            "beam_size": beam_size,
            "max_depth": max_depth,
            "file_scope": file_scope.as_dict() if file_scope.enabled else None,
            "path_count": len(paths),
            "selected_paths": len(ranked_paths),
            "used_llm_rerank": True,
        }
        if traversal.get("strategy"):
            diagnostics["strategy"] = traversal["strategy"]
        summary = self._build_summary(ranked_paths)
        think_notes = self._build_think_notes(request, ranked_paths, beam_size, max_depth)
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics, think_notes=think_notes)

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("BeamSearchTool requires a GraphDeepSearchAdapter instance")
        return adapter

    async def _seed_entities(self, request: ToolRunRequest) -> List[str]:
        seeds: List[str] = []
        seed_limit = self.seed_entity_top_k
        try:
            override = request.extra.get("seed_entity_top_k")
            if override is not None:
                seed_limit = int(override)
        except Exception:
            seed_limit = self.seed_entity_top_k
        seed_limit = max(0, int(seed_limit))
        extra_seeds = request.extra.get("seed_entities")
        if isinstance(extra_seeds, list):
            seeds.extend(str(entity).strip() for entity in extra_seeds if str(entity).strip())
        if request.graph_context:
            seeds.extend(request.graph_context.seed_entities)
            context_meta = request.graph_context.metadata or {}
            extra_context_seeds = context_meta.get("seed_entities") or []
            if isinstance(extra_context_seeds, list):
                seeds.extend(str(item) for item in extra_context_seeds if str(item).strip())
        for evidence in request.context_evidences:
            provenance = evidence.provenance or {}
            for key in ("entity", "head", "tail"):
                candidate = provenance.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    seeds.append(candidate.strip())
        if not seeds:
            llm_seeds = await self._extract_seed_entities_with_llm(request.question, seed_limit)
            seeds.extend(llm_seeds)
        return self._deduplicate(seeds)

    async def _extract_seed_entities_with_llm(self, question: str, limit: int) -> List[str]:
        if limit <= 0:
            return []
        messages = [
            {"role": "system", "content": SEARCH_ENTITY_EXTRACT_PROMPT_EN},
            {"role": "user", "content": f"Query: {question}"},
        ]
        kwargs: Dict[str, Any] = {
            "temperature": float(BEAM_SEARCH_SEED_EXTRACT_TEMPERATURE),
            "max_tokens": int(BEAM_SEARCH_SEED_EXTRACT_MAX_TOKENS),
        }
        low_cost = self._low_cost_model_name(self.llm_connector)
        if low_cost:
            kwargs["model"] = low_cost
        payload = await call_llm_json_with_retry(
            llm_connector=self.llm_connector,
            messages=messages,
            expected="object",
            temperature=float(kwargs["temperature"]),
            max_tokens=int(kwargs["max_tokens"]),
            attempts=None,
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
        return self._deduplicate(results)

    async def _rank_paths(self, request: ToolRunRequest, paths: List[Dict[str, Any]], beam_size: int) -> List[Dict[str, Any]]:
        if not paths:
            return []
        limited = sorted(paths, key=lambda item: item.get("score", 0.0), reverse=True)[: max(beam_size * 2, beam_size)]
        llm_scores = await self._llm_rerank(request, limited)

        def _sort_key(item: Dict[str, Any]) -> float:
            return llm_scores.get(item["path_id"], item.get("score", 0.0))

        ranked = sorted(limited, key=_sort_key, reverse=True)
        return ranked[:beam_size]

    async def _llm_rerank(self, request: ToolRunRequest, paths: List[Dict[str, Any]]) -> Dict[str, float]:
        payload = {
            "question": request.question,
            "paths": [
                {
                    "path_id": path["path_id"],
                    "summary": self._path_summary(path),
                    "score": path.get("score", 0.0),
                }
                for path in paths
            ],
        }
        messages = [
            {
                "role": "system",
                "content": BEAM_SEARCH_RERANK_SYSTEM_PROMPT_V1_EN,
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = await call_llm_async(
            self.llm_connector,
            messages,
            temperature=self.temperature,
        )
        data = safe_json_loads(response, expected="list") or []
        scores: Dict[str, float] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or "path_id" not in item:
                    continue
                try:
                    score = float(item.get("score", 0))
                except (TypeError, ValueError):
                    score = 0.0
                scores[str(item["path_id"])] = score
        return scores

    def _normalize_paths(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            if not item.get("path_id"):
                raise ValueError("Beam search path is missing required path_id")
            path_id = str(item["path_id"])
            nodes = item.get("nodes") or item.get("entities") or []
            if isinstance(nodes, list):
                nodes = [str(node) for node in nodes if str(node).strip()]
            else:
                nodes = [str(nodes)]
            triples = item.get("triples") or []
            if not isinstance(triples, list):
                triples = []
            normalized.append(
                {
                    "path_id": path_id,
                    "nodes": nodes,
                    "triples": triples,
                    "score": float(item.get("score", 0.0)),
                    "summary": item.get("summary"),
                }
            )
        return normalized

    def _paths_from_chunks(self, payload: Dict[str, Any], seeds: List[str]) -> List[Dict[str, Any]]:
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[: self.beam_size * self.max_depth]):
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            nodes = list(seeds[:1])
            target_node = chunk.get("metadata", {}).get("entity") or chunk.get("metadata", {}).get("title")
            if target_node:
                nodes.append(str(target_node))
            nodes.append(chunk.get("chunk_id") or f"chunk-{idx}")
            normalized.append(
                {
                    "path_id": f"chunk-{idx}",
                    "nodes": self._deduplicate(nodes) or ["unknown"],
                    "triples": [],
                    "score": 0.2,
                    "summary": content,
                }
            )
        return normalized

    def _paths_to_evidences(
        self,
        paths: Iterable[Dict[str, Any]],
        *,
        tool_name: str,
        plan_step: str | None,
    ) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for rank, path in enumerate(paths):
            nodes = path.get("nodes") or []
            content = path.get("summary") or self._path_summary(path)
            evidences.append(
                EvidenceChunk(
                    chunk_id=derived_chunk_id(
                        tool_name=tool_name,
                        plan_step=plan_step,
                        label=f"beam_path_{rank}",
                        content=str(content),
                    ),
                    source=tool_name,
                    content=content,
                    kind=EVIDENCE_KIND_DERIVED,
                    score=path.get("score"),
                    provenance={
                        "path": nodes,
                        "triples": path.get("triples", []),
                        "beam_rank": rank,
                    },
                )
            )
        return evidences

    def _build_summary(self, paths: List[Dict[str, Any]]) -> str:
        if not paths:
            return "Beam search executed but no promising paths emerged."
        lines = ["Beam search selected the following reasoning paths:"]
        for idx, path in enumerate(paths):
            nodes = path.get("nodes") or []
            summary = self._path_summary(path)
            lines.append(f"{idx + 1}. {' -> '.join(nodes) or 'unknown path'} | {summary}")
        return "\n".join(lines)

    def _build_think_notes(
        self,
        request: ToolRunRequest,
        paths: List[Dict[str, Any]],
        beam_size: int,
        max_depth: int,
    ) -> List[ThinkNote]:
        if not paths:
            return []
        coverage = min(1.0, len(paths) / max(1, beam_size))
        reasoning = "Beam search highlighted candidate reasoning chains for downstream tools."
        next_actions = [
            "Probe highest-ranked path with deterministic tools.",
            "Escalate to report composer once coverage converges.",
        ]
        note = ThinkNote(
            plan_step_id=request.plan_step,
            reasoning=reasoning,
            confidence_delta=coverage - 0.5,
            coverage_delta=coverage,
            next_actions=next_actions,
            metadata={
                "selected_paths": len(paths),
                "beam_size": beam_size,
                "max_depth": max_depth,
            },
        )
        return [note]

    def _path_summary(self, path: Dict[str, Any]) -> str:
        summary = path.get("summary")
        if summary:
            return str(summary)
        nodes = path.get("nodes") or []
        return " -> ".join(str(node) for node in nodes if str(node).strip())

    @staticmethod
    def _low_cost_model_name(llm_connector) -> Optional[str]:
        cfg = getattr(llm_connector, "config", None)
        token = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
        token = str(token or "").strip()
        return token or None

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in items:
            token = str(item).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered
