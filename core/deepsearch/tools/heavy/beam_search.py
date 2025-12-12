"""Think-on-Graph inspired beam search tool."""
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema, call_llm_async
from ..fast.pattern_probe import PatternProbeTool


class BeamSearchTool(GraphTool):
    """Executes ToG-style beam search to discover promising graph reasoning paths."""

    descriptor = ToolDescriptor(
        name="graph.beam_search",
        channel="graph",
        description="Think-on-Graph style beam search that enumerates candidate KG paths, scores them, and "
        "returns ranked path summaries plus supporting chunks before committing to longer reasoning chains.",
        speed="slow",
        cost="high",
        strategy_tags=("beam_search", "tog", "graph_reasoning"),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.heavy.beam_search",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional seed entities for initializing the beam.",
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
        beam_size: int = 3,
        max_depth: int = 3,
        temperature: float = 0.2,
    ):
        self.llm_connector = llm_connector
        self.beam_size = max(1, beam_size)
        self.max_depth = max(1, max_depth)
        self.temperature = temperature
        # Reuse deterministic tokenization from PatternProbe to mine candidate entities.
        self._pattern_probe = PatternProbeTool(max_terms=6)

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        beam_size = int(request.extra.get("beam_size") or self.beam_size)
        max_depth = int(request.extra.get("max_depth") or self.max_depth)
        seeds = self._seed_entities(request)

        traversal = await adapter.chain_traverse(
            {
                "strategy": "beam_search",
                "question": request.question,
                "beam_size": beam_size,
                "max_depth": max_depth,
                "seed_entities": seeds,
            },
            access_scope=request.access_scope,
        )
        paths = self._normalize_paths(traversal.get("paths"), fallback_prefix="beam")
        if not paths:
            fallback = await adapter.aquery_subgraph(
                request.question,
                channel="graph",
                access_scope=request.access_scope,
            )
            paths = self._paths_from_chunks(fallback, seeds)

        ranked_paths = await self._rank_paths(request, paths, beam_size)
        evidences = self._paths_to_evidences(ranked_paths, adapter.metadata().adapter_name)
        diagnostics = {
            "beam_size": beam_size,
            "max_depth": max_depth,
            "path_count": len(paths),
            "selected_paths": len(ranked_paths),
            "used_llm_rerank": bool(self.llm_connector),
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

    def _seed_entities(self, request: ToolRunRequest) -> List[str]:
        seeds: List[str] = []
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
            seeds.extend(self._extract_question_seeds(request))
        return self._deduplicate(seeds)

    def _extract_question_seeds(self, request: ToolRunRequest) -> List[str]:
        extra = request.extra or {}
        question = request.question or ""
        try:
            keywords = self._pattern_probe._pick_keywords(question, extra)  # type: ignore[attr-defined]
        except Exception:
            keywords = []
        if keywords:
            return keywords
        return self._extract_tokens(question)

    async def _rank_paths(self, request: ToolRunRequest, paths: List[Dict[str, Any]], beam_size: int) -> List[Dict[str, Any]]:
        if not paths:
            return []
        limited = sorted(paths, key=lambda item: item.get("score", 0.0), reverse=True)[: max(beam_size * 2, beam_size)]
        if not self.llm_connector:
            return limited[:beam_size]

        try:
            llm_scores = await self._llm_rerank(request, limited)
        except Exception:
            return limited[:beam_size]

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
                "content": (
                    "You evaluate beam search candidates on a knowledge graph. "
                    "Return JSON array [{\"path_id\": \"...\", \"score\": 0-1}] preferring informative paths."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = await call_llm_async(
            self.llm_connector,
            messages,
            temperature=self.temperature,
        )
        data = json.loads(response)
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

    def _normalize_paths(self, payload: Any, *, fallback_prefix: str) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            path_id = str(item.get("path_id") or f"{fallback_prefix}-{idx}")
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
                    "summary": content[:200],
                }
            )
        return normalized

    def _paths_to_evidences(self, paths: Iterable[Dict[str, Any]], source: str) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for rank, path in enumerate(paths):
            nodes = path.get("nodes") or []
            content = path.get("summary") or self._path_summary(path)
            evidences.append(
                EvidenceChunk(
                    chunk_id=f"beam-path-{path['path_id']}",
                    source=source,
                    content=content,
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

    def _extract_tokens(self, question: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+", question or "")
        return self._deduplicate(token for token in tokens if len(token) >= 3)
