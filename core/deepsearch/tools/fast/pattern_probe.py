"""Rule-based tool that scans chunks using deterministic keyword heuristics."""
import re
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.graph_adapter.base import GraphDeepSearchAdapter
from core.graph_adapter.concurrency import adapter_locked

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema


class PatternProbeTool(GraphTool):
    """Fast, LLM-free probe that mimics grep-like semantics on graph chunks."""

    descriptor = ToolDescriptor(
        name="graph.pattern_scan",
        channel="graph",
        description="Deterministic keyword probe on chunks (grep-style fast scan).",
        speed="fast",
        cost="low",
        strategy_tags=("rule_based", "hipporag", "chunk_triple"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.pattern_scan",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "candidate_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pre-selected keywords injected by planner or reasoning loop.",
                }
            }
        ),
        example_args={
            "question": "OpenAI founders",
            "plan_step": "plan_01",
            "extra": {"candidate_keywords": ["OpenAI", "founders"]},
        },
    )

    CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]")

    def __init__(
        self,
        *,
        max_terms: int = 4,
        min_latin_length: int = 4,
        min_cjk_length: int = 2,
    ):
        self.max_terms = max_terms
        self.min_latin_length = min_latin_length
        self.min_cjk_length = min_cjk_length

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        keywords = self._pick_keywords(request.question, request.extra)
        if not keywords:
            return ToolResult(summary="Pattern scan skipped: no stable keywords extracted.")

        matches: List[EvidenceChunk] = []
        diagnostics: Dict[str, Any] = {"keywords": keywords}
        async with adapter_locked(adapter):
            for keyword in keywords[: self.max_terms]:
                payload = await adapter.aquery_subgraph(
                    keyword,
                    channel="graph",
                    access_scope=request.access_scope,
                )
                keyword_hits = self._filter_chunks(payload, keyword)
                matches.extend(keyword_hits)

        if not matches:
            return ToolResult(
                summary="Pattern scan completed but no chunks matched the extracted keywords.",
                diagnostics=diagnostics,
            )

        summary_lines = []
        for kw in keywords:
            hit_count = sum(1 for ev in matches if ev.provenance.get("pattern") == kw)
            summary_lines.append(f"- {kw}: {hit_count} hits")
        summary = "Pattern scan succeeded:\n" + "\n".join(summary_lines)
        return ToolResult(summary=summary, evidences=matches, diagnostics=diagnostics)

    @staticmethod
    def _require_adapter(adapter: GraphDeepSearchAdapter | None) -> GraphDeepSearchAdapter:
        if adapter is None:
            raise RuntimeError("PatternProbeTool requires a GraphDeepSearchAdapter instance")
        return adapter

    def _pick_keywords(self, question: str, extra: Dict[str, Any]) -> List[str]:
        candidate_terms = extra.get("candidate_keywords")
        if isinstance(candidate_terms, list) and candidate_terms:
            return self._deduplicate([str(term).strip() for term in candidate_terms if str(term).strip()])

        if self._looks_cjk(question):
            return self._tokenize_cjk(question or "")
        return self._tokenize_latin(question or "")

    @staticmethod
    def _filter_chunks(payload: Dict[str, Any], keyword: str) -> List[EvidenceChunk]:
        chunks_payload = payload.get("chunks", [])
        if not isinstance(chunks_payload, list):
            return []
        matches: List[EvidenceChunk] = []
        keyword_lower = keyword.lower()
        for idx, chunk in enumerate(chunks_payload):
            content = str(chunk.get("content") or "")
            if keyword_lower not in content.lower():
                continue
            chunk_id = PatternProbeTool._extract_chunk_id(chunk, keyword, idx)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source=payload.get("metadata", {}).get("adapter", "hipporag"),
                content=content,
                score=1.0,
                provenance={
                    "pattern": keyword,
                    "metadata": chunk.get("metadata", {}),
                    "raw_chunk": chunk,
                },
            )
            matches.append(evidence)
        return matches

    @staticmethod
    def _extract_chunk_id(chunk: Dict[str, Any], keyword: str, idx: int) -> str:
        metadata = chunk.get("metadata") or {}
        return str(
            chunk.get("chunk_id")
            or chunk.get("id")
            or metadata.get("chunk_id")
            or metadata.get("id")
            or f"pattern-{keyword}-{idx}"
        )

    def _tokenize_latin(self, question: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+", question or "")
        filtered = [tok.lower() for tok in tokens if len(tok) >= self.min_latin_length]
        return self._deduplicate(filtered)

    def _tokenize_cjk(self, question: str) -> List[str]:
        tokens = self._jieba_tokens(question)
        normalized: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token or not self._looks_cjk(token):
                continue
            if len(token) < self.min_cjk_length:
                continue
            normalized.append(token)
        if normalized:
            return self._deduplicate(normalized)
        # Fallback to overlapping bigrams so the probe still has signals.
        return self._deduplicate(self._fallback_cjk_bigrams(question))

    def _jieba_tokens(self, text: str) -> List[str]:
        try:
            import jieba  # type: ignore
        except ImportError:
            return []
        return [seg for seg in jieba.lcut(text or "") if seg]

    @classmethod
    def _looks_cjk(cls, text: Optional[str]) -> bool:
        if not text:
            return False
        return bool(cls.CHINESE_PATTERN.search(text))

    @classmethod
    def _deduplicate(cls, tokens: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for token in tokens:
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _fallback_cjk_bigrams(self, text: str) -> List[str]:
        cleaned = "".join(ch for ch in text if self._looks_cjk(ch))
        grams: List[str] = []
        for idx in range(len(cleaned) - 1):
            gram = cleaned[idx : idx + self.min_cjk_length]
            if len(gram) >= self.min_cjk_length:
                grams.append(gram)
        return grams
