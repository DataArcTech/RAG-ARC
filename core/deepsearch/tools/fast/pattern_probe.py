"""Rule-based tool that scans chunks using deterministic keyword heuristics."""
import re
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.graph_adapter.base import GraphDeepSearchAdapter
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.query_clean import clean_query

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

        diagnostics: Dict[str, Any] = {"keywords": keywords}
        max_terms = int(self.max_terms) if self.max_terms is not None else 0
        if max_terms < 0:
            max_terms = 0
        invoked_keywords: List[str] = []
        skipped_keywords: List[str] = []
        diagnostics["invoked_keywords"] = invoked_keywords
        diagnostics["skipped_keywords"] = skipped_keywords
        diagnostics["per_keyword"] = []

        evidence_by_id: Dict[str, EvidenceChunk] = {}
        async with adapter_locked(adapter):
            executed = 0
            for keyword in keywords:
                if max_terms and executed >= max_terms:
                    skipped_keywords.append(keyword)
                    continue
                executed += 1
                invoked_keywords.append(keyword)
                payload = await adapter.aquery_subgraph(
                    keyword,
                    channel="graph",
                    access_scope=request.access_scope,
                )
                keyword_hits = self._filter_chunks(payload, keyword)
                diagnostics["per_keyword"].append({"keyword": keyword, "chunk_count": len(keyword_hits)})
                for ev in keyword_hits:
                    existing = evidence_by_id.get(ev.chunk_id)
                    if existing is None:
                        patterns = ev.provenance.get("patterns") if isinstance(ev.provenance, dict) else None
                        if not isinstance(patterns, list):
                            ev.provenance["patterns"] = [keyword]
                        evidence_by_id[ev.chunk_id] = ev
                        continue
                    provenance = existing.provenance if isinstance(existing.provenance, dict) else {}
                    patterns = provenance.get("patterns")
                    if not isinstance(patterns, list):
                        patterns = []
                    if keyword not in patterns:
                        patterns.append(keyword)
                    provenance["patterns"] = patterns
                    existing.provenance = provenance
                    if ev.score is not None and (existing.score is None or float(ev.score) > float(existing.score)):
                        existing.score = ev.score

        matches = list(evidence_by_id.values())
        diagnostics["unique_match_count"] = len(matches)
        diagnostics["match_chunk_ids"] = [ev.chunk_id for ev in matches]

        if not matches:
            return ToolResult(
                summary="Pattern scan completed but no chunks matched the extracted keywords.",
                diagnostics=diagnostics,
            )

        summary_lines = []
        for kw in keywords:
            hit_count = sum(1 for ev in matches if kw in (ev.provenance.get("patterns") or []))
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

        compact = clean_query(question or "", max_chars=480)
        if self._looks_cjk(compact):
            return self._tokenize_cjk(compact)
        return self._tokenize_latin(compact)

    @staticmethod
    def _filter_chunks(payload: Dict[str, Any], keyword: str) -> List[EvidenceChunk]:
        chunks_payload = payload.get("chunks", [])
        if not isinstance(chunks_payload, list):
            return []
        matches: List[EvidenceChunk] = []
        for idx, chunk in enumerate(chunks_payload):
            content = str(chunk.get("content") or "")
            chunk_id = PatternProbeTool._extract_chunk_id(chunk, keyword, idx)
            score = chunk.get("score")
            if score is None and isinstance(chunk.get("metadata"), dict):
                score = chunk["metadata"].get("score")
            metadata = payload.get("metadata")
            source = "hipporag"
            if isinstance(metadata, dict):
                source = metadata.get("adapter") or metadata.get("adapter_name") or source
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source=str(source),
                content=content,
                score=score,
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
            return self._rank_cjk_terms(normalized)

        extracted = self._regex_cjk_terms(question)
        if extracted:
            return self._rank_cjk_terms(extracted)

        # Final fallback to overlapping bigrams so the probe still has signals.
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

    @staticmethod
    def _regex_cjk_terms(text: str) -> List[str]:
        raw = str(text or "")
        raw = raw.replace("/", " ").replace("／", " ").replace("、", " ").replace("，", " ").replace(",", " ")
        raw = raw.replace("（", " ").replace("）", " ").replace("(", " ").replace(")", " ")
        candidates = re.findall(r"[\u3400-\u9fff]{2,10}", raw)
        return [c.strip() for c in candidates if c.strip()]

    def _rank_cjk_terms(self, tokens: List[str]) -> List[str]:
        """Rank terms so we don't accidentally pick meaningless leading bigrams."""

        stop = {
            "请",
            "给出",
            "一份",
            "结构化",
            "对比",
            "报告",
            "并",
            "每个",
            "关键",
            "事实",
            "标注",
            "引用",
            "基于",
            "已上传",
            "涉及文件",
            "这些",
            "相关",
            "主要",
            "提示",
        }
        seen: set[str] = set()
        scored: List[tuple[int, str]] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token in stop:
                continue
            if token in seen:
                continue
            seen.add(token)
            score = len(token)
            if any(marker in token for marker in ("险", "保", "回报", "退保", "供款", "年金", "提取", "身故", "保障", "风险")):
                score += 6
            if any(marker in token for marker in ("规则", "共同点", "差异", "保证", "非保证")):
                score += 3
            scored.append((score, token))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [token for _, token in scored]
        return ranked[: max(1, self.max_terms)]
