"""Rule-based tool that scans chunks using deterministic keyword heuristics."""
import re
from typing import Any, Dict, Iterable, List, Optional

from config.core.deepsearch.tool_defaults import (
    PATTERN_PROBE_DEFAULT_MAX_TERMS,
    PATTERN_PROBE_DEFAULT_MIN_CJK_LENGTH,
    PATTERN_PROBE_DEFAULT_MIN_LATIN_LENGTH,
)
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from config.core.deepsearch.stopwords import PATTERN_PROBE_CJK_STOPWORDS, PATTERN_PROBE_DEFAULT_STOPWORDS
from core.graph_adapter.base import GraphDeepSearchAdapter
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.query_clean import clean_query
from core.deepsearch.utils.file_scope import chunk_in_scope, resolve_file_scope
from core.utils.stopwords import get_stopwords
from core.utils.text_regex import CJK_DETECT_RE, NONWORD_SPACES_RE

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER

_EN_STOPWORDS = frozenset(word.lower() for word in get_stopwords("en"))
_ZH_STOPWORDS = get_stopwords("zh")


class PatternProbeTool(GraphTool):
    """Fast, LLM-free probe that mimics grep-like semantics on graph chunks."""

    descriptor = ToolDescriptor(
        name="graph.pattern_scan",
        channel="graph",
        description=(
            "Deterministic keyword probe over chunk corpus (grep-style). "
            "Evidence: primary chunks (citeable); respects file_scope when enabled."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("rule_based", "hipporag", "chunk_triple", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.pattern_scan",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "query": {
                    "type": "string",
                    "description": "Optional comma/whitespace-separated keyword hints (used when candidate_keywords is absent).",
                },
                "focus_query": {
                    "type": "string",
                    "description": "Optional focus query text that can be heuristically split into keywords (used when candidate_keywords is absent).",
                },
                "candidate_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pre-selected keywords injected by planner or reasoning loop.",
                },
                "max_terms": {
                    "type": "integer",
                    "description": "Optional override for how many keywords to probe.",
                    "minimum": 0,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Optional override for how many candidate chunks to fetch per keyword (adapter-dependent).",
                    "minimum": 0,
                },
                "match_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to enforce match validation against: content, filename, metadata.",
                },
                "match_case_sensitive": {
                    "type": "boolean",
                    "description": "Whether keyword matching is case-sensitive for Latin strings (defaults to false).",
                },
                "match_strip_whitespace": {
                    "type": "boolean",
                    "description": "Whether to strip whitespace before matching (defaults to true for CJK-heavy keywords).",
                },
            }
        ),
        example_args={
            "question": "OpenAI founders",
            "plan_step": "plan_01",
            "extra": {"candidate_keywords": ["OpenAI", "founders"]},
        },
    )

    CHINESE_PATTERN = CJK_DETECT_RE
    _NONWORD_SPACES = NONWORD_SPACES_RE

    def __init__(
        self,
        *,
        max_terms: int = PATTERN_PROBE_DEFAULT_MAX_TERMS,
        min_latin_length: int = PATTERN_PROBE_DEFAULT_MIN_LATIN_LENGTH,
        min_cjk_length: int = PATTERN_PROBE_DEFAULT_MIN_CJK_LENGTH,
    ):
        self.max_terms = max_terms
        self.min_latin_length = min_latin_length
        self.min_cjk_length = min_cjk_length

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        keywords = self._pick_keywords(request.question, request.extra, request.context_evidences)
        if not keywords:
            return ToolResult(summary="Pattern scan skipped: no stable keywords extracted.")

        diagnostics: Dict[str, Any] = {"keywords": keywords}
        file_scope = resolve_file_scope(
            extra=request.extra,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
            question=request.question,
        )
        if file_scope.enabled:
            diagnostics["file_scope"] = file_scope.as_dict()
        match_fields = self._resolve_match_fields(request.extra)
        diagnostics["match_fields"] = match_fields
        top_k = self._resolve_top_k_override(request.extra)
        if top_k is not None:
            diagnostics["top_k"] = top_k
        case_sensitive = self._resolve_bool(request.extra.get("match_case_sensitive"), default=False)
        diagnostics["match_case_sensitive"] = case_sensitive
        override = request.extra.get("max_terms", None)
        try:
            effective_max = int(override) if override is not None else int(self.max_terms)
        except Exception:
            effective_max = int(self.max_terms) if self.max_terms is not None else 0
        max_terms = effective_max if effective_max is not None else 0
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
                    query_options=(
                        {
                            "top_k": top_k,
                            "file_scope": file_scope.as_dict(),
                        }
                        if top_k is not None and file_scope.enabled
                        else {"top_k": top_k}
                        if top_k is not None
                        else {"file_scope": file_scope.as_dict()}
                        if file_scope.enabled
                        else None
                    ),
                )
                if file_scope.enabled:
                    chunks_payload = payload.get("chunks")
                    if isinstance(chunks_payload, list):
                        filtered_chunks = [
                            chunk
                            for chunk in chunks_payload
                            if isinstance(chunk, dict)
                            and chunk_in_scope(
                                chunk_metadata=(chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}),
                                scope=file_scope,
                            )
                        ]
                        payload = dict(payload)
                        payload["chunks"] = filtered_chunks
                keyword_hits, filter_diag = self._filter_chunks(
                    payload,
                    keyword,
                    match_fields=match_fields,
                    case_sensitive=case_sensitive,
                    strip_whitespace=self._resolve_strip_whitespace(keyword, request.extra),
                )
                diagnostics["per_keyword"].append(
                    {
                        "keyword": keyword,
                        "candidate_chunk_count": filter_diag.get("candidate_chunk_count"),
                        "matched_chunk_count": filter_diag.get("matched_chunk_count"),
                        "filtered_chunk_count": filter_diag.get("filtered_chunk_count"),
                        "filter_reasons": filter_diag.get("filter_reasons"),
                        "file_cluster_counts": filter_diag.get("file_cluster_counts"),
                    }
                )
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
                    matched_fields = provenance.get("matched_fields")
                    if not isinstance(matched_fields, list):
                        matched_fields = []
                    incoming_fields = ev.provenance.get("matched_fields") if isinstance(ev.provenance, dict) else None
                    if isinstance(incoming_fields, list):
                        for field in incoming_fields:
                            token = str(field or "").strip()
                            if token and token not in matched_fields:
                                matched_fields.append(token)
                    if matched_fields:
                        provenance["matched_fields"] = matched_fields
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

    def _pick_keywords(
        self,
        question: str,
        extra: Dict[str, Any],
        context_evidences: Optional[List[EvidenceChunk]] = None,
    ) -> List[str]:
        candidate_terms = extra.get("candidate_keywords")
        if isinstance(candidate_terms, list) and candidate_terms:
            expanded: List[str] = []
            for term in candidate_terms:
                expanded.extend(self._split_keyword_hints(str(term)))
            return self._deduplicate(self._filter_stopwords([token for token in expanded if token]))

        context_terms: List[str] = []
        if context_evidences:
            context_terms = self._context_terms(context_evidences)

        for key in ("query", "focus_query"):
            raw = extra.get(key)
            if isinstance(raw, str) and raw.strip():
                hints = self._split_keyword_hints(raw)
                if hints:
                    merged = hints + context_terms
                    return self._deduplicate(self._filter_stopwords(merged))

        compact = clean_query(question or "", max_chars=480)
        candidates: List[str] = []
        if context_terms:
            candidates.extend(context_terms)
        if compact:
            if self._looks_cjk(compact):
                candidates.extend(self._tokenize_cjk(compact))
            candidates.extend(self._tokenize_latin(compact))

        filtered = self._deduplicate(self._filter_stopwords(candidates))
        return filtered[: max(0, int(self.max_terms))]

    def _context_terms(self, evidences: List[EvidenceChunk]) -> List[str]:
        terms: List[str] = []
        seen: set[str] = set()

        def _push(value: Any) -> None:
            token = str(value or "").strip()
            if not token or token in seen:
                return
            if not self._looks_cjk(token):
                return
            if len(token) < self.min_cjk_length:
                return
            seen.add(token)
            terms.append(token)

        for ev in evidences:
            provenance = getattr(ev, "provenance", None) or {}
            if not isinstance(provenance, dict):
                continue
            triples = provenance.get("triples")
            if not isinstance(triples, list):
                continue
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                _push(triple.get("head"))
                _push(triple.get("tail"))
                _push(triple.get("relation"))
                if len(terms) >= max(8, int(self.max_terms or 0)):
                    return terms[: max(8, int(self.max_terms or 0))]
        return terms

    def _split_keyword_hints(self, raw: str) -> List[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        parts = re.split(r"[，,;；\\n\\t\\s\\|｜/、]+", text)
        cleaned: List[str] = []
        for part in parts:
            token = part.strip().strip("\"'`()[]{}<>")
            if not token:
                continue
            if self._looks_cjk(token):
                if len(token) < self.min_cjk_length:
                    continue
            else:
                if len(token) < self.min_latin_length:
                    continue
            cleaned.append(token)
        return cleaned

    @classmethod
    def _filter_stopwords(cls, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        filtered: List[str] = []
        for token in tokens:
            value = str(token).strip()
            if not value:
                continue
            if value in PATTERN_PROBE_DEFAULT_STOPWORDS:
                continue
            if value in _ZH_STOPWORDS:
                continue
            lowered = value.lower()
            if any(ch.isalpha() for ch in value) and lowered in _EN_STOPWORDS:
                continue
            filtered.append(value)
        return filtered

    @staticmethod
    def _resolve_bool(value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        raw = str(value).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _resolve_top_k_override(extra: Dict[str, Any]) -> int | None:
        raw = extra.get("top_k")
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _resolve_match_fields(extra: Dict[str, Any]) -> List[str]:
        raw = extra.get("match_fields")
        fields: List[str] = []
        if isinstance(raw, list):
            for item in raw:
                name = str(item or "").strip().lower()
                if not name:
                    continue
                fields.append(name)
        if not fields:
            # Strict by default: only accept chunks that contain the keyword in content.
            fields = ["content"]
        seen: set[str] = set()
        out: List[str] = []
        for field in fields:
            if field in seen:
                continue
            seen.add(field)
            out.append(field)
        return out

    def _resolve_strip_whitespace(self, keyword: str, extra: Dict[str, Any]) -> bool:
        override = extra.get("match_strip_whitespace")
        if override is not None:
            return self._resolve_bool(override, default=self._looks_cjk(keyword))
        # For CJK-heavy keywords we default to stripping whitespace as PDF extraction can insert it.
        return self._looks_cjk(keyword)

    def _normalize_match_text(self, text: str, *, case_sensitive: bool, strip_whitespace: bool) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        if strip_whitespace:
            value = self._NONWORD_SPACES.sub("", value)
            value = re.sub(r"[()（）\\[\\]{}<>《》「」『』【】]", "", value)
        value = value.replace("／", "/").replace("，", ",").replace("（", "(").replace("）", ")")
        if not case_sensitive:
            value = value.lower()
        return value

    def _matched_fields(
        self,
        chunk: Dict[str, Any],
        *,
        normalized_keyword: str,
        match_fields: List[str],
        case_sensitive: bool,
        strip_whitespace: bool,
    ) -> List[str]:
        if not normalized_keyword:
            return []
        matched: List[str] = []

        if "content" in match_fields:
            content = self._normalize_match_text(
                str(chunk.get("content") or ""),
                case_sensitive=case_sensitive,
                strip_whitespace=strip_whitespace,
            )
            if content and normalized_keyword in content:
                matched.append("content")

        metadata = chunk.get("metadata")
        meta_dict = metadata if isinstance(metadata, dict) else {}

        if "filename" in match_fields:
            fname = self._normalize_match_text(
                str(
                    meta_dict.get("source_file_name")
                    or meta_dict.get("filename")
                    or meta_dict.get("source_file")
                    or meta_dict.get("file_path")
                    or meta_dict.get("path")
                    or ""
                ),
                case_sensitive=case_sensitive,
                strip_whitespace=strip_whitespace,
            )
            if fname and normalized_keyword in fname:
                matched.append("filename")

        if "metadata" in match_fields and meta_dict:
            for key in ("title", "heading", "section", "doc_title", "document_title"):
                value = meta_dict.get(key)
                if not isinstance(value, str) or not value.strip():
                    continue
                compact = self._normalize_match_text(
                    value,
                    case_sensitive=case_sensitive,
                    strip_whitespace=strip_whitespace,
                )
                if compact and normalized_keyword in compact:
                    matched.append("metadata")
                    break

        return matched

    @staticmethod
    def _source_file_cluster(chunk: Dict[str, Any]) -> str | None:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            return None
        for key in ("source_file_id", "file_id", "document_id", "doc_id"):
            value = metadata.get(key)
            if value is None:
                continue
            token = str(value).strip()
            if token:
                return f"id:{token}"
        for key in ("source_file_name", "filename", "source_file", "file_path", "path"):
            value = metadata.get(key)
            if not isinstance(value, str):
                continue
            token = value.strip()
            if token:
                return f"name:{token}"
        return None

    def _filter_chunks(
        self,
        payload: Dict[str, Any],
        keyword: str,
        *,
        match_fields: List[str],
        case_sensitive: bool,
        strip_whitespace: bool,
    ) -> tuple[List[EvidenceChunk], Dict[str, Any]]:
        chunks_payload = payload.get("chunks", [])
        if not isinstance(chunks_payload, list):
            return [], {"candidate_chunk_count": 0, "matched_chunk_count": 0, "filtered_chunk_count": 0}

        normalized_keyword = self._normalize_match_text(
            keyword,
            case_sensitive=case_sensitive,
            strip_whitespace=strip_whitespace,
        )
        if not normalized_keyword:
            return [], {
                "candidate_chunk_count": len(chunks_payload),
                "matched_chunk_count": 0,
                "filtered_chunk_count": len(chunks_payload),
                "filter_reasons": {"empty_keyword": len(chunks_payload)},
                "file_cluster_counts": {},
            }

        matches: List[EvidenceChunk] = []
        filtered = 0
        filter_reasons: Dict[str, int] = {}
        file_cluster_counts: Dict[str, int] = {}

        metadata = payload.get("metadata")
        source = "hipporag"
        if isinstance(metadata, dict):
            source = metadata.get("adapter") or metadata.get("adapter_name") or source

        for idx, chunk in enumerate(chunks_payload):
            content = str(chunk.get("content") or "")
            chunk_id = PatternProbeTool._extract_chunk_id(chunk, keyword, idx)
            score = chunk.get("score")
            if score is None and isinstance(chunk.get("metadata"), dict):
                score = chunk["metadata"].get("score")

            matched_fields = self._matched_fields(
                chunk,
                normalized_keyword=normalized_keyword,
                match_fields=match_fields,
                case_sensitive=case_sensitive,
                strip_whitespace=strip_whitespace,
            )
            if not matched_fields:
                filtered += 1
                filter_reasons["no_match"] = filter_reasons.get("no_match", 0) + 1
                continue

            cluster = self._source_file_cluster(chunk)
            if cluster:
                file_cluster_counts[cluster] = file_cluster_counts.get(cluster, 0) + 1

            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source=str(source),
                content=content,
                kind=EVIDENCE_KIND_PRIMARY,
                score=score,
                provenance={
                    "pattern": keyword,
                    "matched_fields": matched_fields,
                    "metadata": chunk.get("metadata", {}),
                    "raw_chunk": chunk,
                },
            )
            matches.append(evidence)

        return matches, {
            "candidate_chunk_count": len(chunks_payload),
            "matched_chunk_count": len(matches),
            "filtered_chunk_count": filtered,
            "filter_reasons": filter_reasons,
            "file_cluster_counts": dict(sorted(file_cluster_counts.items(), key=lambda kv: (-kv[1], kv[0])))
            if file_cluster_counts
            else {},
        }

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

        return []

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

    @staticmethod
    def _regex_cjk_terms(text: str) -> List[str]:
        raw = str(text or "")
        raw = raw.replace("/", " ").replace("／", " ").replace("、", " ").replace("，", " ").replace(",", " ")
        raw = raw.replace("（", " ").replace("）", " ").replace("(", " ").replace(")", " ")
        candidates = re.findall(r"[\u3400-\u9fff]{2,10}", raw)
        return [c.strip() for c in candidates if c.strip()]

    def _rank_cjk_terms(self, tokens: List[str]) -> List[str]:
        """Rank terms so we don't accidentally pick meaningless leading bigrams."""

        stop = PATTERN_PROBE_CJK_STOPWORDS
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
            scored.append((score, token))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [token for _, token in scored]
        return ranked[: max(1, self.max_terms)]
