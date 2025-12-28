"""Report composer for DeepSearch.

This reporter converts DeepSearch execution traces into a readable, end-user report by
prompting an LLM with the collected evidence, highlights, and execution metadata.
"""
import re
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.utils.stopwords import get_stopwords

from core.deepsearch.report.consistency_checker import ConsistencyChecker
from core.deepsearch.report.citation_agent import CitationAgent
from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter, render_markdown_from_structured
from core.deepsearch.trace import emit_trace
from core.presentation.evidence import build_deepsearch_evidence
from core.deepsearch.memory import EvidenceBank
from config.core.deepsearch import report_composer_defaults as composer_defaults
from config.core.deepsearch.stopwords import EVIDENCE_RANK_STOPWORDS
from core.deepsearch.utils.file_scope import (
    extract_titles_from_question,
    filter_evidences,
    normalize_filename_token,
    resolve_file_scope,
)

_EN_STOPWORDS = frozenset(word.lower() for word in get_stopwords("en"))


def _rank_evidences_for_question(question: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministically rank evidences for prompt inclusion (match-first, score-second)."""

    if not items:
        return []
    q = str(question or "")
    anchors: list[str] = []
    # Language-agnostic anchor extraction: pull compact unicode word tokens + numbers.
    for token in re.findall(r"\d[\d,\.%]*", q, flags=re.UNICODE):
        token = token.strip()
        if not token:
            continue
        if token not in anchors:
            anchors.append(token)
    token_re = re.compile(
        rf"[^\W_]{{{composer_defaults.DEFAULT_EVIDENCE_RANK_ASCII_ANCHOR_MIN},{composer_defaults.DEFAULT_EVIDENCE_RANK_ASCII_ANCHOR_MAX}}}",
        flags=re.UNICODE,
    )
    for token in token_re.findall(q):
        token = token.strip()
        if not token or token in EVIDENCE_RANK_STOPWORDS:
            continue
        if _EN_STOPWORDS and token.lower() in _EN_STOPWORDS:
            continue
        if token not in anchors:
            anchors.append(token)

    normalized_anchors = [a for a in anchors if a and a not in EVIDENCE_RANK_STOPWORDS][
        : composer_defaults.DEFAULT_EVIDENCE_RANK_MAX_ANCHORS
    ]

    def _score(ev: Dict[str, Any]) -> tuple[int, int, float, int, str]:
        content = str(ev.get("content") or "")
        lowered = content.lower()
        match_count = 0
        for term in normalized_anchors:
            term_lower = term.lower()
            if term_lower and term_lower in lowered:
                match_count += 1
        raw_score = ev.get("score")
        try:
            numeric_score = float(raw_score) if raw_score is not None else 0.0
        except Exception:
            numeric_score = 0.0
        chunk_id = str(ev.get("chunk_id") or "")
        source = str(ev.get("source") or "").strip().lower()
        toolish_id = ":" in chunk_id or any(chunk_id.startswith(prefix) for prefix in composer_defaults.TOOLISH_CHUNK_ID_PREFIXES)
        toolish_source = source in composer_defaults.TOOLISH_SOURCE_NAMES
        primary = 0 if (toolish_id or toolish_source) else 1
        # Prefer matched content, then score, then shorter snippets (for prompt efficiency).
        return (primary, 1 if match_count > 0 else 0, match_count, numeric_score, -len(content), chunk_id)

    return sorted([ev for ev in items if isinstance(ev, dict)], key=_score, reverse=True)


def _extract_question_scope_terms(question: str) -> List[str]:
    """Best-effort extraction of query scope anchors from the question.

    Language-agnostic heuristic: prefer quoted spans and long unicode word runs.
    """

    q = str(question or "").strip()
    if not q:
        return []

    candidates: list[str] = []
    # Quoted spans (covers a few common quotation styles without relying on a specific language).
    for pat in (r"《([^》]{1,120})》", r"\"([^\"]{1,120})\"", r"“([^”]{1,120})”", r"'([^']{1,120})'", r"‘([^’]{1,120})’"):
        for match in re.finditer(pat, q):
            token = (match.group(1) or "").strip()
            if token and token not in candidates:
                candidates.append(token)

    # Long unicode word runs (letters/digits) optionally with simple separators.
    for match in re.finditer(r"[\w\-\(\)\.]{4,60}", q, flags=re.UNICODE):
        token = match.group(0).strip()
        if not token:
            continue
        # Avoid purely numeric strings.
        if not any(ch.isalpha() for ch in token):
            continue
        if token not in candidates:
            candidates.append(token)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = normalize_filename_token(raw)
        if not token:
            continue
        # Drop common instruction tokens that frequently appear in prompts but are not part of user scope.
        low_raw = token.lower()
        if "chunk_id" in low_raw or "chunkid" in low_raw:
            continue
        if "not found in evidence" in low_raw:
            continue
        low = token.lower()
        if low in seen:
            continue
        seen.add(low)
        normalized.append(token)
        if len(normalized) >= 20:
            break

    # Prune overly-generic tokens: if a token is contained in a longer token, keep the longer one.
    lowered = [t.lower() for t in normalized]
    kept: list[str] = []
    for idx, token in enumerate(normalized):
        t_low = lowered[idx]
        if any((other != t_low) and (t_low in other) and (len(other) >= len(t_low) + 2) for other in lowered):
            continue
        kept.append(token)
    return kept[:10]


def _filter_evidences_by_question_scope(question: str, items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    terms = _extract_question_scope_terms(question)
    if len(terms) < 2:
        return items, {"scope_terms_applied": False}

    # Prefer terms that are discriminative across the retrieved evidence pool.
    # This keeps cross-doc questions from "drifting" into unrelated files when generic
    # words (e.g. "payment", "plan") appear everywhere.
    evidence_count = len([ev for ev in items if isinstance(ev, dict)])
    max_common_ratio = 0.65
    rare_ratio_cutoff = 0.45
    max_effective_terms = 6
    term_stats: Dict[str, Dict[str, Any]] = {}
    for term in terms:
        term_lower = term.lower()
        hits = 0
        for ev in items:
            if not isinstance(ev, dict):
                continue
            content = str(ev.get("content") or "").lower()
            prov = ev.get("provenance") if isinstance(ev.get("provenance"), dict) else {}
            meta = prov.get("metadata") if isinstance(prov, dict) and isinstance(prov.get("metadata"), dict) else {}
            filename = None
            if isinstance(meta, dict):
                filename = meta.get("filename") or meta.get("source_file_name") or meta.get("path")
                if not filename:
                    chunk_meta = meta.get("chunk_metadata")
                    if isinstance(chunk_meta, dict):
                        filename = chunk_meta.get("filename") or chunk_meta.get("source_file_name") or chunk_meta.get("path")
            filename_lower = str(filename or "").lower()
            if term_lower and (term_lower in content or term_lower in filename_lower):
                hits += 1
        ratio = (hits / evidence_count) if evidence_count else 0.0
        term_stats[term] = {"hits": hits, "ratio": round(ratio, 3)}

    # Guardrail: if the question includes at least one high-specificity anchor (numbers / roman numerals / parentheses)
    # and we cannot find it in the evidence pool at all, do NOT fall back to unrelated evidences.
    def _looks_like_anchor(term: str) -> bool:
        if re.search(r"\d", term):
            return True
        # Roman numerals as standalone markers (avoid matching common letters inside words).
        if re.search(r"(?:^|[^A-Za-z])[IVX]{1,6}(?:$|[^A-Za-z])", term):
            return True
        if re.search(r"[()（）]", term):
            return True
        return False

    anchor_terms = [t for t in terms if _looks_like_anchor(t)]
    if anchor_terms:
        if all(int(term_stats.get(t, {}).get("hits") or 0) <= 0 for t in anchor_terms):
            return [], {
                "scope_terms_applied": True,
                "hard_filter": True,
                "anchor_miss": True,
                "anchor_terms": anchor_terms,
                "scope_terms": terms,
                "term_stats": term_stats,
            }

    dropped_missing = [t for t in terms if int(term_stats.get(t, {}).get("hits") or 0) <= 0]
    dropped_common = [t for t in terms if t not in dropped_missing and float(term_stats.get(t, {}).get("ratio") or 0.0) > max_common_ratio]
    candidates = [t for t in terms if t not in dropped_missing and t not in dropped_common]
    if len(candidates) < 2:
        return items, {
            "scope_terms_applied": False,
            "scope_terms": terms,
            "term_stats": term_stats,
            "dropped_missing": dropped_missing,
            "dropped_common": dropped_common,
            "fallback": True,
        }

    # Take the rarest terms first (ratio asc, then prefer longer tokens).
    ranked = sorted(
        candidates,
        key=lambda t: (float(term_stats.get(t, {}).get("ratio") or 1.0), -len(t)),
    )
    rare = [t for t in ranked if float(term_stats.get(t, {}).get("ratio") or 1.0) <= rare_ratio_cutoff]
    if len(rare) >= 2:
        discriminative_terms = rare[:max_effective_terms]
    else:
        discriminative_terms = ranked[: max(2, min(max_effective_terms, len(ranked)))]
    if len(discriminative_terms) < 2:
        return items, {
            "scope_terms_applied": False,
            "scope_terms": terms,
            "term_stats": term_stats,
            "dropped_missing": dropped_missing,
            "dropped_common": dropped_common,
            "fallback": True,
        }

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for ev in items:
        if not isinstance(ev, dict):
            continue
        content = str(ev.get("content") or "").lower()
        prov = ev.get("provenance") if isinstance(ev.get("provenance"), dict) else {}
        meta = prov.get("metadata") if isinstance(prov, dict) and isinstance(prov.get("metadata"), dict) else {}
        filename = None
        if isinstance(meta, dict):
            filename = meta.get("filename") or meta.get("source_file_name") or meta.get("path")
            if not filename:
                chunk_meta = meta.get("chunk_metadata")
                if isinstance(chunk_meta, dict):
                    filename = chunk_meta.get("filename") or chunk_meta.get("source_file_name") or chunk_meta.get("path")
        filename_lower = str(filename or "").lower()
        in_scope = any(term.lower() in content or term.lower() in filename_lower for term in discriminative_terms)
        if in_scope:
            kept.append(ev)
        else:
            dropped += 1

    if kept:
        return kept, {
            "scope_terms_applied": True,
            "scope_terms": terms,
            "scope_terms_effective": discriminative_terms,
            "term_stats": term_stats,
            "dropped_missing": dropped_missing,
            "dropped_common": dropped_common,
            "kept": len(kept),
            "dropped": dropped,
        }
    # Avoid false negatives: if heuristic drops everything, keep original set.
    return items, {
        "scope_terms_applied": False,
        "scope_terms": terms,
        "scope_terms_effective": discriminative_terms,
        "term_stats": term_stats,
        "dropped_missing": dropped_missing,
        "dropped_common": dropped_common,
        "kept": len(items),
        "dropped": 0,
        "fallback": True,
    }


def _is_tool_generated_evidence(evidence: Dict[str, Any]) -> bool:
    chunk_id = str(evidence.get("chunk_id") or evidence.get("evidence_id") or "").strip()
    if not chunk_id:
        return False
    lowered = chunk_id.lower()
    if ":" in chunk_id or any(lowered.startswith(prefix) for prefix in composer_defaults.TOOLISH_CHUNK_ID_PREFIXES):
        return True
    source = str(evidence.get("source") or "").strip().lower()
    return source in composer_defaults.TOOLISH_SOURCE_NAMES


def _split_authoritative_evidences(evidences: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split evidences into authoritative corpus chunks vs tool-generated artifacts.

    Tool-generated artifacts (e.g. context rollups) are helpful for debugging and guidance, but should
    never be treated as source-of-truth citations in the final report.
    """

    authoritative: List[Dict[str, Any]] = []
    generated: List[Dict[str, Any]] = []
    for ev in evidences or []:
        if not isinstance(ev, dict):
            continue
        if _is_tool_generated_evidence(ev):
            generated.append(ev)
            continue
        # Non-tool evidences (regardless of chunk_id format) are treated as authoritative.
        authoritative.append(ev)
    return authoritative, generated


def _trim_text(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    limit = max(0, int(max_chars))
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _slim_diagnostics(
    payload: Any,
    *,
    max_keys: int = composer_defaults.DEFAULT_DIAGNOSTICS_MAX_KEYS,
    max_value_chars: int = composer_defaults.DEFAULT_DIAGNOSTICS_MAX_VALUE_CHARS,
) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    trimmed: Dict[str, Any] = {}
    for key in sorted(payload.keys(), key=lambda k: str(k))[: max(0, int(max_keys))]:
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float, bool)):
            trimmed[str(key)] = raw
        elif isinstance(raw, str):
            trimmed[str(key)] = _trim_text(raw, max_chars=max_value_chars)
        elif isinstance(raw, (list, tuple)) and len(raw) <= composer_defaults.DEFAULT_DIAGNOSTICS_MAX_SMALL_LIST_ITEMS:
            item_limit = composer_defaults.DEFAULT_DIAGNOSTICS_LIST_ITEM_PREVIEW_CHARS
            trimmed[str(key)] = [(_trim_text(item, max_chars=item_limit) if isinstance(item, str) else item) for item in raw]
        else:
            continue
    return trimmed or None


class DeepSearchReporter:
    """Generate a structured report from DeepSearch traces."""

    STRUCTURED_REPORT_VERSION = "2.0"

    def __init__(
        self,
        template_store,
        config,
        *,
        llm_connector: Any | None = None,
        graph_store: Any | None = None,
    ):
        self.template_store = template_store
        if not isinstance(config, dict) or not config:
            raise ValueError("DeepSearchReporter requires a non-empty config dict")
        self.config = dict(config)
        self.llm_connector = llm_connector
        self.graph_store = graph_store

        self.max_highlights = int(self.config["max_highlights"])
        self.max_evidence_items = int(self.config["max_evidence_items"])
        self.report_temperature = float(self.config["report_temperature"])
        self.report_max_evidence_chars = int(self.config["report_max_evidence_chars"])
        self.report_max_graph_chain_items = int(self.config["report_max_graph_chain_items"])
        self.report_max_seed_entities = int(self.config["report_max_seed_entities"])
        self.enable_llm_report = bool(self.config["enable_llm_report"])
        self.enable_consistency_check = bool(self.config["enable_consistency_check"])
        self.consistency_temperature = float(self.config["consistency_temperature"])
        self.enable_citation_agent = bool(self.config["enable_citation_agent"])
        self.parallel_sections = bool(self.config["parallel_sections"])
        self.max_parallel_sections = int(self.config["max_parallel_sections"])
        self.sectionwise_writer = bool(self.config["sectionwise_writer"])
        self.sectionwise_retain_k = int(self.config["sectionwise_retain_k"])
        self.citation_aliases = bool(self.config["citation_aliases"])
        self.include_appendices_in_answer = bool(self.config.get("include_appendices_in_answer", False))

    async def compose(
        self,
        reasoning_trace: Dict[str, Any],
        external_evidence: Optional[Iterable[Dict[str, Any] | EvidenceChunk]] = None,
    ) -> Dict[str, Any]:
        """Compose a report payload for downstream clients (API/CLI/MCP)."""

        if not self.enable_llm_report:
            raise RuntimeError("Deterministic reports are disabled; enable LLM report generation instead.")
        if self.enable_llm_report and self.llm_connector is None:
            raise RuntimeError("LLM connector is required for LLM report generation")

        trace = reasoning_trace or {}
        question = (trace.get("question") or "").strip()
        if not question:
            raise ValueError("DeepSearch reasoning trace is missing 'question'")

        evidences = self._merge_evidences(trace.get("evidences"), external_evidence, question=question)
        authoritative_evidences, tool_evidences = _split_authoritative_evidences(evidences)
        graph_context = trace.get("graph_context") if isinstance(trace.get("graph_context"), dict) else {}
        ctx_meta = graph_context.get("metadata") if isinstance(graph_context, dict) else {}
        file_scope = resolve_file_scope(graph_context_metadata=(ctx_meta if isinstance(ctx_meta, dict) else {}), question=question)
        file_scope_diag: Dict[str, Any] = {}
        if file_scope.enabled:
            authoritative_evidences, file_scope_diag = filter_evidences(authoritative_evidences, scope=file_scope)
        question_scope_diag: Dict[str, Any] = {}
        if authoritative_evidences:
            authoritative_evidences, question_scope_diag = _filter_evidences_by_question_scope(question, authoritative_evidences)
        if not authoritative_evidences:
            msg = "No reliable evidence was retrieved for this question."
            if file_scope.enabled:
                tokens = ", ".join(file_scope.filename_contains) if file_scope.filename_contains else ""
                if tokens:
                    msg = f"No reliable evidence was found within the specified file scope ({tokens})."
                else:
                    msg = "No reliable evidence was found within the specified file scope."

            request_context = self._extract_request_context(trace)
            metadata: Dict[str, Any] = {
                "question": question,
                "adapter_metadata": trace.get("adapter_metadata") or {},
                "graph_summary": self._graph_summary(trace.get("graph_traversals") or []),
                "plan": {
                    "steps": trace.get("plan_steps") or [],
                    "completed": sum(1 for step in ((trace.get("reasoning_steps") or []) or []) if isinstance(step, dict) and step.get("status") == "done"),
                },
                "coverage_metrics": trace.get("coverage_metrics") or {},
                "pending_external": trace.get("pending_external") or [],
                "parallel_thinking_runs": int(self.config["parallel_thinking_runs"]),
                "report_profile": {
                    "include_graph_viz": bool(self.config["include_graph_viz"]),
                    "enable_custom_summary": bool(self.config["enable_custom_summary"]),
                },
                "evidence_profile": {
                    "total": len(evidences),
                    "authoritative": 0,
                    "tool_generated": len(tool_evidences),
                },
            }
            if file_scope_diag:
                metadata["evidence_profile"]["file_scope"] = file_scope_diag
            if question_scope_diag:
                metadata["evidence_profile"]["question_scope"] = question_scope_diag
            if request_context:
                metadata["request_context"] = request_context
            if metadata["report_profile"]["include_graph_viz"]:
                metadata["graph_visualization"] = trace.get("graph_traversals") or []

                structured_report = {
                    "format_version": self.STRUCTURED_REPORT_VERSION,
                    "title": question,
                    "short_answer": msg,
                    "summary": msg,
                    "sections": [],
                    "limitations": [
                        "Insufficient evidence is available to produce verifiable, citation-backed conclusions; to avoid mis-citations, report generation was stopped.",
                    ],
                    "next_steps": [
                        "Confirm the target file(s) are indexed successfully and visible under the current user/tenant scope.",
                        "Rewrite the query with more specific clause/page/table keywords or provide more context hints.",
                    ],
                    "citations": [],
                    "evidence_index": [],
                    "graph_evidence": {},
                    "text": msg,
                "context": request_context or {},
                "generation": {"mode": "deterministic_no_evidence"},
            }
            return {
                "question": question,
                "answer": msg,
                "highlights": [],
                "evidences": [],
                "structured_report": structured_report,
                "metadata": metadata,
            }

        requested_titles = extract_titles_from_question(question)
        if len(requested_titles) >= 2:
            covered: set[str] = set()
            normalized = [normalize_filename_token(t) for t in requested_titles]
            normalized_map = {t: token for t, token in zip(requested_titles, normalized) if token}

            def _extract_filename(ev: Dict[str, Any]) -> str | None:
                prov = ev.get("provenance") or {}
                if not isinstance(prov, dict):
                    return None

                meta = prov.get("metadata") or {}
                if isinstance(meta, dict):
                    for key in ("filename", "source_file_name", "source_file", "file_path", "path"):
                        value = meta.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()

                    chunk_meta = meta.get("chunk_metadata")
                    if isinstance(chunk_meta, dict):
                        for key in ("filename", "source_file_name", "source_file", "file_path", "path"):
                            value = chunk_meta.get(key)
                            if isinstance(value, str) and value.strip():
                                return value.strip()

                raw_chunk = prov.get("raw_chunk")
                if isinstance(raw_chunk, dict):
                    raw_meta = raw_chunk.get("metadata")
                    if isinstance(raw_meta, dict):
                        for key in ("filename", "source_file_name", "source_file", "file_path", "path"):
                            value = raw_meta.get(key)
                            if isinstance(value, str) and value.strip():
                                return value.strip()

                return None

            for ev in authoritative_evidences:
                if not isinstance(ev, dict):
                    continue
                filename = _extract_filename(ev)
                if not filename:
                    continue
                lowered = filename.lower()
                for title, token in normalized_map.items():
                    if token.lower() in lowered:
                        covered.add(title)

            missing = [t for t in requested_titles if t not in covered]
            if missing:
                missing_text = ", ".join(missing)
                msg = (
                    "Unable to complete the comparison: no reliable evidence was retrieved for the following named file(s): "
                    f"{missing_text}."
                )
                if covered:
                    msg += f" (Only retrieved evidence from: {', '.join(sorted(covered))})"

                request_context = self._extract_request_context(trace)
                metadata: Dict[str, Any] = {
                    "question": question,
                    "adapter_metadata": trace.get("adapter_metadata") or {},
                    "graph_summary": self._graph_summary(trace.get("graph_traversals") or []),
                    "plan": {
                        "steps": trace.get("plan_steps") or [],
                        "completed": sum(1 for step in (trace.get("reasoning_steps") or []) if isinstance(step, dict) and step.get("status") == "done"),
                    },
                    "coverage_metrics": trace.get("coverage_metrics") or {},
                    "pending_external": trace.get("pending_external") or [],
                    "parallel_thinking_runs": int(self.config["parallel_thinking_runs"]),
                    "report_profile": {
                        "include_graph_viz": bool(self.config["include_graph_viz"]),
                        "enable_custom_summary": bool(self.config["enable_custom_summary"]),
                    },
                    "evidence_profile": {
                        "total": len(evidences),
                        "authoritative": len(authoritative_evidences),
                        "tool_generated": len(tool_evidences),
                    },
                }
                if file_scope_diag:
                    metadata["evidence_profile"]["file_scope"] = file_scope_diag
                if question_scope_diag:
                    metadata["evidence_profile"]["question_scope"] = question_scope_diag
                if request_context:
                    metadata["request_context"] = request_context
                if metadata["report_profile"]["include_graph_viz"]:
                    metadata["graph_visualization"] = trace.get("graph_traversals") or []

                structured_report = {
                    "format_version": self.STRUCTURED_REPORT_VERSION,
                    "title": question,
                    "short_answer": msg,
                    "summary": msg,
                    "sections": [],
                    "limitations": [
                        "A comparison requires verifiable evidence for each named file; to avoid cross-file mis-citations, the comparison was not generated.",
                    ],
                    "next_steps": [
                        "Confirm the missing file(s) are indexed successfully and visible under the current user/tenant scope.",
                        "Split the comparison into two single-file questions and merge the results after manual verification.",
                    ],
                    "citations": [],
                    "evidence_index": [],
                    "graph_evidence": {},
                    "text": msg,
                    "context": request_context or {},
                    "generation": {"mode": "deterministic_missing_named_files"},
                }
                return {
                    "question": question,
                    "answer": msg,
                    "highlights": [],
                    "evidences": authoritative_evidences,
                    "structured_report": structured_report,
                    "metadata": metadata,
                }
        llm_base_evidences = authoritative_evidences
        llm_evidences, alias_bundle = (
            self._alias_evidences_for_llm(llm_base_evidences) if self.citation_aliases else (llm_base_evidences, None)
        )
        reasoning_steps = trace.get("reasoning_steps") or []
        highlights = self._build_highlights(reasoning_steps)
        coverage_metrics = trace.get("coverage_metrics") or {}
        gap_result = trace.get("gap_result") or {}
        request_context = self._extract_request_context(trace)

        metadata: Dict[str, Any] = {
            "question": question,
            "adapter_metadata": trace.get("adapter_metadata") or {},
            "graph_summary": self._graph_summary(trace.get("graph_traversals") or []),
            "plan": {
                "steps": trace.get("plan_steps") or [],
                "completed": sum(1 for step in (reasoning_steps or []) if step.get("status") == "done"),
            },
            "reasoning_steps": reasoning_steps,
            "think_notes": trace.get("think_notes") or [],
            "tool_results": trace.get("tool_results") or [],
            "coverage_metrics": coverage_metrics,
            "gap_result": gap_result,
            "pending_external": trace.get("pending_external") or [],
            "parallel_thinking_runs": int(self.config["parallel_thinking_runs"]),
            "report_profile": {
                "include_graph_viz": bool(self.config["include_graph_viz"]),
                "enable_custom_summary": bool(self.config["enable_custom_summary"]),
            },
            "evidence_profile": {
                "total": len(evidences),
                "authoritative": len(authoritative_evidences),
                "tool_generated": len(tool_evidences),
            },
        }
        if file_scope_diag:
            metadata["evidence_profile"]["file_scope"] = file_scope_diag
        if question_scope_diag:
            metadata["evidence_profile"]["question_scope"] = question_scope_diag
        if alias_bundle:
            metadata["citation_aliases"] = {
                "enabled": True,
                "count": alias_bundle.get("count"),
                "sample": alias_bundle.get("sample"),
            }
        if request_context:
            metadata["request_context"] = request_context
        if metadata["report_profile"]["include_graph_viz"]:
            metadata["graph_visualization"] = trace.get("graph_traversals") or []

        context = self._build_llm_context(
            trace=trace,
            highlights=highlights,
            evidences=llm_evidences,
            coverage=coverage_metrics,
            gap_result=gap_result,
            request_context=request_context,
        )
        writer = DeepSearchLLMReportWriter(
            self.llm_connector,
            temperature=self.report_temperature,
            max_evidence_items=self.max_evidence_items,
            max_evidence_chars=self.report_max_evidence_chars,
            max_graph_chain_items=self.report_max_graph_chain_items,
            parallel_sections=self.parallel_sections,
            max_parallel_sections=self.max_parallel_sections,
            synthesis_section_max_chars=int(self.config["synthesis_section_max_chars"]),
        )
        outline = await writer.build_outline(question=question, context=context)
        if self.sectionwise_writer:
            structured_llm = await writer.write_report_sectionwise(
                question=question,
                outline=outline,
                context=context,
                retain_k=max(0, int(self.sectionwise_retain_k)),
            )
        elif self.parallel_sections:
            structured_llm = await writer.write_report_parallel(question=question, outline=outline, context=context)
        else:
            structured_llm = await writer.write_report(question=question, outline=outline, context=context)
        if alias_bundle:
            structured_llm, alias_diag = self._rewrite_citation_aliases(structured_llm, alias_bundle)
            if alias_diag:
                metadata["citation_alias_rewrite"] = alias_diag
        summary_text = structured_llm.get("short_answer") or structured_llm.get("summary")
        if isinstance(summary_text, str) and summary_text.strip() and authoritative_evidences:
            if not re.search(r"(?:\[[^\[\]]+\]|【[^【】]+】)", summary_text):
                raise ValueError("short_answer is missing inline citations (cite-first hard gate).")
        if self.enable_citation_agent:
            structured_llm, audit = CitationAgent().process(
                structured_report=structured_llm,
                evidences=authoritative_evidences,
                graph_evidence=context.get("graph_evidence") or {},
                max_chunk_index_items=self.max_evidence_items,
                citation_aliases=(alias_bundle.get("alias_to_original") if alias_bundle else None),
            )
            metadata["citation_audit"] = audit
        markdown_text = render_markdown_from_structured(structured_llm)
        if self.include_appendices_in_answer:
            markdown_text = _append_chunk_evidence(
                markdown_text,
                evidences=authoritative_evidences,
                max_items=self.max_evidence_items,
            )
            markdown_text = _append_tool_evidence(
                markdown_text,
                evidences=tool_evidences,
                max_items=self.max_evidence_items,
            )
            markdown_text = _append_graph_appendix(
                markdown_text,
                graph_evidence=context.get("graph_evidence") or {},
                max_seed_entities=self.report_max_seed_entities,
                max_chain_items=self.report_max_graph_chain_items,
            )

        structured_report = {
            "format_version": self.STRUCTURED_REPORT_VERSION,
            "title": structured_llm.get("title") or question,
            "short_answer": structured_llm.get("short_answer") or structured_llm.get("summary") or "",
            "summary": structured_llm.get("short_answer") or structured_llm.get("summary") or "",
            "sections": structured_llm.get("sections") or [],
            "limitations": structured_llm.get("limitations") or [],
            "next_steps": structured_llm.get("next_steps") or [],
            "citations": structured_llm.get("citations") or [],
            "evidence_index": structured_llm.get("evidence_index") or [],
            "graph_evidence": context.get("graph_evidence") or {},
            "text": markdown_text,
            "context": request_context or {},
            "generation": {"mode": "llm"},
        }

        if self.enable_consistency_check and authoritative_evidences:
            checker = ConsistencyChecker(
                self.llm_connector,
                temperature=self.consistency_temperature,
                max_retries=int(self.config["consistency_max_retries"]),
            )
            result = await checker.check(
                question=question,
                report_markdown=markdown_text,
                evidences=authoritative_evidences,
                structured_report=structured_report,
                max_evidence_chars=self.report_max_evidence_chars,
                max_claims=int(self.config["consistency_max_claims"]),
            )
            structured_report["consistency_check"] = result.model_dump()
            if not result.is_consistent:
                metadata["quality_warnings"] = [issue.model_dump() for issue in result.issues]

        metadata["structured_report"] = {key: structured_report[key] for key in structured_report if key != "text"}

        return {
            "question": question,
            "answer": markdown_text,
            "evidences": evidences,
            "highlights": highlights,
            "structured_report": structured_report,
            "metadata": metadata,
        }

    @staticmethod
    def _alias_evidences_for_llm(evidences: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
        """Replace long chunk IDs with stable short aliases for LLM prompting.

        The alias is treated as the chunk_id inside prompts, then rewritten back to the original
        chunk_id before running CitationAgent / ConsistencyChecker.
        """

        alias_to_original: Dict[str, str] = {}
        original_to_alias: Dict[str, str] = {}
        aliased: List[Dict[str, Any]] = []
        for idx, evidence in enumerate(evidences, start=1):
            if not isinstance(evidence, dict):
                continue
            original_id = str(evidence.get("chunk_id") or "").strip()
            if not original_id:
                continue
            alias = f"chunk_{idx:03d}"
            alias_to_original[alias] = original_id
            original_to_alias[original_id] = alias
            copied = dict(evidence)
            copied["chunk_id"] = alias
            provenance = copied.get("provenance") or {}
            if isinstance(provenance, dict):
                meta = provenance.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                meta.setdefault("original_chunk_id", original_id)
                provenance["metadata"] = meta
                copied["provenance"] = provenance
            aliased.append(copied)
        if not aliased:
            return evidences, None
        sample = [
            {"alias": a, "chunk_id": b}
            for a, b in list(alias_to_original.items())[: composer_defaults.DEFAULT_ALIAS_SAMPLE_SIZE]
        ]
        return aliased, {
            "count": len(alias_to_original),
            "alias_to_original": alias_to_original,
            "original_to_alias": original_to_alias,
            "sample": sample,
        }

    _CITATION_RE = re.compile(
        rf"(?:\[(?P<bracket>[^\[\]]{{1,{composer_defaults.DEFAULT_CITATION_TOKEN_MAX_CHARS}}})\]|【(?P<cjk>[^【】]{{1,{composer_defaults.DEFAULT_CITATION_TOKEN_MAX_CHARS}}})】)"
    )

    def _rewrite_citation_aliases(
        self,
        structured: Dict[str, Any],
        alias_bundle: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        alias_to_original = alias_bundle.get("alias_to_original") or {}
        if not isinstance(alias_to_original, dict) or not alias_to_original:
            return structured, {}
        normalized_map = {str(k).strip().lower(): str(v) for k, v in alias_to_original.items()}

        replaced = 0
        unknown: List[str] = []

        def _normalize_token(token: str) -> str | None:
            raw = (token or "").strip()
            if not raw:
                return None
            low = raw.lower()

            def _lookup(candidate: str) -> str | None:
                key = (candidate or "").strip().lower()
                if not key:
                    return None
                if key in normalized_map:
                    return normalized_map[key]
                return None

            direct = _lookup(low)
            if direct:
                return direct

            cleaned = low.strip("`'\"(){}<>.,;: ")
            cleaned_lookup = _lookup(cleaned)
            if cleaned_lookup:
                return cleaned_lookup

            compact = re.sub(r"\s+", "", cleaned)
            compact_lookup = _lookup(compact)
            if compact_lookup:
                return compact_lookup

            if compact.isdigit():
                idx = int(compact)
                mapped = _lookup(f"chunk_{idx:03d}")
                if mapped:
                    return mapped

            if "chunk" not in compact:
                return None

            token_digits = re.sub(r"[^0-9]", "", compact)
            if token_digits.isdigit():
                idx = int(token_digits)
                mapped = _lookup(f"chunk_{idx:03d}")
                if mapped:
                    return mapped
            return None

        def _rewrite_text(text: str) -> str:
            nonlocal replaced

            def _sub(match: re.Match[str]) -> str:
                nonlocal replaced
                token = match.group("bracket") or match.group("cjk") or ""
                mapped = _normalize_token(token)
                if mapped:
                    replaced += 1
                    return f"[{mapped}]"

                low = token.strip().lower()
                if "chunk" in low and any(sep in low for sep in (",", ";", " ")):
                    parts = re.findall(r"chunk\s*[_-]?\s*\d{1,4}", token, flags=re.IGNORECASE)
                    mapped_parts: List[str] = []
                    unmapped_parts: List[str] = []
                    for part in parts:
                        mapped_part = _normalize_token(part)
                        if mapped_part:
                            mapped_parts.append(mapped_part)
                        else:
                            unmapped_parts.append(part)
                    if mapped_parts and not unmapped_parts:
                        replaced += len(mapped_parts)
                        return " ".join(f"[{item}]" for item in mapped_parts)
                    if unmapped_parts:
                        unknown.extend([part.strip() for part in unmapped_parts if str(part).strip()])

                if "chunk" in low:
                    unknown.append(token.strip())
                return match.group(0)

            return self._CITATION_RE.sub(_sub, text)

        rewritten = dict(structured)
        for key in ("short_answer", "summary", "title"):
            if isinstance(rewritten.get(key), str):
                rewritten[key] = _rewrite_text(rewritten[key])
        sections = rewritten.get("sections")
        if isinstance(sections, list):
            new_sections: List[Dict[str, Any]] = []
            for section in sections:
                if not isinstance(section, dict):
                    continue
                updated = dict(section)
                body = updated.get("body_markdown")
                if isinstance(body, str):
                    updated["body_markdown"] = _rewrite_text(body)
                new_sections.append(updated)
            rewritten["sections"] = new_sections
        for list_key in ("limitations", "next_steps"):
            items = rewritten.get(list_key)
            if isinstance(items, list):
                rewritten[list_key] = [_rewrite_text(str(item)) if isinstance(item, str) else item for item in items]
        citations = rewritten.get("citations")
        if isinstance(citations, list):
            new_citations: List[Dict[str, Any]] = []
            for entry in citations:
                if not isinstance(entry, dict):
                    continue
                updated = dict(entry)
                ev_id = updated.get("evidence_id")
                if isinstance(ev_id, str):
                    mapped = _normalize_token(ev_id)
                    if mapped:
                        updated["evidence_id"] = mapped
                        replaced += 1
                new_citations.append(updated)
            rewritten["citations"] = new_citations

        diag = {
            "replaced": replaced,
            "unknown_aliases": sorted({token for token in unknown})[: composer_defaults.DEFAULT_UNKNOWN_ALIAS_SAMPLE_SIZE],
        }
        return rewritten, diag

    def _build_llm_context(
        self,
        *,
        trace: Dict[str, Any],
        highlights: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        gap_result: Dict[str, Any],
        request_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan_steps = trace.get("plan_steps") or []
        reasoning_steps = trace.get("reasoning_steps") or []
        tool_results = list(trace.get("tool_results") or [])
        pending_external = trace.get("pending_external") or []

        methodology_summary_chars = int(self.config["methodology_summary_chars"])
        keep_tool_results = int(self.config["keep_tool_results"])
        if keep_tool_results == 0:
            tool_results = []
        elif keep_tool_results > 0:
            tool_results = tool_results[-keep_tool_results:]
        methodology = {
            "plan_steps": plan_steps,
            "reasoning_steps": [
                {
                    "step_id": step.get("step_id"),
                    "description": step.get("description"),
                    "channel": step.get("channel"),
                    "status": step.get("status"),
                    "output_summary": _trim_text(step.get("output_summary"), max_chars=methodology_summary_chars),
                    "tool": step.get("tool")
                    or (step.get("metadata") or {}).get("tool")
                    or (step.get("diagnostics") or {}).get("tool"),
                }
                for step in reasoning_steps
                if isinstance(step, dict)
            ],
            "tool_results": [
                {
                    "plan_step_id": entry.get("plan_step_id"),
                    "tool_name": entry.get("tool_name"),
                    "channel": entry.get("channel"),
                    "summary": (entry.get("result") or {}).get("summary") if isinstance(entry.get("result"), dict) else None,
                    "diagnostics": _slim_diagnostics(
                        (entry.get("result") or {}).get("diagnostics") if isinstance(entry.get("result"), dict) else None
                    ),
                }
                for entry in tool_results
                if isinstance(entry, dict)
            ],
        }

        coverage_bundle = {
            "coverage_metrics": coverage,
            "gap_result": gap_result,
            "pending_external": pending_external,
        }

        graph_evidence_full = self._build_graph_evidence(trace, evidences)
        graph_evidence_llm = self._slim_graph_evidence_for_llm(graph_evidence_full)
        bank = EvidenceBank()
        bank.add_many(evidences)
        outline_index_limit = self.max_evidence_items if self.max_evidence_items is not None else None
        evidence_index = bank.index_for_prompt(
            max_items=outline_index_limit,
            max_summary_chars=int(self.config["outline_evidence_summary_chars"]),
        )

        return {
            "question": trace.get("question") or "",
            "final_answer": trace.get("final_answer") or "",
            "highlights": highlights,
            "evidences": evidences,
            "evidence_index": evidence_index,
            "graph_chain": (trace.get("graph_chain") or [])[: self.report_max_graph_chain_items],
            "graph_evidence": graph_evidence_llm,
            "methodology": methodology,
            "coverage": coverage_bundle,
            "request_context": request_context,
        }

    def _merge_evidences(
        self,
        internal: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
        external: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
        *,
        question: str,
    ) -> List[Dict[str, Any]]:
        internal_items = [item for item in (self._normalize_evidence(payload) for payload in (internal or [])) if item]
        external_items = [item for item in (self._normalize_evidence(payload) for payload in (external or [])) if item]

        max_items = self.max_evidence_items if self.max_evidence_items is not None else None
        if max_items is not None and max_items <= 0:
            return []

        # Rank before applying the final max_items budget so prompts focus on high-signal chunks.
        internal_items = _rank_evidences_for_question(question, internal_items)
        external_items = _rank_evidences_for_question(question, external_items)

        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _add(items: List[Dict[str, Any]], limit: int | None) -> None:
            nonlocal merged
            for chunk in items:
                source = str(chunk.get("source") or "").strip()
                chunk_id = chunk.get("chunk_id") or self._hash_content(chunk)
                key = f"{source}::{chunk_id}"
                if key in seen:
                    continue
                seen.add(key)
                chunk.setdefault("chunk_id", chunk_id)
                merged.append(chunk)
                if limit is not None and len(merged) >= limit:
                    return

        if external_items and max_items is not None:
            external_budget = min(
                len(external_items),
                max(
                    composer_defaults.DEFAULT_EXTERNAL_EVIDENCE_MIN_ITEMS,
                    max_items // composer_defaults.DEFAULT_EXTERNAL_EVIDENCE_FRACTION_DIVISOR,
                ),
            )
            internal_budget = max(0, max_items - external_budget)
            _add(internal_items, internal_budget)
            _add(external_items, max_items)
            if len(merged) < max_items:
                _add(internal_items, max_items)
                _add(external_items, max_items)
            return merged[:max_items]

        _add(internal_items, max_items)
        if max_items is not None and len(merged) >= max_items:
            return merged[:max_items]
        _add(external_items, max_items)
        return merged[:max_items] if max_items is not None else merged

    def _iter_evidences(
        self,
        internal: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
        external: Optional[Iterable[Dict[str, Any] | EvidenceChunk]],
    ) -> Iterable[Dict[str, Any]]:
        for payload in (internal or []):
            normalized = self._normalize_evidence(payload)
            if normalized:
                yield normalized
        for payload in (external or []):
            normalized = self._normalize_evidence(payload)
            if normalized:
                yield normalized

    @staticmethod
    def _normalize_evidence(payload: Dict[str, Any] | EvidenceChunk | None) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if isinstance(payload, EvidenceChunk):
            return payload.model_dump()
        if isinstance(payload, dict) and payload.get("content"):
            return dict(payload)
        return None

    @staticmethod
    def _hash_content(chunk: Dict[str, Any]) -> str:
        return hashed_chunk_id(source=str(chunk.get("source") or ""), content=str(chunk.get("content") or ""))

    def _extract_request_context(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        graph_context = trace.get("graph_context") or {}
        metadata = graph_context.get("metadata") or {}
        request_metadata = metadata.get("request_metadata")
        context: Dict[str, Any] = {}
        if isinstance(request_metadata, dict):
            for key, value in request_metadata.items():
                if value is None:
                    continue
                context[str(key)] = str(value)
        access_scope = graph_context.get("access_scope") or {}
        scope_id = access_scope.get("scope_id")
        if scope_id and "scope_id" not in context:
            context["scope_id"] = str(scope_id)
        return context

    def _build_highlights(self, reasoning_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        highlights: List[Dict[str, Any]] = []
        for step in reasoning_steps:
            if step.get("status") != "done":
                continue
            summary = (step.get("output_summary") or "").strip()
            if not summary:
                continue
            highlights.append(
                {
                    "step_id": step.get("step_id"),
                    "description": step.get("description"),
                    "summary": summary,
                    "evidence_ids": step.get("produced_evidence_ids") or [],
                }
            )
            if self.max_highlights > 0 and len(highlights) >= self.max_highlights:
                break
        return highlights

    @staticmethod
    def _graph_summary(traversals: List[Dict[str, Any]]) -> Dict[str, Any]:
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        for record in traversals:
            for node in record.get("visited_nodes", []) or []:
                if node:
                    node_ids.add(str(node))
            for edge in record.get("visited_edges", []) or []:
                if edge:
                    edge_ids.add(str(edge))
        return {
            "traversal_count": len(traversals),
            "unique_nodes": len(node_ids),
            "unique_edges": len(edge_ids),
        }

    def _build_graph_evidence(self, trace: Dict[str, Any], evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a graph evidence bundle (seeds/chain) when possible."""

        payload = {
            "reasoning": {"evidences": evidences},
            "report": {"evidences": evidences},
            "graph_chain": trace.get("graph_chain") or [],
        }
        return build_deepsearch_evidence(payload, chunk_limit=self.max_evidence_items, graph_store=self.graph_store)

    @staticmethod
    def _slim_graph_evidence_for_llm(graph_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce graph evidence payload for LLM prompts to avoid context blow-ups.

        The full public payload can still include detailed chunk/graph snapshots, but the report-writing
        prompt should only receive compact graph signals (seeds + chain + stats) because the authoritative
        evidence snippets are provided separately.
        """

        if not isinstance(graph_evidence, dict):
            return {}
        seed_entities = graph_evidence.get("seed_entities") if isinstance(graph_evidence.get("seed_entities"), list) else []
        graph_chain = graph_evidence.get("graph_chain") if isinstance(graph_evidence.get("graph_chain"), list) else []
        graph_stats = graph_evidence.get("graph_stats") if isinstance(graph_evidence.get("graph_stats"), dict) else {}
        return {
            "seed_entities": seed_entities[: composer_defaults.DEFAULT_GRAPH_EVIDENCE_SEED_ENTITIES_MAX],
            "graph_chain": graph_chain[: composer_defaults.DEFAULT_GRAPH_CHAIN_PREVIEW_MAX],
            "graph_stats": graph_stats,
        }


def _append_chunk_evidence(
    markdown_text: str,
    *,
    evidences: List[Dict[str, Any]],
    max_items: int | None = None,
    preview_length: int = composer_defaults.DEFAULT_EVIDENCE_PREVIEW_CHARS,
) -> str:
    """Append a compact section listing chunk evidence with content preview."""

    if not evidences:
        return markdown_text

    capped = evidences[:max_items] if max_items is not None and max_items > 0 else evidences
    if not capped:
        return markdown_text

    blocks: List[str] = ["", "## Appendix: Chunk Evidence"]
    for entry in capped:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        source = str(entry.get("source") or "").strip()
        content = str(entry.get("content") or "").strip()
        if not chunk_id:
            continue
        preview = content[:preview_length].replace("\n", " ").strip()
        if len(content) > preview_length:
            preview += "..."
        source_tag = f" ({source})" if source else ""
        blocks.append(f"- [{chunk_id}]{source_tag}: {preview}")

    return (markdown_text or "").rstrip() + "\n" + "\n".join(blocks).rstrip() + "\n"


def _append_tool_evidence(
    markdown_text: str,
    *,
    evidences: List[Dict[str, Any]],
    max_items: int | None = None,
    preview_length: int = composer_defaults.DEFAULT_EVIDENCE_PREVIEW_CHARS,
) -> str:
    """Append non-authoritative tool-generated artifacts for debugging."""

    if not evidences:
        return markdown_text

    capped = evidences[:max_items] if max_items is not None and max_items > 0 else evidences
    if not capped:
        return markdown_text

    blocks: List[str] = ["", "## Appendix: Tool Artifacts (Non-authoritative)"]
    blocks.append(
        "These entries are generated by tools/LLMs (e.g. rollups). They can help debugging, but should NOT be treated as source-of-truth citations."
    )
    for entry in capped:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        source = str(entry.get("source") or "").strip()
        content = str(entry.get("content") or "").strip()
        if not chunk_id:
            continue
        preview = content[:preview_length].replace("\n", " ").strip()
        if len(content) > preview_length:
            preview += "..."
        source_tag = f" ({source})" if source else ""
        blocks.append(f"- [{chunk_id}]{source_tag}: {preview}")

    return (markdown_text or "").rstrip() + "\n" + "\n".join(blocks).rstrip() + "\n"


def _append_graph_appendix(
    markdown_text: str,
    *,
    graph_evidence: Dict[str, Any],
    max_seed_entities: int,
    max_chain_items: int,
    max_items: int = composer_defaults.DEFAULT_EVIDENCE_PREVIEW_MAX_ITEMS,
) -> str:
    """Append a compact appendix listing graph evidence artifacts."""

    if not isinstance(graph_evidence, dict):
        return markdown_text

    seeds = list(graph_evidence.get("seed_entities") or [])
    chain = list(graph_evidence.get("graph_chain") or [])
    stats = graph_evidence.get("graph_stats") or {}

    if not any((seeds, chain, stats)):
        return markdown_text

    blocks: List[str] = ["", "## Appendix: Graph Evidence"]
    if stats:
        node_count = stats.get("nodes")
        edge_count = stats.get("edges")
        category_count = len(stats.get("categories") or []) if isinstance(stats.get("categories"), list) else None
        blocks.append(
            f"- Graph stats: nodes={node_count}, edges={edge_count}, categories={category_count}"
        )
    if seeds:
        blocks.append("")
        blocks.append("### Seed Entities")
        seeds = seeds[: max(int(max_seed_entities), 0)]
        blocks.extend(f"- {item}" for item in seeds)
    if chain:
        blocks.append("")
        blocks.append("### Graph Chain")
        chain = chain[: max(int(max_chain_items), 0)]
        blocks.extend(f"{idx}. {edge}" for idx, edge in enumerate(chain, start=1))

    return (markdown_text or "").rstrip() + "\n" + "\n".join(blocks).rstrip() + "\n"
