"""LLM-driven report generation for DeepSearch."""
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from config.core.deepsearch import report_writer_defaults as report_defaults
from core.deepsearch.memory import EvidenceBank
from core.utils.json_extract import extract_json_from_text as _extract_json_from_text
from core.utils.citations import extract_sup_keys
from core.deepsearch.utils.language_policy import infer_user_language
from config.output_limits import DEEPSEARCH_MAX_IMAGE_INPUTS
from core.utils.multimodal_images import collect_image_paths_from_deepsearch_evidences
from core.utils.multimodal_llm import call_llm_with_optional_images_async
from core.utils.llm_json import call_llm_json_with_retry, repair_json_from_raw_with_retry
from core.prompts.deepsearch.report import (
    REPORT_OUTLINE_SYSTEM_PROMPT_EN,
    REPORT_OUTLINE_USER_PROMPT_EN,
    REPORT_WRITE_SYSTEM_PROMPT_EN,
    REPORT_WRITE_USER_PROMPT_EN,
    DEEPSEARCH_WRITE_USER_PROMPT_EN,
    REPORT_SOURCE_SELECT_SYSTEM_PROMPT_EN,
    REPORT_SOURCE_SELECT_USER_PROMPT_EN,
    REPORT_STYLE_DEEPSEARCH_HINT_EN,
    REPORT_STYLE_RESEARCH_HINT_EN,
    SECTION_WRITE_SYSTEM_PROMPT_EN,
    SECTION_WRITE_USER_PROMPT_EN,
    JSON_REPAIR_USER_PROMPT_EN,
)
from core.deepsearch.report.llm_writer_budget import (
    dump_json as _dump_json,
    limit_evidences as _limit_evidences,
    limit_highlights as _limit_highlights,
    slim_coverage as _slim_coverage,
    slim_graph_evidence as _slim_graph_evidence,
    slim_methodology as _slim_methodology,
)


def _preview_head_mid_tail(text: str, *, max_chars: int) -> str:
    """Compact preview for navigation-only indexing (head/middle/tail).

    Used for source-selection prompts so short-context models still see a bit of the
    middle (often where table rows / numeric values live).
    """

    raw = str(text or "").strip().replace("\n", " ")
    limit = max(0, int(max_chars))
    if limit <= 0:
        return ""
    if len(raw) <= limit:
        return raw
    if limit <= 12:
        return raw[:limit]

    head = max(4, int(limit * 0.4))
    tail = max(4, int(limit * 0.3))
    mid = max(0, limit - head - tail - 1)  # 1 char for ellipsis
    if mid < 4:
        head = max(4, limit - tail - 1)
        mid = 0

    start = raw[:head].rstrip()
    end = raw[-tail:].lstrip()
    if mid <= 0:
        return f"{start}…{end}".strip()

    center_start = max(0, (len(raw) // 2) - (mid // 2))
    center = raw[center_start : center_start + mid].strip()
    return f"{start}…{center}…{end}".strip()


def _query_terms(question: str, *, max_terms: int = 8) -> List[str]:
    """Extract lightweight query terms for navigation-only indexing.

    Avoid domain-specific rules. We keep:
    - alnum tokens (len>=2)
    - CJK tokens (len>=2)
    """

    q = str(question or "").strip()
    if not q:
        return []
    terms: List[str] = []
    seen: set[str] = set()

    # Alnum words
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,}", q):
        t = tok.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(t)
        if len(terms) >= max_terms:
            return terms

    # CJK runs
    for tok in re.findall(r"[\u4e00-\u9fff]{2,}", q):
        t = tok.strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        terms.append(t)
        if len(terms) >= max_terms:
            break
    return terms[:max_terms]


def _term_hit_preview(text: str, *, terms: Sequence[str], max_terms: int = 4) -> str:
    """Return a compact 'hits=' preview string for the selection index."""

    if not terms:
        return ""
    raw = str(text or "")
    lower = raw.lower()
    hits: List[tuple[str, int]] = []
    for t in terms:
        if not t:
            continue
        key = t.lower()
        count = lower.count(key) if key else 0
        if count > 0:
            hits.append((t, int(count)))
    if not hits:
        return ""
    hits.sort(key=lambda x: x[1], reverse=True)
    hits = hits[: max(1, int(max_terms))]
    return "hits=" + ",".join(f"{t}:{c}" for t, c in hits)


def _page_signal_stats(text: str) -> dict[str, int]:
    """Lightweight signals to help the model choose the right pages.

    This is navigation-only and does NOT change the underlying evidence content.
    """

    raw = str(text or "")
    lower = raw.lower()
    irr = lower.count("irr")
    pct = raw.count("%")
    # Keep this intentionally simple; we only want a rough "table density" hint.
    nums = len(re.findall(r"[0-9]+[.][0-9]+", raw))
    return {"irr": int(irr), "pct": int(pct), "nums": int(nums)}


def _expand_contiguous_pages(
    *,
    bank: EvidenceBank,
    candidate_ids: Sequence[str],
    selected_ids: Sequence[str],
    max_sources: int,
) -> List[str]:
    """Expand selected read.pages evidences to prefer contiguous pages within the same file.

    We keep the model's choice as the anchor, then (when possible) add immediate neighbors
    (p-1/p+1) from the same file until reaching max_sources.
    """

    limit = max(1, int(max_sources))
    ordered: List[str] = []
    seen: set[str] = set()
    for eid in selected_ids or []:
        token = str(eid or "").strip()
        if not token or token in seen:
            continue
        if token in candidate_ids:
            ordered.append(token)
            seen.add(token)
    if len(ordered) >= limit:
        return ordered[:limit]

    # Build (file_id, page)->evidence_id index for read.pages only.
    page_index: dict[tuple[str, int], str] = {}
    for eid in candidate_ids or []:
        rec = bank.get(str(eid or "").strip())
        if rec is None:
            continue
        if (rec.source or "").strip() != "read.pages":
            continue
        prov = rec.provenance or {}
        file_id = str(prov.get("source_file_id") or "").strip()
        if not file_id:
            meta = prov.get("metadata") if isinstance(prov.get("metadata"), dict) else {}
            file_id = str((meta or {}).get("source_file_id") or "").strip()
        if not file_id:
            continue
        page_start = prov.get("page_start")
        page_end = prov.get("page_end")
        try:
            ps = int(page_start) if page_start is not None else None
        except Exception:
            ps = None
        try:
            pe = int(page_end) if page_end is not None else None
        except Exception:
            pe = None
        if ps is None:
            continue
        if pe is not None and pe != ps:
            # `read.pages` currently emits per-page evidences; ranges can be supported later.
            continue
        page_index[(file_id, ps)] = str(rec.evidence_id)

    # Expand around each selected read.pages page, preferring tighter neighborhoods first.
    # We do a small BFS on page distance: 1,2,3,... until we fill the budget or no new pages exist.
    anchors: List[tuple[str, int]] = []
    for eid in ordered:
        rec = bank.get(eid)
        if rec is None or (rec.source or "").strip() != "read.pages":
            continue
        prov = rec.provenance or {}
        file_id = str(prov.get("source_file_id") or "").strip()
        if not file_id:
            meta = prov.get("metadata") if isinstance(prov.get("metadata"), dict) else {}
            file_id = str((meta or {}).get("source_file_id") or "").strip()
        if not file_id:
            continue
        page_start = prov.get("page_start")
        try:
            ps = int(page_start) if page_start is not None else None
        except Exception:
            ps = None
        if ps is None:
            continue
        anchors.append((file_id, ps))

    for dist in range(1, 6):
        if len(ordered) >= limit:
            break
        for file_id, page in anchors:
            if len(ordered) >= limit:
                break
            for candidate_page in (page - dist, page + dist):
                if len(ordered) >= limit:
                    break
                if candidate_page <= 0:
                    continue
                neighbor_id = page_index.get((file_id, int(candidate_page)))
                if not neighbor_id or neighbor_id in seen:
                    continue
                ordered.append(neighbor_id)
                seen.add(neighbor_id)

    return ordered[:limit]

def _allowed_key_ints(source_key_map: Dict[str, str]) -> List[int]:
    keys: List[int] = []
    for raw in (source_key_map or {}).keys():
        try:
            keys.append(int(str(raw).strip()))
        except Exception:  # noqa: BLE001
            continue
    return sorted(set(keys))


def _has_any_allowed_citation(text: str, *, allowed_keys: Sequence[int]) -> bool:
    allowed: set[int] = set()
    for k in allowed_keys or ():
        try:
            allowed.add(int(k))
        except Exception:  # noqa: BLE001
            continue
    return _has_supported_inline_citation(text, allowed_keys=allowed)

def _has_supported_inline_citation(text: str, *, allowed_keys: set[int]) -> bool:
    if not text or not allowed_keys:
        return False
    max_key = max(allowed_keys) if allowed_keys else None
    for key in extract_sup_keys(text, max_key=max_key):
        if key in allowed_keys:
            return True
    return False


def _has_triple_chain_lines(text: str) -> bool:
    """Detect triple-chain lines in Markdown blockquotes (e.g. '> A->rel->B')."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(">"):
            continue
        if "->" in stripped:
            return True
    return False


def _coerce_source_key_list(value: Any) -> List[int]:
    if isinstance(value, (list, tuple, set)):
        out: List[int] = []
        for item in value:
            try:
                num = int(str(item).strip())
            except Exception:  # noqa: BLE001
                continue
            if num <= 0:
                continue
            if num not in out:
                out.append(num)
        return out
    return []


def _all_sup_keys_allowed(text: str, *, allowed_keys: Sequence[int]) -> bool:
    allowed: set[int] = set()
    for k in allowed_keys or ():
        try:
            allowed.add(int(k))
        except Exception:  # noqa: BLE001
            continue
    if not allowed:
        return True
    max_key = max(allowed) if allowed else None
    for key in extract_sup_keys(text or "", max_key=max_key):
        if key not in allowed:
            return False
    return True


def _extract_evidence_ids_from_index(evidence_index: Any) -> List[str]:
    """Extract citable ids (chunk_id) from the evidence index payload."""
    if not isinstance(evidence_index, list):
        return []
    ordered: List[str] = []
    seen: set[str] = set()
    for entry in evidence_index:
        if not isinstance(entry, dict):
            continue
        token = str(entry.get("chunk_id") or entry.get("evidence_id") or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _allowed_keys_for_evidence_ids(
    evidence_ids: Sequence[str],
    *,
    source_key_map: Dict[str, str],
) -> set[int]:
    reverse: Dict[str, int] = {}
    for key, ev_id in source_key_map.items():
        try:
            key_num = int(str(key).strip())
        except Exception:  # noqa: BLE001
            continue
        token = str(ev_id or "").strip()
        if token:
            reverse[token] = key_num
    allowed: set[int] = set()
    for ev_id in evidence_ids:
        token = str(ev_id or "").strip()
        key_num = reverse.get(token)
        if key_num is not None:
            allowed.add(int(key_num))
    return allowed


def _fallback_outline(question: str, *, evidence_index: Any) -> List[Dict[str, Any]]:
    """Deterministic fallback outline when the LLM outline step fails.

    This exists to keep DeepSearch service available even when the outline LLM returns
    invalid schema. It is only used when we have at least one evidence_id to cite.
    """
    evidence_ids = _extract_evidence_ids_from_index(evidence_index)
    if not evidence_ids:
        return []
    primary = evidence_ids[: min(6, len(evidence_ids))]
    primary_one = primary[:1]
    primary_three = primary[: min(3, len(primary))] or primary_one

    if infer_user_language(str(question or "")) == "en":
        return [
            {
                "title": "Direct Answer",
                "section_type": "narrative",
                "purpose": "Answer the user question directly, using only the provided evidence.",
                "evidence_ids": primary_three,
            },
            {
                "title": "Supporting Details",
                "section_type": "narrative",
                "purpose": "Provide key supporting details and caveats grounded in evidence.",
                "evidence_ids": primary,
            },
            {
                "title": "Graph Signals",
                "section_type": "graph_summary",
                "purpose": "Summarize graph-derived entities/relations relevant to the question, grounded in evidence.",
                "evidence_ids": primary_three,
            },
            {
                "title": "Limitations & Next Steps",
                "section_type": "meta",
                "purpose": "State evidence limitations and propose next steps without inventing facts.",
                "evidence_ids": primary_one,
            },
        ]

    return [
        {
            "title": "直接结论",
            "section_type": "narrative",
            "purpose": "直接回答用户问题，仅使用已提供的证据。",
            "evidence_ids": primary_three,
        },
        {
            "title": "关键依据",
            "section_type": "narrative",
            "purpose": "列出支撑结论的关键细节与注意事项，必须有证据支撑。",
            "evidence_ids": primary,
        },
        {
            "title": "图谱信号",
            "section_type": "graph_summary",
            "purpose": "总结与问题相关的图谱实体/关系信号，并明确证据来源。",
            "evidence_ids": primary_three,
        },
        {
            "title": "局限与下一步",
            "section_type": "meta",
            "purpose": "说明证据与覆盖的局限，并给出下一步建议，不得臆造事实。",
            "evidence_ids": primary_one,
        },
    ]


class ReportSectionSpec(BaseModel):
    """A single report section specification produced by the outline step."""

    title: str = Field(..., min_length=1)
    section_type: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)
    evidence_ids: List[str] = Field(default_factory=list)


class ReportTextPayload(BaseModel):
    """Minimal report payload produced by the report-writing LLM step."""

    text: str = Field(..., min_length=1)
    thinking: str | None = Field(default=None, description="Optional short reasoning summary (non-sensitive).")


class DeepSearchLLMReportWriter:
    """Generate report outlines and full reports using an LLM connector."""

    @staticmethod
    def _resolve_report_style(context: Dict[str, Any] | None) -> str:
        token = str((context or {}).get("report_style") or "").strip().lower()
        return token if token in {"deepsearch", "research"} else "deepsearch"

    @staticmethod
    def _language_enforcement_prompt(question: str) -> str | None:
        from config.benchmark_mode import benchmark_mode_enabled

        if benchmark_mode_enabled():
            return None
        q = str(question or "").strip()
        if not q:
            return None
        lang = infer_user_language(q)
        if lang == "en":
            return (
                "Output language policy (STRICT): The user question is in English.\n"
                "- Write titles/purposes/text in English.\n"
                "- Keep machine-readable tags (e.g., section_type) as short technical identifiers (do NOT translate them).\n"
                "- If evidence snippets are non-English, translate them into English in your writing.\n"
                "- Do NOT output Simplified/Traditional Chinese.\n"
            )
        if lang == "zh":
            return (
                "Output language policy (STRICT): The user question is in Chinese.\n"
                "- Write titles/purposes/text in Simplified Chinese.\n"
                "- Keep machine-readable tags (e.g., section_type) as short technical identifiers (do NOT translate them).\n"
                "- If evidence snippets are non-Chinese, translate them into Chinese in your writing.\n"
                "- Do NOT switch languages due to file names or document titles.\n"
            )
        return None

    @classmethod
    def _system_prompt_with_language(cls, base_prompt: str, *, question: str) -> str:
        hint = cls._language_enforcement_prompt(question)
        if not hint:
            return base_prompt
        return f"{base_prompt.rstrip()}\n\n{hint.strip()}\n"

    @classmethod
    def _system_prompt_with_style(cls, base_prompt: str, *, question: str, context: Dict[str, Any]) -> str:
        prompt = cls._system_prompt_with_language(base_prompt, question=question)
        style = cls._resolve_report_style(context)
        if style == "research":
            return f"{prompt.rstrip()}\n\n{REPORT_STYLE_RESEARCH_HINT_EN.strip()}\n"
        # Default: deepsearch (strict question-scoped report).
        return f"{prompt.rstrip()}\n\n{REPORT_STYLE_DEEPSEARCH_HINT_EN.strip()}\n"

    def __init__(
        self,
        llm_connector: Any,
        *,
        temperature: float = report_defaults.DEFAULT_REPORT_TEMPERATURE,
        max_retries: int = report_defaults.DEFAULT_REPORT_MAX_RETRIES,
        json_repair_attempts: int = report_defaults.DEFAULT_REPORT_JSON_REPAIR_ATTEMPTS,
        max_evidence_items: int | None = report_defaults.DEFAULT_REPORT_MAX_EVIDENCE_ITEMS,
        max_graph_chain_items: int | None = report_defaults.DEFAULT_REPORT_MAX_GRAPH_CHAIN_ITEMS,
        parallel_sections: bool = report_defaults.DEFAULT_REPORT_PARALLEL_SECTIONS,
        max_parallel_sections: int = report_defaults.DEFAULT_REPORT_MAX_PARALLEL_SECTIONS,
        max_section_evidence_items: int = report_defaults.DEFAULT_REPORT_MAX_SECTION_EVIDENCE_ITEMS,
        synthesis_section_max_chars: int = report_defaults.DEFAULT_REPORT_SYNTHESIS_SECTION_MAX_CHARS,
    ) -> None:
        self.llm_connector = llm_connector
        self.temperature = temperature
        self.max_retries = max_retries
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.max_evidence_items = max_evidence_items
        self.max_graph_chain_items = max_graph_chain_items
        self.parallel_sections = parallel_sections
        self.max_parallel_sections = max_parallel_sections
        self.max_section_evidence_items = max(1, int(max_section_evidence_items))
        self.synthesis_section_max_chars = int(synthesis_section_max_chars)

    async def build_outline(self, *, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Produce a JSON outline used to guide report writing."""

        highlights = context.get("highlights") or []
        evidences = context.get("evidences") or []
        graph_chain = context.get("graph_chain") or []
        evidence_index = context.get("evidence_index") or []
        if not isinstance(evidence_index, list):
            evidence_index = []

        evidence_index_json = _dump_json(evidence_index)
        available_ids = _extract_evidence_ids_from_index(evidence_index)
        user_prompt = REPORT_OUTLINE_USER_PROMPT_EN.format(
            question=question,
            highlight_count=len(highlights),
            evidence_count=len(evidences),
            graph_chain_count=len(graph_chain),
            evidence_index_json=evidence_index_json,
        )
        messages = [
            {
                "role": "system",
                "content": self._system_prompt_with_style(
                    REPORT_OUTLINE_SYSTEM_PROMPT_EN,
                    question=question,
                    context=context,
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        last_raw: str | None = None
        retries = max(int(self.max_retries), 1)
        for attempt in range(retries):
            payload = await call_llm_json_with_retry(
                llm_connector=self.llm_connector,
                messages=messages,
                expected="list",
                temperature=self.temperature,
                attempts=max(1, int(self.json_repair_attempts)),
                return_raw=True,
            )
            if isinstance(payload, tuple):
                data, raw = payload
            else:
                data, raw = payload, ""
            raw = str(raw or "")
            last_raw = raw
            if not isinstance(data, list):
                data = _safe_parse_json(raw, expected="list")
            parsed_list = False
            try:
                snippet = _extract_first_json(str(raw or "").strip())
                parsed_list = bool(snippet and snippet.lstrip().startswith("["))
            except Exception:
                parsed_list = False
            sections = _coerce_outline(data)
            if sections and available_ids:
                # Some models omit evidence_ids; fill with a small stable pool so sectionwise
                # report generation can proceed while remaining cite-first.
                fill_ids = available_ids[: min(3, len(available_ids))]
                for section in sections:
                    if not section.evidence_ids:
                        section.evidence_ids = list(fill_ids)
            missing_titles: List[str] = []
            validation_error: str | None = None
            if not sections:
                validation_error = "outline_schema_invalid"
            else:
                missing_titles = [section.title for section in sections if not section.evidence_ids]
                if missing_titles:
                    validation_error = "missing_evidence_ids"

            if validation_error is None:
                return [item.model_dump() for item in sections]

            if attempt >= retries - 1:
                # Deterministic fallback: if the outline LLM fails schema validation, we still want
                # the report pipeline to proceed as long as we have at least one citable evidence id.
                # This is not a "bypass": it remains cite-first because it only uses ids from the evidence index.
                fallback = _fallback_outline(question, evidence_index=evidence_index)
                if fallback:
                    return fallback
                raise ValueError(
                    "Report outline generation failed (cite-first requirements not satisfied). "
                    f"reason={validation_error} missing_sections={missing_titles} raw={_snippet(raw)}"
                )

            schema_hint = (
                "Return ONLY a JSON array. Each item MUST be an object with ALL keys:\n"
                "- title: string\n"
                "- section_type: string\n"
                "- purpose: string\n"
                "- evidence_ids: array of strings (chunk_id values from the evidence index; MUST be non-empty)\n"
            )
            repair_prompt = (
                "Fix the outline.\n"
                f"Reason: {validation_error}\n"
                f"Sections missing evidence_ids: {missing_titles}\n\n"
                f"{schema_hint}\n"
                "Rules:\n"
                "- Every section MUST have a non-empty evidence_ids list.\n"
                "- Use ONLY chunk_id values from the evidence index.\n"
                "- If a section cannot be supported by the provided evidence index, REMOVE that section.\n\n"
                "Evidence index (chunk_id + short summary; use these ids in the outline):\n"
                f"{evidence_index_json}\n\n"
                f"Previous (invalid) output:\n{_snippet(raw, limit=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS))}\n"
            )
            messages = [
                {
                    "role": "system",
                    "content": self._system_prompt_with_style(
                        REPORT_OUTLINE_SYSTEM_PROMPT_EN,
                        question=question,
                        context=context,
                    ),
                },
                {"role": "user", "content": repair_prompt},
            ]
        raise RuntimeError(f"Report outline generation failed after retries. raw={_snippet(last_raw or '')}")

    async def write_report(self, *, question: str, outline: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Write a structured report JSON based on the outline and the context bundle."""

        report_style = self._resolve_report_style(context)
        outline_json = _dump_json(outline)
        highlights = list(context.get("highlights") or [])
        raw_method = dict(context.get("methodology") or {})
        if context.get("final_answer"):
            raw_method["final_answer"] = context.get("final_answer")
        raw_graph_evidence = context.get("graph_evidence") or {}
        raw_coverage = context.get("coverage") or {}

        graph_chain_raw = list(context.get("graph_chain") or [])
        evidences_raw = list(context.get("evidences") or [])

        last_exc: Exception | None = None
        budget_levels = report_defaults.REPORT_WRITE_PROMPT_BUDGET_LEVELS
        budget_attempts = min(len(budget_levels), max(int(self.max_retries), 1))

        for attempt in range(budget_attempts):
            budget = budget_levels[attempt]

            method = _slim_methodology(raw_method, level=budget.methodology_level)
            graph_evidence = _slim_graph_evidence(raw_graph_evidence, level=budget.graph_evidence_level)
            coverage = _slim_coverage(raw_coverage, level=budget.coverage_level)

            highlights_limited = _limit_highlights(
                highlights,
                max_items=budget.highlights_max_items,
            )

            graph_chain_limit = _apply_divisor_limit(self.max_graph_chain_items, budget.graph_chain_items_divisor)
            graph_chain = graph_chain_raw[:graph_chain_limit] if graph_chain_limit is not None else list(graph_chain_raw)

            evidence_items_limit = _apply_divisor_limit(self.max_evidence_items, budget.evidence_items_divisor)
            evidences = _limit_evidences(evidences_raw, evidence_items_limit)
            evidence_pack = ""
            source_key_map: Dict[str, str] = {}
            if evidences:
                shared_bank = context.get("_shared_evidence_bank")
                if shared_bank is not None and hasattr(shared_bank, "add_many"):
                    bank = EvidenceBank()
                    bank.add_many(evidences)
                else:
                    bank = EvidenceBank()
                    bank.add_many(evidences)

                all_ids = bank.ids()
                q_terms = _query_terms(question, max_terms=8)
                select_enabled = bool(getattr(report_defaults, "DEFAULT_REPORT_SOURCE_SELECT_ENABLED", True))
                min_sources = int(getattr(report_defaults, "DEFAULT_REPORT_SOURCE_SELECT_MIN_SOURCES", 3))
                max_sources = int(getattr(report_defaults, "DEFAULT_REPORT_SOURCE_SELECT_MAX_SOURCES", 8))
                preview_chars = int(getattr(report_defaults, "DEFAULT_REPORT_SOURCE_SELECT_PREVIEW_CHARS", 180))
                min_sources = max(1, min(min_sources, 32))
                max_sources = max(1, min(max_sources, 32))
                if max_sources < min_sources:
                    max_sources = min_sources

                selected_ids: List[str] = []
                if select_enabled and self.llm_connector is not None and len(all_ids) > max_sources:
                    # Assign keys for selection indexing (not used for final citations).
                    all_key_map = bank.source_key_map_for_prompt(all_ids)
                    index_lines: List[str] = []
                    for key_str, ev_id in sorted(all_key_map.items(), key=lambda kv: int(kv[0])):
                        rec = bank.get(ev_id)
                        if rec is None:
                            continue
                        hint = bank._extract_provenance_hint(rec.provenance)  # noqa: SLF001 (local helper)
                        preview = _preview_head_mid_tail(str(rec.content or ""), max_chars=preview_chars)
                        signals = _page_signal_stats(str(rec.content or ""))
                        hit_preview = _term_hit_preview(str(rec.content or ""), terms=q_terms, max_terms=4)
                        hint_str = f" ({hint})" if hint else ""
                        index_lines.append(
                            f"- Source key={key_str}{hint_str} {hit_preview} "
                            f"signals=irr:{signals['irr']},%:{signals['pct']},nums:{signals['nums']}: {preview}"
                        )

                    select_prompt = REPORT_SOURCE_SELECT_USER_PROMPT_EN.format(
                        question=question,
                        source_index="\n".join(index_lines).strip(),
                    )
                    select_messages = [
                        {
                            "role": "system",
                            "content": self._system_prompt_with_style(
                                REPORT_SOURCE_SELECT_SYSTEM_PROMPT_EN,
                                question=question,
                                context=context,
                            ),
                        },
                        {"role": "user", "content": select_prompt},
                    ]
                    sel_obj = await call_llm_json_with_retry(
                        llm_connector=self.llm_connector,
                        messages=select_messages,
                        expected="dict",
                        temperature=self.temperature,
                        attempts=max(1, int(self.json_repair_attempts)),
                    )
                    if not isinstance(sel_obj, dict):
                        sel_obj = {}
                    sel_keys = _coerce_source_key_list(sel_obj.get("answer"))
                    sel_keys = sel_keys[:max_sources]
                    if len(sel_keys) < min_sources:
                        sel_keys = list(range(1, min_sources + 1))

                    chosen: List[str] = []
                    for k in sel_keys:
                        ev_id = str(all_key_map.get(str(k)) or "").strip()
                        if ev_id and ev_id in all_ids and ev_id not in chosen:
                            chosen.append(ev_id)
                    selected_ids = chosen if chosen else all_ids[:max_sources]
                else:
                    selected_ids = all_ids[:max_sources] if select_enabled else all_ids

                # Prefer contiguous pages when the model anchors on one page in a multi-page table/section.
                selected_ids = _expand_contiguous_pages(
                    bank=bank,
                    candidate_ids=all_ids,
                    selected_ids=selected_ids,
                    max_sources=max_sources,
                )

                # Re-key to a compact 1..n keyspace for writing/citations.
                selected_ids = [eid for eid in selected_ids if eid in all_ids]
                source_key_map = bank.source_key_map_for_prompt(selected_ids)
                evidence_pack = bank.evidence_pack_for_prompt(selected_ids, source_key_map=source_key_map)

            if report_style == "deepsearch":
                user_prompt = DEEPSEARCH_WRITE_USER_PROMPT_EN.format(
                    question=question,
                    outline_json=outline_json,
                    highlights_json=_dump_json(highlights_limited),
                    method_json=_dump_json(method),
                    graph_evidence_json=_dump_json(graph_evidence),
                    graph_chain_json=_dump_json(graph_chain),
                    evidence_pack=evidence_pack,
                    coverage_json=_dump_json(coverage),
                )
            else:
                user_prompt = REPORT_WRITE_USER_PROMPT_EN.format(
                    question=question,
                    outline_json=outline_json,
                    highlights_json=_dump_json(highlights_limited),
                    method_json=_dump_json(method),
                    graph_evidence_json=_dump_json(graph_evidence),
                    graph_chain_json=_dump_json(graph_chain),
                    evidence_pack=evidence_pack,
                    coverage_json=_dump_json(coverage),
                )
            verification_feedback = str(context.get("verification_feedback") or "").strip()
            if verification_feedback:
                user_prompt += f"\n\n{verification_feedback}\n"
            messages = [
                {
                    "role": "system",
                    "content": self._system_prompt_with_style(
                        REPORT_WRITE_SYSTEM_PROMPT_EN,
                        question=question,
                        context=context,
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            image_paths = collect_image_paths_from_deepsearch_evidences(
                evidences if isinstance(evidences, list) else [],
                max_images=DEEPSEARCH_MAX_IMAGE_INPUTS,
            )

            try:
                raw = await self._call(messages, phase="report", image_paths=image_paths)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_context_limit_error(exc) and attempt < budget_attempts - 1:
                    continue
                raise

            data = _safe_parse_json(raw, expected="dict")
            if not data:
                repaired = await self._attempt_json_repair(
                    base_messages=messages,
                    raw=raw,
                    error=_json_parse_error(raw, expected="dict"),
                    phase="report_repair",
                    expected="dict",
                )
                data = repaired or {}
            if not data:
                # Do not accept non-JSON report payloads. Treat JSON-only as a hard contract and retry.
                last_exc = ValueError(f"Report writing returned invalid JSON. raw={_snippet(raw)}")
                continue
            try:
                report = ReportTextPayload.model_validate(data)
            except ValidationError as exc:
                last_exc = ValueError(f"Report writing returned an invalid schema. raw={_snippet(raw)}")
                continue
            payload = report.model_dump()
            allowed_keys = _allowed_key_ints(source_key_map)
            if isinstance(payload.get("text"), str):
                text_out = str(payload.get("text") or "").strip()
                # Strict gate: reject invented graph triple chains when no graph chain edges exist.
                if not graph_chain and _has_triple_chain_lines(text_out):
                    last_exc = ValueError("report contains triple-chain lines but graph_chain is empty")
                    continue
                # Strict gate: require at least one valid inline citation when evidence exists.
                if evidence_pack and allowed_keys and not _has_any_allowed_citation(text_out, allowed_keys=allowed_keys):
                    last_exc = ValueError("report text is missing any valid inline citations")
                    continue
                # Strict gate: do not allow citations outside the allowlist.
                if allowed_keys and not _all_sup_keys_allowed(text_out, allowed_keys=allowed_keys):
                    last_exc = ValueError("report contains citation keys outside the Evidence Pack allowlist")
                    continue
                payload["text"] = text_out
            if source_key_map:
                payload["source_key_map"] = source_key_map
            payload["_evidence_pack"] = evidence_pack
            return payload

        raise RuntimeError(f"Report writing exceeded context budget: {last_exc}") from last_exc

    async def _call(
        self,
        messages: List[Dict[str, Any]],
        *,
        phase: str,
        image_paths: Optional[List[Path]] = None,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(max(self.max_retries, 1)):
            try:
                return await call_llm_with_optional_images_async(
                    self.llm_connector,
                    messages=messages,
                    user_message_index=len(messages) - 1,
                    image_paths=image_paths or [],
                    warn_context=f"deepsearch.{phase}",
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_context_limit_error(exc):
                    break
                if _is_rate_limit_error(exc) and attempt < self.max_retries - 1:
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                else:
                    await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise RuntimeError(f"Report LLM call failed during {phase}: {last_exc}") from last_exc

    @staticmethod
    def _render_sections_markdown(*, title: str, sections: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        title = str(title or "").strip()
        if title:
            blocks.append(f"# {title}")
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_title = str(section.get("title") or "").strip()
            body = str(section.get("body_markdown") or "").strip()
            if section_title and body:
                blocks.extend(["", f"## {section_title}", body])
        return "\n".join(blocks).strip()

    async def write_report_parallel(
        self, *, question: str, outline: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write a structured report by generating sections in parallel.

        This method generates each section independently and concurrently, then merges
        them into a final report structure. Best for outlines with 3+ independent sections.
        """
        shared_bank = context.get("_shared_evidence_bank")
        bank = EvidenceBank()
        raw_evidences = context.get("evidences") or []
        if isinstance(raw_evidences, list):
            limited_evidences = _limit_evidences(
                raw_evidences,
                self.max_evidence_items,
            )
            bank.add_many(limited_evidences)
        else:
            limited_evidences = []
        graph_chain = list(context.get("graph_chain") or [])
        if self.max_graph_chain_items is not None:
            graph_chain = graph_chain[: max(self.max_graph_chain_items, 0)]
        source_key_map = bank.source_key_map_for_prompt(bank.ids())

        semaphore = asyncio.Semaphore(self.max_parallel_sections)

        async def write_section_with_limit(section: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                evidence_ids = EvidenceBank.normalize_evidence_ids(section.get("evidence_ids"))
                if not evidence_ids:
                    raise ValueError("Parallel section writing requires explicit evidence_ids per section.")
                evidence_pack = bank.evidence_pack_for_prompt(
                    evidence_ids,
                    source_key_map=source_key_map,
                )
                allowed_keys = _allowed_keys_for_evidence_ids(evidence_ids, source_key_map=source_key_map)
                return await self._write_single_section(
                    question=question,
                    section=section,
                    evidence_pack=evidence_pack,
                    allowed_keys=allowed_keys,
                    graph_chain=graph_chain,
                    context=context,
                )

        tasks = [write_section_with_limit(section) for section in outline]
        results = await asyncio.gather(*tasks)

        sections: List[Dict[str, Any]] = []
        all_citations: List[Dict[str, Any]] = []
        for idx, result in enumerate(results):
            section_spec = outline[idx] if idx < len(outline) and isinstance(outline[idx], dict) else {}
            sections.append(
                {
                    "title": result.get("title") or section_spec.get("title") or "",
                    "section_type": result.get("section_type") or section_spec.get("section_type") or "",
                    "body_markdown": result.get("body_markdown") or "",
                }
            )
            for cit in result.get("citations") or []:
                if isinstance(cit, dict) and cit.get("evidence_id"):
                    cit["location_in_report"] = result.get("title")
                    all_citations.append(cit)

        title_limit = int(report_defaults.DEFAULT_PARALLEL_TITLE_MAX_CHARS)
        title = question[:title_limit] + ("..." if len(question) > title_limit else "")
        text = self._render_sections_markdown(title=title, sections=sections)
        full_evidence_pack = bank.evidence_pack_for_prompt(bank.ids(), source_key_map=source_key_map)
        return {
            "text": text,
            "citations": all_citations,
            "source_key_map": source_key_map,
            "_evidence_pack": full_evidence_pack,
        }

    async def write_report_sectionwise(
        self,
        *,
        question: str,
        outline: List[Dict[str, Any]],
        context: Dict[str, Any],
        retain_k: int = report_defaults.SECTIONWISE_RETAIN_K_DEFAULT,
    ) -> Dict[str, Any]:
        """Write a structured report section-by-section with recency retention.

        Each section retrieves only the evidence it needs, while keeping a small
        recency window of previously used evidence snippets for continuity.
        """

        shared_bank = context.get("_shared_evidence_bank")
        bank = EvidenceBank()
        raw_evidences = context.get("evidences") or []
        if isinstance(raw_evidences, list):
            limited_evidences = _limit_evidences(
                raw_evidences,
                self.max_evidence_items,
            )
            bank.add_many(limited_evidences)
        else:
            limited_evidences = []
        source_key_map = bank.source_key_map_for_prompt(bank.ids())

        graph_chain = list(context.get("graph_chain") or [])
        if self.max_graph_chain_items is not None:
            graph_chain = graph_chain[: max(self.max_graph_chain_items, 0)]

        recent_ids: List[str] = []
        used_ids_union: List[str] = []
        sections: List[Dict[str, Any]] = []
        all_citations: List[Dict[str, Any]] = []

        for idx, section in enumerate(outline, start=1):
            evidence_ids = EvidenceBank.normalize_evidence_ids(section.get("evidence_ids"))
            if not evidence_ids:
                raise ValueError("Sectionwise writer requires explicit evidence_ids per section.")
            selected_ids = [
                str(ev_id or "").strip()
                for ev_id in EvidenceBank.normalize_evidence_ids(evidence_ids)
                if str(ev_id or "").strip()
            ]
            used_ids_union.extend(selected_ids)

            merged_ids = list(selected_ids)
            if retain_k > 0 and recent_ids:
                for cid in recent_ids:
                    token = str(cid or "").strip()
                    if token and token not in merged_ids:
                        merged_ids.append(token)

            purpose = str(section.get("purpose") or "").strip()
            if sections:
                prev_count = int(report_defaults.SECTIONWISE_PREVIOUS_TITLES_MAX)
                prev_titles = [str(s.get("title") or "").strip() for s in sections[-prev_count:]]
                prev_titles = [t for t in prev_titles if t]
                if prev_titles:
                    purpose = "\n\n".join([purpose, "Previous sections (titles): " + " / ".join(prev_titles)]).strip()
            if selected_ids:
                max_ids = int(report_defaults.SECTIONWISE_PRIMARY_EVIDENCE_IDS_MAX)
                purpose = "\n\n".join(
                    [purpose, "Primary evidence_ids for this section: " + ", ".join(selected_ids[:max_ids])]
                ).strip()
            if recent_ids:
                purpose = "\n\n".join([purpose, "Recency-retained evidence_ids: " + ", ".join(recent_ids)]).strip()

            evidence_pack = bank.evidence_pack_for_prompt(
                merged_ids,
                source_key_map=source_key_map,
            )
            allowed_keys = _allowed_keys_for_evidence_ids(merged_ids, source_key_map=source_key_map)
            section_result = await self._write_single_section(
                question=question,
                section={**section, "purpose": purpose},
                evidence_pack=evidence_pack,
                allowed_keys=allowed_keys,
                graph_chain=graph_chain,
                context=context,
            )
            sections.append(
                {
                    "title": section_result.get("title") or section.get("title") or f"Section {idx}",
                    "section_type": section_result.get("section_type") or section.get("section_type") or "",
                    "body_markdown": section_result.get("body_markdown") or "",
                }
            )
            for cit in section_result.get("citations") or []:
                if isinstance(cit, dict) and cit.get("evidence_id"):
                    cit["location_in_report"] = section_result.get("title") or section.get("title")
                    all_citations.append(cit)

            recent_ids = EvidenceBank.update_recency(recent_ids, used_ids=selected_ids, retain_k=int(retain_k))

        if not used_ids_union:
            raise ValueError("Sectionwise synthesis requires at least one evidence_id.")
        title_limit = int(report_defaults.DEFAULT_PARALLEL_TITLE_MAX_CHARS)
        title = question[:title_limit] + ("..." if len(question) > title_limit else "")
        text = self._render_sections_markdown(title=title, sections=sections)

        full_evidence_pack = bank.evidence_pack_for_prompt(
            list(dict.fromkeys(used_ids_union)), source_key_map=source_key_map,
        )
        return {
            "text": text,
            "citations": all_citations,
            "source_key_map": source_key_map,
            "_evidence_pack": full_evidence_pack,
        }

    async def _write_single_section(
        self,
        *,
        question: str,
        section: Dict[str, Any],
        evidence_pack: str,
        allowed_keys: set[int],
        graph_chain: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a single section using the section-specific prompt."""
        chain_limit = len(graph_chain)
        last_exc: Exception | None = None

        for attempt in range(max(self.max_retries, 1)):
            limited_chain = graph_chain[: max(0, chain_limit)] if chain_limit else []

            user_prompt = SECTION_WRITE_USER_PROMPT_EN.format(
                question=question,
                section_title=section.get("title", ""),
                section_type=section.get("section_type", ""),
                section_purpose=section.get("purpose", ""),
                evidence_pack=evidence_pack,
                graph_chain_json=_dump_json(limited_chain),
            )
            base_messages = [
                {
                    "role": "system",
                    "content": self._system_prompt_with_style(
                        SECTION_WRITE_SYSTEM_PROMPT_EN,
                        question=question,
                        context=context,
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            try:
                raw = await self._call(base_messages, phase=f"section:{section.get('title', 'unknown')}")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                raise

            data = _safe_parse_json(raw, expected="dict")
            if not data:
                repaired = await self._attempt_json_repair(
                    base_messages=base_messages,
                    raw=raw,
                    error=_json_parse_error(raw, expected="dict"),
                    phase="section_repair",
                    expected="dict",
                )
                data = repaired or {}
            if not data:
                last_exc = ValueError(f"Section writing returned invalid JSON. raw={_snippet(raw)}")
                if attempt < self.max_retries - 1:
                    continue
                break
            body = str(data.get("body_markdown") or "").strip()
            if allowed_keys and body and not _has_supported_inline_citation(body, allowed_keys=allowed_keys):
                last_exc = ValueError("Section body is missing supported inline citations.")
                if attempt < self.max_retries - 1:
                    continue
            return {
                "title": data.get("title") or section.get("title", ""),
                "section_type": data.get("section_type") or section.get("section_type", ""),
                "body_markdown": body,
                "citations": data.get("citations") or [],
            }

        raise RuntimeError(f"Section LLM call failed: {last_exc}") from last_exc

    async def _attempt_json_repair(
        self,
        *,
        base_messages: List[Dict[str, str]],
        raw: str,
        error: str,
        phase: str,
        expected: str,
    ) -> Any:
        if self.json_repair_attempts <= 0:
            return None
        _ = phase  # keep signature stable for call sites/log context
        snippet = _snippet(raw, limit=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS))
        expected_label = "object" if expected == "dict" else ("array" if expected == "list" else expected)
        repair_prompt = JSON_REPAIR_USER_PROMPT_EN.format(
            expected_top_level=expected_label,
            error=error,
            raw_snippet=snippet,
        )
        repaired = await repair_json_from_raw_with_retry(
            llm_connector=self.llm_connector,
            messages=base_messages,
            raw=str(raw or ""),
            expected=expected,
            temperature=self.temperature,
            attempts=int(self.json_repair_attempts),
            max_raw_chars=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS),
            retry_instruction=repair_prompt,
        )
        return repaired if repaired else None


def render_markdown_from_structured(structured: Dict[str, Any]) -> str:
    """Render a Markdown report from a structured report dict."""

    text = structured.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    title = str(structured.get("title") or "DeepSearch Report").strip()
    summary = str(structured.get("short_answer") or structured.get("summary") or "").strip()
    sections = structured.get("sections") or []

    blocks: List[str] = [f"# {title}"]
    if summary:
        blocks.extend(["", summary])

    for section in sections:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or "").strip()
        body = str(section.get("body_markdown") or "").strip()
        if section_title and body:
            blocks.extend(["", f"## {section_title}", body])
    # Evidence Index is intentionally omitted from the public Markdown because the rendered
    # answer already contains inline citations and a numbered References section is appended later.

    return "\n".join(blocks).strip()


def _safe_parse_json(raw: str, *, expected: str) -> Any:
    value, _error = _try_parse_json(raw, expected=expected)
    if value is None:
        return [] if expected == "list" else {}
    return value


def _extract_first_json(text: str) -> str | None:
    return _extract_json_from_text(text)


def _try_parse_json(raw: str, *, expected: str) -> tuple[Any | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty_output"
    extracted = _extract_json_from_text(text)
    if extracted is None:
        return None, "no_json_found"
    # Prefer the shared tolerant parser first (handles common LLM JSON formatting issues).
    from core.utils.json_extract import safe_json_loads as _safe_json_loads  # local import to avoid cycles

    expected_type = expected if expected in {"dict", "list"} else None
    parsed = _safe_json_loads(extracted, expected=expected_type)
    if parsed is not None:
        return parsed, ""
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc.msg} (line {exc.lineno}, col {exc.colno})"
    if expected == "list" and not isinstance(value, list):
        return None, f"type_mismatch: expected list, got {type(value).__name__}"
    if expected == "dict" and not isinstance(value, dict):
        return None, f"type_mismatch: expected dict, got {type(value).__name__}"
    return value, ""


def _json_parse_error(raw: str, *, expected: str) -> str:
    _value, error = _try_parse_json(raw, expected=expected)
    return error or "unknown"


def _coerce_outline(raw: Any) -> List[ReportSectionSpec]:
    if not isinstance(raw, list):
        return []
    sections: List[ReportSectionSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            sections.append(ReportSectionSpec.model_validate(item))
        except ValidationError:
            continue
    return sections


def _snippet(raw: str, limit: int = report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS) -> str:
    text = (raw or "").strip().replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _apply_divisor_limit(value: int | None, divisor: int) -> int | None:
    if value is None:
        return None
    denom = max(int(divisor), 1)
    return max(int(value // denom), 0)


def _apply_divisor_limit_required(value: int, divisor: int) -> int:
    denom = max(int(divisor), 1)
    return max(int(value // denom), 0)


_RATE_LIMIT_RE = re.compile(r"(rate limit|too many requests|\\b429\\b)", re.IGNORECASE)
_CONTEXT_LIMIT_RE = re.compile(
    r"(context length|maximum context|too many tokens|context window|prompt is too long|reduce your prompt)",
    re.IGNORECASE,
)


def _is_rate_limit_error(exc: Exception) -> bool:
    return bool(_RATE_LIMIT_RE.search(str(exc)))


def _is_context_limit_error(exc: Exception) -> bool:
    return bool(_CONTEXT_LIMIT_RE.search(str(exc)))


__all__ = ["DeepSearchLLMReportWriter", "render_markdown_from_structured"]
