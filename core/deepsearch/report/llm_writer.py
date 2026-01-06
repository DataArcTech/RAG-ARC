"""LLM-driven report generation for DeepSearch."""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from config.core.deepsearch import report_writer_defaults as report_defaults
from core.deepsearch.memory import EvidenceBank
from core.deepsearch.tools.base import call_llm_async
from core.utils.json_extract import extract_json_from_text as _extract_json_from_text
from core.utils.text_regex import CJK_DETECT_RE
from core.utils.text_regex import INLINE_CITATION_TOKEN_RE
from core.deepsearch.utils.language_policy import infer_user_language
from core.prompts.deepsearch.report import (
    REPORT_OUTLINE_SYSTEM_PROMPT,
    REPORT_OUTLINE_USER_PROMPT,
    REPORT_WRITE_SYSTEM_PROMPT,
    REPORT_WRITE_USER_PROMPT,
    PARALLEL_SYNTHESIS_SYSTEM_PROMPT,
    PARALLEL_SYNTHESIS_USER_PROMPT,
    SECTION_WRITE_SYSTEM_PROMPT,
    SECTION_WRITE_USER_PROMPT,
    JSON_REPAIR_USER_PROMPT,
)

_CITATION_TOKEN_RE = INLINE_CITATION_TOKEN_RE


def _has_supported_inline_citation(text: str, *, allowed_ids: set[str]) -> bool:
    if not text or not allowed_ids:
        return False
    for match in _CITATION_TOKEN_RE.finditer(text):
        token = (match.group("bracket") or match.group("cjk") or "").strip()
        if not token:
            continue
        candidates = re.split(r"[,\s]+", token)
        for candidate in candidates:
            cand = candidate.strip()
            if cand and cand in allowed_ids:
                return True
    return False


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


class ReportSectionDraft(BaseModel):
    """A single report section draft produced by the writing step."""

    title: str = Field(..., min_length=1)
    section_type: str = Field(..., min_length=1)
    body_markdown: str = Field(..., min_length=1)


class ReportCitation(BaseModel):
    """A citation entry linking an evidence ID to a usage rationale."""

    evidence_id: str = Field(..., min_length=1)
    source_type: Optional[str] = None
    source: Optional[str] = None
    used_for: str = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    location_in_report: Optional[str] = None


class LLMStructuredReport(BaseModel):
    """A structured report produced by the report-writing LLM step."""

    title: str = Field(..., min_length=1)
    short_answer: str = Field(..., min_length=1)
    # Backward-compat shim: some callers may still look for `summary`.
    summary: Optional[str] = None
    sections: List[ReportSectionDraft] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    citations: List[ReportCitation] = Field(default_factory=list)


class DeepSearchLLMReportWriter:
    """Generate report outlines and full reports using an LLM connector."""

    @staticmethod
    def _language_enforcement_prompt(question: str) -> str | None:
        q = str(question or "").strip()
        if not q:
            return None
        lang = infer_user_language(q)
        if lang == "en":
            return (
                "Output language policy (STRICT): The user question is in English.\n"
                "- Write ALL fields in English (title, short_answer, sections, limitations, next_steps).\n"
                "- If evidence snippets are non-English, translate them into English in your writing.\n"
                "- Do NOT output Simplified/Traditional Chinese.\n"
            )
        if lang == "zh":
            return (
                "Output language policy (STRICT): The user question is in Chinese.\n"
                "- Write ALL fields in Simplified Chinese (title, short_answer, sections, limitations, next_steps).\n"
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

    def __init__(
        self,
        llm_connector: Any,
        *,
        temperature: float = report_defaults.DEFAULT_REPORT_TEMPERATURE,
        max_retries: int = report_defaults.DEFAULT_REPORT_MAX_RETRIES,
        json_repair_attempts: int = report_defaults.DEFAULT_REPORT_JSON_REPAIR_ATTEMPTS,
        max_evidence_items: int | None = report_defaults.DEFAULT_REPORT_MAX_EVIDENCE_ITEMS,
        max_evidence_chars: int = report_defaults.DEFAULT_REPORT_MAX_EVIDENCE_CHARS,
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
        self.max_evidence_chars = max_evidence_chars
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
        user_prompt = REPORT_OUTLINE_USER_PROMPT.format(
            question=question,
            highlight_count=len(highlights),
            evidence_count=len(evidences),
            graph_chain_count=len(graph_chain),
            evidence_index_json=evidence_index_json,
        )
        messages = [
            {"role": "system", "content": self._system_prompt_with_language(REPORT_OUTLINE_SYSTEM_PROMPT, question=question)},
            {"role": "user", "content": user_prompt},
        ]
        last_raw: str | None = None
        retries = max(int(self.max_retries), 1)
        for attempt in range(retries):
            raw = await self._call(messages, phase="outline")
            last_raw = raw
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
                fallback = _fallback_outline(question, evidence_index=evidence_index) if parsed_list else []
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
                "Evidence index (id + short summary; cite these ids in the outline):\n"
                f"{evidence_index_json}\n\n"
                f"Previous (invalid) output:\n{_snippet(raw, limit=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS))}\n"
            )
            messages = [
                {"role": "system", "content": self._system_prompt_with_language(REPORT_OUTLINE_SYSTEM_PROMPT, question=question)},
                {"role": "user", "content": repair_prompt},
            ]
        raise RuntimeError(f"Report outline generation failed after retries. raw={_snippet(last_raw or '')}")

    async def write_report(self, *, question: str, outline: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Write a structured report JSON based on the outline and the context bundle."""

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
                text_max_chars=budget.highlight_text_max_chars,
            )

            graph_chain_limit = _apply_divisor_limit(self.max_graph_chain_items, budget.graph_chain_items_divisor)
            graph_chain = graph_chain_raw[:graph_chain_limit] if graph_chain_limit is not None else list(graph_chain_raw)

            evidence_items_limit = _apply_divisor_limit(self.max_evidence_items, budget.evidence_items_divisor)
            evidence_chars_limit = _apply_divisor_limit_required(self.max_evidence_chars, budget.evidence_chars_divisor)
            evidences = _limit_evidences(evidences_raw, evidence_items_limit, evidence_chars_limit)

            user_prompt = REPORT_WRITE_USER_PROMPT.format(
                question=question,
                outline_json=outline_json,
                highlights_json=_dump_json(highlights_limited),
                method_json=_dump_json(method),
                graph_evidence_json=_dump_json(graph_evidence),
                graph_chain_json=_dump_json(graph_chain),
                evidence_json=_dump_json(evidences),
                coverage_json=_dump_json(coverage),
            )
            messages = [
                {"role": "system", "content": self._system_prompt_with_language(REPORT_WRITE_SYSTEM_PROMPT, question=question)},
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw = await self._call(messages, phase="report")
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
                raise ValueError(f"Report writing returned invalid JSON. raw={_snippet(raw)}")
            try:
                report = LLMStructuredReport.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"Report writing returned an invalid schema. raw={_snippet(raw)}") from exc
            return report.model_dump()

        raise RuntimeError(f"Report writing exceeded context budget: {last_exc}") from last_exc

    async def _call(self, messages: List[Dict[str, str]], *, phase: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(max(self.max_retries, 1)):
            try:
                return await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_context_limit_error(exc):
                    break
                if _is_rate_limit_error(exc) and attempt < self.max_retries - 1:
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                else:
                    await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise RuntimeError(f"Report LLM call failed during {phase}: {last_exc}") from last_exc

    async def write_report_parallel(
        self, *, question: str, outline: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write a structured report by generating sections in parallel.

        This method generates each section independently and concurrently, then merges
        them into a final report structure. Best for outlines with 3+ independent sections.
        """
        bank = EvidenceBank()
        raw_evidences = context.get("evidences") or []
        if isinstance(raw_evidences, list):
            bank.add_many(raw_evidences)
        graph_chain = list(context.get("graph_chain") or [])
        if self.max_graph_chain_items is not None:
            graph_chain = graph_chain[: max(self.max_graph_chain_items, 0)]

        semaphore = asyncio.Semaphore(self.max_parallel_sections)

        async def write_section_with_limit(section: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                evidence_ids = EvidenceBank.normalize_evidence_ids(section.get("evidence_ids"))
                if not evidence_ids:
                    raise ValueError("Parallel section writing requires explicit evidence_ids per section.")
                section_evidences = bank.select_evidences(evidence_ids, max_chars=self.max_evidence_chars)
                return await self._write_single_section(
                    question=question,
                    section=section,
                    evidences=section_evidences,
                    graph_chain=graph_chain,
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
        synthesis_ids: list[str] = []
        for section in outline:
            synthesis_ids.extend(EvidenceBank.normalize_evidence_ids(section.get("evidence_ids")))
        if not synthesis_ids:
            raise ValueError("Parallel report synthesis requires evidence_ids from the outline.")
        synthesis_evidences = bank.select_evidences(synthesis_ids, max_chars=self.max_evidence_chars)
        synthesis = await self._synthesize_parallel_fields(
            question=question,
            outline=outline,
            sections=sections,
            evidences=synthesis_evidences,
            coverage=context.get("coverage") or {},
        )
        if synthesis:
            title = synthesis.get("title") or title
            short_answer = str(synthesis.get("short_answer") or synthesis.get("summary") or "").strip()
            limitations = synthesis.get("limitations") or []
            next_steps = synthesis.get("next_steps") or []
        else:
            short_answer = ""
            limitations = []
            next_steps = []
        if not isinstance(limitations, list):
            limitations = []
        if not isinstance(next_steps, list):
            next_steps = []
        summary_text = str(short_answer or "").strip()
        return {
            "title": title,
            "short_answer": summary_text,
            "summary": summary_text,
            "sections": sections,
            "limitations": [str(item) for item in limitations if str(item).strip()],
            "next_steps": [str(item) for item in next_steps if str(item).strip()],
            "citations": all_citations,
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

        bank = EvidenceBank()
        raw_evidences = context.get("evidences") or []
        if isinstance(raw_evidences, list):
            bank.add_many(raw_evidences)

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
            selected = bank.select_evidences(evidence_ids, max_chars=self.max_evidence_chars)
            selected_ids = [
                str(ev.get("chunk_id") or "").strip()
                for ev in selected
                if isinstance(ev, dict) and str(ev.get("chunk_id") or "").strip()
            ]
            used_ids_union.extend(selected_ids)

            retained: List[Dict[str, Any]] = []
            if retain_k > 0 and recent_ids:
                retained = bank.select_evidences(recent_ids, max_chars=self.max_evidence_chars)

            merged_evidences = list(selected)
            seen = {ev.get("chunk_id") for ev in merged_evidences if isinstance(ev, dict)}
            for ev in retained:
                cid = ev.get("chunk_id") if isinstance(ev, dict) else None
                if cid and cid in seen:
                    continue
                merged_evidences.append(ev)
                if cid:
                    seen.add(cid)

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

            section_result = await self._write_single_section(
                question=question,
                section={**section, "purpose": purpose},
                evidences=merged_evidences,
                graph_chain=graph_chain,
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
        synthesis_evidences = bank.select_evidences(used_ids_union, max_chars=self.max_evidence_chars)
        synthesis = await self._synthesize_parallel_fields(
            question=question,
            outline=outline,
            sections=sections,
            evidences=synthesis_evidences,
            coverage=context.get("coverage") or {},
        )
        title_limit = int(report_defaults.DEFAULT_PARALLEL_TITLE_MAX_CHARS)
        title = (synthesis.get("title") if synthesis else None) or (
            question[:title_limit] + ("..." if len(question) > title_limit else "")
        )
        short_answer = str((synthesis.get("short_answer") if synthesis else None) or (synthesis.get("summary") if synthesis else None) or "").strip()
        limitations = (synthesis.get("limitations") if synthesis else None) or []
        next_steps = (synthesis.get("next_steps") if synthesis else None) or []
        if not isinstance(limitations, list):
            limitations = []
        if not isinstance(next_steps, list):
            next_steps = []

        return {
            "title": title,
            "short_answer": short_answer,
            "summary": short_answer,
            "sections": sections,
            "limitations": [str(item) for item in limitations if str(item).strip()],
            "next_steps": [str(item) for item in next_steps if str(item).strip()],
            "citations": all_citations,
        }

    async def _synthesize_parallel_fields(
        self,
        *,
        question: str,
        outline: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
        coverage: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize title/short_answer/limitations/next_steps after parallel section drafting."""

        outline_json = _dump_json(outline)
        sections_json = _dump_json(self._compact_sections_for_synthesis(sections, max_chars=self.synthesis_section_max_chars))
        limited_evidences = _limit_evidences(evidences, len(evidences), int(self.max_evidence_chars))
        evidence_json = _dump_json(limited_evidences)
        coverage_json = _dump_json(_slim_coverage(coverage or {}, level=0))

        user_prompt = PARALLEL_SYNTHESIS_USER_PROMPT.format(
            question=question,
            outline_json=outline_json,
            sections_json=sections_json,
            evidence_json=evidence_json,
            coverage_json=coverage_json,
        )
        messages = [
            {"role": "system", "content": self._system_prompt_with_language(PARALLEL_SYNTHESIS_SYSTEM_PROMPT, question=question)},
            {"role": "user", "content": user_prompt},
        ]
        raw = await self._call(messages, phase="parallel_synthesis")
        parsed = _safe_parse_json(raw, expected="dict")
        if not isinstance(parsed, dict) or not parsed:
            repaired = await self._attempt_json_repair(
                base_messages=messages,
                raw=raw,
                error=_json_parse_error(raw, expected="dict"),
                phase="parallel_synthesis_repair",
                expected="dict",
            )
            parsed = repaired or {}
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(f"Parallel synthesis returned invalid JSON. raw={_snippet(raw)}")
        allowed_ids = {
            str(item.get("chunk_id") or "").strip()
            for item in limited_evidences
            if isinstance(item, dict) and str(item.get("chunk_id") or "").strip()
        }
        short_answer = str(parsed.get("short_answer") or parsed.get("summary") or "").strip()
        if allowed_ids and short_answer and not _has_supported_inline_citation(short_answer, allowed_ids=allowed_ids):
            raise ValueError("Synthesis short_answer is missing supported inline citations.")
        parsed["short_answer"] = short_answer
        parsed["summary"] = short_answer
        return parsed

    @staticmethod
    def _compact_sections_for_synthesis(sections: List[Dict[str, Any]], *, max_chars: int) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            body = str(section.get("body_markdown") or "").strip()
            if max_chars > 0 and len(body) > max_chars:
                body = body[:max_chars].rstrip() + "..."
            if title or body:
                payload.append({"title": title, "body_markdown": body})
        return payload

    async def _write_single_section(
        self,
        *,
        question: str,
        section: Dict[str, Any],
        evidences: List[Dict[str, Any]],
        graph_chain: List[str],
    ) -> Dict[str, Any]:
        """Generate a single section using the section-specific prompt."""
        max_items = min(max(1, self.max_section_evidence_items), len(evidences) or 0) if evidences else 0
        max_chars = int(self.max_evidence_chars)
        chain_limit = len(graph_chain)
        last_exc: Exception | None = None

        for attempt in range(max(self.max_retries, 1)):
            limited_evidences = _limit_evidences(evidences, max_items if max_items else None, max_chars)
            limited_chain = graph_chain[: max(0, chain_limit)] if chain_limit else []
            allowed_ids = {
                str(item.get("chunk_id") or "").strip()
                for item in limited_evidences
                if isinstance(item, dict) and str(item.get("chunk_id") or "").strip()
            }

            user_prompt = SECTION_WRITE_USER_PROMPT.format(
                question=question,
                section_title=section.get("title", ""),
                section_type=section.get("section_type", ""),
                section_purpose=section.get("purpose", ""),
                evidence_json=_dump_json(limited_evidences),
                graph_chain_json=_dump_json(limited_chain),
            )
            base_messages = [
                {"role": "system", "content": self._system_prompt_with_language(SECTION_WRITE_SYSTEM_PROMPT, question=question)},
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
            if allowed_ids and body and not _has_supported_inline_citation(body, allowed_ids=allowed_ids):
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
        snippet = _snippet(raw, limit=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS))
        expected_label = "object" if expected == "dict" else ("array" if expected == "list" else expected)
        repair_prompt = JSON_REPAIR_USER_PROMPT.format(
            expected_top_level=expected_label,
            error=error,
            raw_snippet=snippet,
        )
        messages = base_messages + [{"role": "assistant", "content": str(raw or "")}, {"role": "user", "content": repair_prompt}]
        last_raw = raw
        for attempt in range(int(self.json_repair_attempts)):
            last_raw = await self._call(messages, phase=f"{phase}:{attempt + 1}")
            parsed = _safe_parse_json(last_raw, expected=expected)
            if parsed:
                return parsed
            # Keep the most recent output in the repair thread.
            messages = base_messages + [
                {"role": "assistant", "content": str(last_raw or "")},
                {"role": "user", "content": repair_prompt},
            ]
        return None


def render_markdown_from_structured(structured: Dict[str, Any]) -> str:
    """Render a Markdown report from a structured report dict."""

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


def _dump_json(payload: Any) -> str:
    """Token-efficient JSON rendering for prompts (no pretty indent)."""

    def _default(value: Any) -> str:
        return str(value)

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_default)


def _slim_methodology(method: Dict[str, Any], *, level: int) -> Dict[str, Any]:
    """Shrink methodology payload for prompt budgeting."""

    if not isinstance(method, dict):
        return {}
    normalized_level = max(0, int(level))

    final_answer = method.get("final_answer")
    plan_steps = method.get("plan_steps") if isinstance(method.get("plan_steps"), list) else []
    reasoning_steps = method.get("reasoning_steps") if isinstance(method.get("reasoning_steps"), list) else []
    tool_results = method.get("tool_results") if isinstance(method.get("tool_results"), list) else []

    if normalized_level <= 0:
        return method

    if normalized_level == 1:
        slim_plan = []
        for step in plan_steps[: report_defaults.SLIM_METHOD_LEVEL_1_PLAN_STEPS]:
            if isinstance(step, dict):
                slim_plan.append(
                    {
                        "step_id": step.get("step_id"),
                        "description": step.get("description"),
                        "channel": step.get("channel"),
                        "tool": step.get("tool"),
                    }
                )
        slim_reasoning = []
        for step in reasoning_steps[: report_defaults.SLIM_METHOD_LEVEL_1_REASONING_STEPS]:
            if isinstance(step, dict):
                slim_reasoning.append(
                    {
                        "step_id": step.get("step_id"),
                        "description": step.get("description"),
                        "status": step.get("status"),
                        "output_summary": step.get("output_summary"),
                        "tool": step.get("tool"),
                    }
                )
        payload: Dict[str, Any] = {"plan_steps": slim_plan, "reasoning_steps": slim_reasoning}
        if final_answer:
            payload["final_answer"] = final_answer
        return payload

    if normalized_level == 2:
        slim_plan = []
        for step in plan_steps[: report_defaults.SLIM_METHOD_LEVEL_2_PLAN_STEPS]:
            if isinstance(step, dict):
                slim_plan.append({"step_id": step.get("step_id"), "description": step.get("description")})
        slim_reasoning = []
        for step in reasoning_steps[: report_defaults.SLIM_METHOD_LEVEL_2_REASONING_STEPS]:
            if isinstance(step, dict):
                slim_reasoning.append(
                    {"step_id": step.get("step_id"), "output_summary": step.get("output_summary"), "tool": step.get("tool")}
                )
        payload = {"plan_steps": slim_plan, "reasoning_steps": slim_reasoning}
        if final_answer:
            payload["final_answer"] = final_answer
        return payload

    payload = {}
    if final_answer:
        payload["final_answer"] = final_answer
    return payload


def _slim_graph_evidence(graph_evidence: Any, *, level: int) -> Dict[str, Any]:
    if not isinstance(graph_evidence, dict):
        return {}
    normalized_level = max(0, int(level))
    if normalized_level <= 0:
        return graph_evidence
    seed_entities = graph_evidence.get("seed_entities") if isinstance(graph_evidence.get("seed_entities"), list) else []
    graph_stats = graph_evidence.get("graph_stats") if isinstance(graph_evidence.get("graph_stats"), dict) else {}
    if normalized_level == 1:
        return {
            "seed_entities": seed_entities[: report_defaults.SLIM_GRAPH_EVIDENCE_LEVEL_SEED_ENTITIES],
            "graph_stats": graph_stats,
        }
    return {"seed_entities": seed_entities[: report_defaults.SLIM_GRAPH_EVIDENCE_LEVEL_SEED_ENTITIES]}


def _slim_coverage(coverage: Any, *, level: int) -> Dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    normalized_level = max(0, int(level))
    if normalized_level <= 0:
        # Still strip large/potentially noisy lists to avoid accidental prompt blowups.
        trimmed = dict(coverage)
        pending = trimmed.get("pending_external")
        max_pending = int(report_defaults.SLIM_COVERAGE_BASE_PENDING_EXTERNAL_MAX_ITEMS)
        if isinstance(pending, list) and len(pending) > max_pending:
            trimmed["pending_external"] = pending[:max_pending]
            trimmed["pending_external_truncated"] = True
        return trimmed

    gap_result = coverage.get("gap_result") if isinstance(coverage.get("gap_result"), dict) else {}
    metrics = coverage.get("coverage_metrics") if isinstance(coverage.get("coverage_metrics"), dict) else {}

    if normalized_level == 1:
        return {
            "coverage_metrics": {
                "coverage_score": metrics.get("coverage_score"),
                "confidence_score": metrics.get("confidence_score"),
                "coverage_ratio": metrics.get("coverage_ratio"),
                "evidence_count": metrics.get("evidence_count"),
                "missing_topics": metrics.get("missing_topics") or [],
            },
            "gap_result": {
                "coverage_score": gap_result.get("coverage_score"),
                "confidence_score": gap_result.get("confidence_score"),
                "should_trigger_external": gap_result.get("should_trigger_external"),
                "missing_topics": gap_result.get("missing_topics") or [],
                "reason": gap_result.get("reason"),
            },
        }

    if normalized_level == 2:
        return {
            "gap_result": {
                "should_trigger_external": gap_result.get("should_trigger_external"),
                "missing_topics": gap_result.get("missing_topics") or [],
                "reason": gap_result.get("reason"),
            }
        }

    return {}


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


def _limit_evidences(evidences: List[Dict[str, Any]], max_items: int | None, max_chars: int) -> List[Dict[str, Any]]:
    limited: List[Dict[str, Any]] = []
    subset = list(evidences) if max_items is None else evidences[: max(max_items, 0)]
    for entry in subset:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("chunk_id")
        content = str(entry.get("content") or "")
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        limited.append(
            {
                "chunk_id": chunk_id,
                "source": entry.get("source"),
                "content": content,
                "score": entry.get("score"),
            }
        )
    return limited


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


def _limit_highlights(highlights: list[Any], *, max_items: int | None, text_max_chars: int | None) -> list[dict[str, Any]]:
    if not isinstance(highlights, list) or not highlights:
        return []
    subset = highlights if max_items is None else highlights[: max(int(max_items), 0)]
    limited: list[dict[str, Any]] = []
    for item in subset:
        if not isinstance(item, dict):
            continue
        payload: dict[str, Any] = dict(item)
        if text_max_chars is not None:
            limit = max(int(text_max_chars), 0)
            for key in ("summary", "content", "details", "text"):
                value = payload.get(key)
                if isinstance(value, str) and limit > 0 and len(value) > limit:
                    payload[key] = value[:limit].rstrip() + "..."
        limited.append(payload)
    return limited


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
