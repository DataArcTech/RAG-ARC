"""LLM-driven report generation for DeepSearch."""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from config.output_limits import DEEPSEARCH_TOP_CHUNKS
from core.deepsearch.memory import EvidenceBank
from core.deepsearch.tools.base import call_llm_async
from core.prompts.deepsearch.report import (
    REPORT_OUTLINE_SYSTEM_PROMPT,
    REPORT_OUTLINE_USER_PROMPT,
    REPORT_WRITE_SYSTEM_PROMPT,
    REPORT_WRITE_USER_PROMPT,
    PARALLEL_SYNTHESIS_SYSTEM_PROMPT,
    PARALLEL_SYNTHESIS_USER_PROMPT,
    SECTION_WRITE_SYSTEM_PROMPT,
    SECTION_WRITE_USER_PROMPT,
)

_CITATION_TOKEN_RE = re.compile(r"(?:\[(?P<bracket>[^\[\]]{1,128})\]|【(?P<cjk>[^【】]{1,128})】)")


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


class ReportSectionSpec(BaseModel):
    """A single report section specification produced by the outline step."""

    title: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)
    evidence_ids: List[str] = Field(default_factory=list)


class ReportSectionDraft(BaseModel):
    """A single report section draft produced by the writing step."""

    title: str = Field(..., min_length=1)
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
    summary: str = Field(..., min_length=1)
    sections: List[ReportSectionDraft] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    citations: List[ReportCitation] = Field(default_factory=list)


class DeepSearchLLMReportWriter:
    """Generate report outlines and full reports using an LLM connector."""

    def __init__(
        self,
        llm_connector: Any,
        *,
        temperature: float = 0.2,
        max_retries: int = 2,
        max_evidence_items: int | None = 24,
        max_evidence_chars: int = 900,
        max_graph_chain_items: int | None = 48,
        parallel_sections: bool = False,
        max_parallel_sections: int = 4,
        max_section_evidence_items: int = 10,
    ) -> None:
        self.llm_connector = llm_connector
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_evidence_items = max_evidence_items
        self.max_evidence_chars = max_evidence_chars
        self.max_graph_chain_items = max_graph_chain_items
        self.parallel_sections = parallel_sections
        self.max_parallel_sections = max_parallel_sections
        self.max_section_evidence_items = max(1, int(max_section_evidence_items))

    async def build_outline(self, *, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Produce a JSON outline used to guide report writing."""

        highlights = context.get("highlights") or []
        evidences = context.get("evidences") or []
        graph_chain = context.get("graph_chain") or []
        evidence_index = context.get("evidence_index") or []
        if not isinstance(evidence_index, list):
            evidence_index = []

        user_prompt = REPORT_OUTLINE_USER_PROMPT.format(
            question=question,
            highlight_count=len(highlights),
            evidence_count=len(evidences),
            graph_chain_count=len(graph_chain),
            evidence_index_json=_dump_json(evidence_index),
        )
        messages = [
            {"role": "system", "content": REPORT_OUTLINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = await self._call(messages, phase="outline")
        data = _safe_parse_json(raw, expected="list")
        sections = _coerce_outline(data)
        if not sections:
            raise ValueError(f"Report outline generation returned an invalid JSON payload. raw={_snippet(raw)}")
        return [item.model_dump() for item in sections]

    async def write_report(self, *, question: str, outline: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Write a structured report JSON based on the outline and the context bundle."""

        outline_json = _dump_json(outline)
        highlights = list(context.get("highlights") or [])
        raw_method = dict(context.get("methodology") or {})
        if context.get("final_answer"):
            raw_method["final_answer"] = context.get("final_answer")
        raw_graph_evidence = context.get("graph_evidence") or {}
        raw_coverage = context.get("coverage") or {}

        raw: str | None = None
        last_exc: Exception | None = None
        max_items = self.max_evidence_items
        max_chars = self.max_evidence_chars
        graph_chain_limit = self.max_graph_chain_items
        method_shrink = 0
        coverage_shrink = 0
        graph_shrink = 0
        highlight_limit = len(highlights)

        for attempt in range(max(self.max_retries, 1)):
            graph_chain = list(context.get("graph_chain") or [])
            if graph_chain_limit is not None:
                graph_chain = graph_chain[: max(graph_chain_limit, 0)]
            evidences = _limit_evidences(context.get("evidences") or [], max_items, max_chars)

            bounded_highlights = highlights[: max(0, int(highlight_limit))]
            method = _slim_methodology(raw_method, level=method_shrink)
            graph_evidence = _slim_graph_evidence(raw_graph_evidence, level=graph_shrink)
            coverage = _slim_coverage(raw_coverage, level=coverage_shrink)

            user_prompt = REPORT_WRITE_USER_PROMPT.format(
                question=question,
                outline_json=outline_json,
                highlights_json=_dump_json(bounded_highlights),
                method_json=_dump_json(method),
                graph_evidence_json=_dump_json(graph_evidence),
                graph_chain_json=_dump_json(graph_chain),
                evidence_json=_dump_json(evidences),
                coverage_json=_dump_json(coverage),
            )
            messages = [
                {"role": "system", "content": REPORT_WRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw = await self._call(messages, phase="report")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_context_limit_error(exc) and attempt < self.max_retries - 1:
                    if max_items is None:
                        max_items = 12
                    else:
                        max_items = max(4, max_items // 2)
                    max_chars = max(200, max_chars // 2)
                    if graph_chain_limit is None:
                        graph_chain_limit = 12
                    else:
                        graph_chain_limit = max(0, graph_chain_limit // 2)
                    if highlight_limit > 0:
                        highlight_limit = max(0, highlight_limit // 2)
                    if method_shrink < 3:
                        method_shrink += 1
                    if coverage_shrink < 3:
                        coverage_shrink += 1
                    if graph_shrink < 2:
                        graph_shrink += 1
                    continue
                raise RuntimeError(f"Report LLM call failed: {exc}") from exc

            data = _safe_parse_json(raw, expected="dict")
            if not data:
                raise ValueError(f"Report writing returned invalid JSON. raw={_snippet(raw)}")
            try:
                report = LLMStructuredReport.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"Report writing returned an invalid schema. raw={_snippet(raw)}") from exc
            return report.model_dump()

        raise RuntimeError(f"Report LLM call failed: {last_exc}") from last_exc

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
        if len(outline) <= 2:
            return await self.write_report(question=question, outline=outline, context=context)

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
                section_evidences = bank.select_evidences(
                    evidence_ids if evidence_ids else None,
                    fallback_k=min(self.max_section_evidence_items, len(bank.ids())),
                    max_chars=self.max_evidence_chars,
                )
                return await self._write_single_section(
                    question=question,
                    section=section,
                    evidences=section_evidences,
                    graph_chain=graph_chain,
                )

        tasks = [write_section_with_limit(section) for section in outline]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sections: List[Dict[str, Any]] = []
        all_citations: List[Dict[str, Any]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                sections.append({
                    "title": outline[idx].get("title", f"Section {idx + 1}"),
                    "body_markdown": f"*Section generation failed: {result}*",
                })
            else:
                sections.append({
                    "title": result.get("title") or outline[idx].get("title", ""),
                    "body_markdown": result.get("body_markdown") or "",
                })
                for cit in result.get("citations") or []:
                    if isinstance(cit, dict) and cit.get("evidence_id"):
                        cit["location_in_report"] = result.get("title")
                        all_citations.append(cit)

        title = question[:80] + ("..." if len(question) > 80 else "")
        synthesis_ids: list[str] = []
        for section in outline:
            synthesis_ids.extend(EvidenceBank.normalize_evidence_ids(section.get("evidence_ids")))
        synthesis_evidences = bank.select_evidences(
            synthesis_ids if synthesis_ids else None,
            fallback_k=min(self.max_evidence_items or 0, len(bank.ids())) if self.max_evidence_items else 0,
            max_chars=self.max_evidence_chars,
        )
        synthesis = await self._synthesize_parallel_fields(
            question=question,
            outline=outline,
            sections=sections,
            evidences=synthesis_evidences,
            coverage=context.get("coverage") or {},
        )
        if synthesis:
            title = synthesis.get("title") or title
            summary = str(synthesis.get("summary") or "").strip()
            limitations = synthesis.get("limitations") or []
            next_steps = synthesis.get("next_steps") or []
        else:
            summary = ""
            limitations = []
            next_steps = []
        if not isinstance(limitations, list):
            limitations = []
        if not isinstance(next_steps, list):
            next_steps = []
        summary_text = str(summary or "").strip()
        return {
            "title": title,
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
        retain_k: int = 5,
    ) -> Dict[str, Any]:
        """Write a structured report section-by-section with recency retention.

        Each section retrieves only the evidence it needs, while keeping a small
        recency window of previously used evidence snippets for continuity.
        """

        if len(outline) <= 2:
            return await self.write_report(question=question, outline=outline, context=context)

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
            selected = bank.select_evidences(
                evidence_ids if evidence_ids else None,
                fallback_k=min(self.max_section_evidence_items, len(bank.ids())),
                max_chars=self.max_evidence_chars,
            )
            selected_ids = [
                str(ev.get("chunk_id") or "").strip()
                for ev in selected
                if isinstance(ev, dict) and str(ev.get("chunk_id") or "").strip()
            ]
            used_ids_union.extend(selected_ids)

            retained: List[Dict[str, Any]] = []
            if retain_k > 0 and recent_ids:
                retained = bank.select_evidences(recent_ids, fallback_k=0, max_chars=self.max_evidence_chars)

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
                prev_titles = [str(s.get("title") or "").strip() for s in sections[-2:]]
                prev_titles = [t for t in prev_titles if t]
                if prev_titles:
                    purpose = "\n\n".join([purpose, "Previous sections (titles): " + " / ".join(prev_titles)]).strip()
            if selected_ids:
                purpose = "\n\n".join([purpose, "Primary evidence_ids for this section: " + ", ".join(selected_ids[:12])]).strip()
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
                    "body_markdown": section_result.get("body_markdown") or "",
                }
            )
            for cit in section_result.get("citations") or []:
                if isinstance(cit, dict) and cit.get("evidence_id"):
                    cit["location_in_report"] = section_result.get("title") or section.get("title")
                    all_citations.append(cit)

            recent_ids = EvidenceBank.update_recency(recent_ids, used_ids=selected_ids, retain_k=int(retain_k))

        synthesis_evidences = bank.select_evidences(
            used_ids_union if used_ids_union else None,
            fallback_k=min(self.max_evidence_items or 0, len(bank.ids())) if self.max_evidence_items else 0,
            max_chars=self.max_evidence_chars,
        )
        synthesis = await self._synthesize_parallel_fields(
            question=question,
            outline=outline,
            sections=sections,
            evidences=synthesis_evidences,
            coverage=context.get("coverage") or {},
        )
        title = (synthesis.get("title") if synthesis else None) or (question[:80] + ("..." if len(question) > 80 else ""))
        summary = str(synthesis.get("summary") or "").strip() if synthesis else ""
        limitations = (synthesis.get("limitations") if synthesis else None) or []
        next_steps = (synthesis.get("next_steps") if synthesis else None) or []
        if not isinstance(limitations, list):
            limitations = []
        if not isinstance(next_steps, list):
            next_steps = []

        return {
            "title": title,
            "summary": str(summary or "").strip(),
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
        """Synthesize title/summary/limitations/next_steps after parallel section drafting."""

        outline_json = _dump_json(outline)
        max_section_chars = 1200
        max_evidence_items = len(evidences)
        max_evidence_chars = int(self.max_evidence_chars)
        last_exc: Exception | None = None
        coverage_shrink = 0

        for attempt in range(max(self.max_retries, 1)):
            sections_json = _dump_json(self._compact_sections_for_synthesis(sections, max_chars=max_section_chars))
            limited_evidences = _limit_evidences(
                evidences,
                max_evidence_items if max_evidence_items else None,
                max_evidence_chars,
            )
            evidence_json = _dump_json(limited_evidences)
            coverage_json = _dump_json(_slim_coverage(coverage or {}, level=coverage_shrink))

            user_prompt = PARALLEL_SYNTHESIS_USER_PROMPT.format(
                question=question,
                outline_json=outline_json,
                sections_json=sections_json,
                evidence_json=evidence_json,
                coverage_json=coverage_json,
            )
            messages = [
                {"role": "system", "content": PARALLEL_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                raw = await self._call(messages, phase="parallel_synthesis")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_context_limit_error(exc) and attempt < self.max_retries - 1:
                    if max_section_chars > 400:
                        max_section_chars = max(400, max_section_chars // 2)
                    if max_evidence_items and max_evidence_items > 6:
                        max_evidence_items = max(6, max_evidence_items // 2)
                    if max_evidence_chars and max_evidence_chars > 200:
                        max_evidence_chars = max(200, max_evidence_chars // 2)
                    if coverage_shrink < 3:
                        coverage_shrink += 1
                    continue
                raise

            parsed = _safe_parse_json(raw, expected="dict")
            if not isinstance(parsed, dict):
                return {}
            allowed_ids = {
                str(item.get("chunk_id") or "").strip()
                for item in limited_evidences
                if isinstance(item, dict) and str(item.get("chunk_id") or "").strip()
            }
            summary = str(parsed.get("summary") or "").strip()
            if allowed_ids and summary and not _has_supported_inline_citation(summary, allowed_ids=allowed_ids):
                last_exc = ValueError("Synthesis summary is missing supported inline citations.")
                if attempt < self.max_retries - 1:
                    continue
                parsed["summary"] = ""
            return parsed

        raise RuntimeError(f"Parallel synthesis LLM call failed: {last_exc}") from last_exc

    @staticmethod
    def _compact_sections_for_synthesis(sections: List[Dict[str, Any]], *, max_chars: int = 1200) -> List[Dict[str, str]]:
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

    @staticmethod
    def _heuristic_summary(*, question: str, sections: List[Dict[str, Any]]) -> str:
        for section in sections:
            if not isinstance(section, dict):
                continue
            body = str(section.get("body_markdown") or "").strip()
            if body:
                return body.splitlines()[0].strip()
        return str(question or "").strip()

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
                section_purpose=section.get("purpose", ""),
                evidence_json=_dump_json(limited_evidences),
                graph_chain_json=_dump_json(limited_chain),
            )
            messages = [
                {"role": "system", "content": SECTION_WRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                raw = await self._call(messages, phase=f"section:{section.get('title', 'unknown')}")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_context_limit_error(exc) and attempt < self.max_retries - 1:
                    if max_items and max_items > 2:
                        max_items = max(2, max_items // 2)
                    if max_chars and max_chars > 200:
                        max_chars = max(200, max_chars // 2)
                    if chain_limit and chain_limit > 6:
                        chain_limit = max(6, chain_limit // 2)
                    continue
                raise

            data = _safe_parse_json(raw, expected="dict")
            if not data:
                return {"title": section.get("title", ""), "body_markdown": "", "citations": []}
            body = str(data.get("body_markdown") or "").strip()
            if allowed_ids and body and not _has_supported_inline_citation(body, allowed_ids=allowed_ids):
                last_exc = ValueError("Section body is missing supported inline citations.")
                if attempt < self.max_retries - 1:
                    continue
            return {
                "title": data.get("title") or section.get("title", ""),
                "body_markdown": body,
                "citations": data.get("citations") or [],
            }

        raise RuntimeError(f"Section LLM call failed: {last_exc}") from last_exc


def render_markdown_from_structured(structured: Dict[str, Any]) -> str:
    """Render a Markdown report from a structured report dict."""

    title = str(structured.get("title") or "DeepSearch Report").strip()
    summary = str(structured.get("summary") or "").strip()
    sections = structured.get("sections") or []
    limitations = structured.get("limitations") or []
    next_steps = structured.get("next_steps") or []
    citations = structured.get("citations") or []

    def _has_cjk(text: str) -> bool:
        for ch in text:
            code = ord(ch)
            if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
                return True
        return False
    is_cjk = _has_cjk(title) or _has_cjk(summary)

    blocks: List[str] = [f"# {title}"]
    if summary:
        blocks.extend(["", ("## 结论" if is_cjk else "## Answer"), summary])

    for section in sections:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or "").strip()
        body = str(section.get("body_markdown") or "").strip()
        if section_title and body:
            blocks.extend(["", f"## {section_title}", body])

    if limitations:
        blocks.extend(
            [
                "",
                ("## 局限" if is_cjk else "## Limitations"),
                "\n".join(f"- {item}" for item in limitations if str(item).strip()),
            ]
        )
    if next_steps:
        blocks.extend(
            [
                "",
                ("## 下一步" if is_cjk else "## Next Steps"),
                "\n".join(f"- {item}" for item in next_steps if str(item).strip()),
            ]
        )
    # Evidence Index is intentionally omitted from the public Markdown because the rendered
    # answer already contains inline citations and a numbered References section is appended later.

    return "\n".join(blocks).strip()


def _safe_parse_json(raw: str, *, expected: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return [] if expected == "list" else {}
    parsed = _extract_first_json(text)
    if parsed is None:
        return [] if expected == "list" else {}
    try:
        value = json.loads(parsed)
    except json.JSONDecodeError:
        return [] if expected == "list" else {}
    if expected == "list" and isinstance(value, list):
        return value
    if expected == "dict" and isinstance(value, dict):
        return value
    return [] if expected == "list" else {}


def _extract_first_json(text: str) -> str | None:
    start = text.find("{")
    start_list = text.find("[")
    if start == -1 or (0 <= start_list < start):
        start = start_list
    if start == -1:
        return None

    open_brace = {"{": "}", "[": "]"}
    opener = text[start]
    closer = open_brace.get(opener)
    if closer is None:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


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
        for step in plan_steps[:12]:
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
        for step in reasoning_steps[:24]:
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
        for step in plan_steps[:10]:
            if isinstance(step, dict):
                slim_plan.append({"step_id": step.get("step_id"), "description": step.get("description")})
        slim_reasoning = []
        for step in reasoning_steps[:16]:
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
        return {"seed_entities": seed_entities[:12], "graph_stats": graph_stats}
    return {"seed_entities": seed_entities[:12]}


def _slim_coverage(coverage: Any, *, level: int) -> Dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    normalized_level = max(0, int(level))
    if normalized_level <= 0:
        # Still strip large/potentially noisy lists to avoid accidental prompt blowups.
        trimmed = dict(coverage)
        pending = trimmed.get("pending_external")
        if isinstance(pending, list) and len(pending) > 6:
            trimmed["pending_external"] = pending[:6]
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


def _snippet(raw: str, limit: int = 500) -> str:
    text = (raw or "").strip().replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "…"
    return text


_RATE_LIMIT_RE = re.compile(r"(rate limit|too many requests|\\b429\\b)", re.IGNORECASE)
_CONTEXT_LIMIT_RE = re.compile(
    r"(context length|maximum context|too many tokens|context window|prompt is too long|reduce your prompt)",
    re.IGNORECASE,
)


def _is_rate_limit_error(exc: Exception) -> bool:
    return bool(_RATE_LIMIT_RE.search(str(exc)))


def _is_context_limit_error(exc: Exception) -> bool:
    return bool(_CONTEXT_LIMIT_RE.search(str(exc)))


def _is_context_limit_error(exc: Exception) -> bool:
    return bool(_CONTEXT_LIMIT_RE.search(str(exc)))


__all__ = ["DeepSearchLLMReportWriter", "render_markdown_from_structured"]
