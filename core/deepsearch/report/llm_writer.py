"""LLM-driven report generation for DeepSearch."""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from config.output_limits import DEEPSEARCH_TOP_CHUNKS
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


class ReportSectionSpec(BaseModel):
    """A single report section specification produced by the outline step."""

    title: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)


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
    ) -> None:
        self.llm_connector = llm_connector
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_evidence_items = max_evidence_items
        self.max_evidence_chars = max_evidence_chars
        self.max_graph_chain_items = max_graph_chain_items
        self.parallel_sections = parallel_sections
        self.max_parallel_sections = max_parallel_sections

    async def build_outline(self, *, question: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Produce a JSON outline used to guide report writing."""

        highlights = context.get("highlights") or []
        evidences = context.get("evidences") or []
        graph_chain = context.get("graph_chain") or []

        user_prompt = REPORT_OUTLINE_USER_PROMPT.format(
            question=question,
            highlight_count=len(highlights),
            evidence_count=len(evidences),
            graph_chain_count=len(graph_chain),
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

    async def write_report(self, *, question: str, outline: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Write a structured report JSON based on the outline and the context bundle."""

        outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
        highlights = context.get("highlights") or []
        method = dict(context.get("methodology") or {})
        if context.get("final_answer"):
            method["final_answer"] = context.get("final_answer")
        graph_evidence = context.get("graph_evidence") or {}
        coverage = context.get("coverage") or {}

        raw: str | None = None
        last_exc: Exception | None = None
        max_items = self.max_evidence_items
        max_chars = self.max_evidence_chars
        graph_chain_limit = self.max_graph_chain_items

        for attempt in range(max(self.max_retries, 1)):
            graph_chain = list(context.get("graph_chain") or [])
            if graph_chain_limit is not None:
                graph_chain = graph_chain[: max(graph_chain_limit, 0)]
            evidences = _limit_evidences(context.get("evidences") or [], max_items, max_chars)

            user_prompt = REPORT_WRITE_USER_PROMPT.format(
                question=question,
                outline_json=outline_json,
                highlights_json=json.dumps(highlights, ensure_ascii=False, indent=2),
                method_json=json.dumps(method, ensure_ascii=False, indent=2),
                graph_evidence_json=json.dumps(graph_evidence, ensure_ascii=False, indent=2),
                graph_chain_json=json.dumps(graph_chain, ensure_ascii=False, indent=2),
                evidence_json=json.dumps(evidences, ensure_ascii=False, indent=2),
                coverage_json=json.dumps(coverage, ensure_ascii=False, indent=2),
            )
            messages = [
                {"role": "system", "content": REPORT_WRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
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
                    continue
                if _is_rate_limit_error(exc) and attempt < self.max_retries - 1:
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                    continue
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2.0, 0.5 * (attempt + 1)))
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
                if _is_rate_limit_error(exc) and attempt < self.max_retries - 1:
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                else:
                    await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise RuntimeError(f"Report LLM call failed during {phase}: {last_exc}") from last_exc

    async def write_report_parallel(
        self, *, question: str, outline: List[Dict[str, str]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write a structured report by generating sections in parallel.

        This method generates each section independently and concurrently, then merges
        them into a final report structure. Best for outlines with 3+ independent sections.
        """
        if len(outline) <= 2:
            return await self.write_report(question=question, outline=outline, context=context)

        evidences = _limit_evidences(
            context.get("evidences") or [], self.max_evidence_items, self.max_evidence_chars
        )
        graph_chain = list(context.get("graph_chain") or [])
        if self.max_graph_chain_items is not None:
            graph_chain = graph_chain[: max(self.max_graph_chain_items, 0)]

        semaphore = asyncio.Semaphore(self.max_parallel_sections)

        async def write_section_with_limit(section: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                return await self._write_single_section(
                    question=question,
                    section=section,
                    evidences=evidences,
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
        synthesis = await self._synthesize_parallel_fields(
            question=question,
            outline=outline,
            sections=sections,
            evidences=evidences,
            coverage=context.get("coverage") or {},
        )
        if synthesis:
            title = synthesis.get("title") or title
            summary = synthesis.get("summary") or ""
            limitations = synthesis.get("limitations") or []
            next_steps = synthesis.get("next_steps") or []
        else:
            summary = self._heuristic_summary(question=question, sections=sections)
            limitations = []
            next_steps = []
        if not isinstance(limitations, list):
            limitations = []
        if not isinstance(next_steps, list):
            next_steps = []
        summary_text = str(summary or "").strip() or self._heuristic_summary(question=question, sections=sections)
        return {
            "title": title,
            "summary": summary_text,
            "sections": sections,
            "limitations": [str(item) for item in limitations if str(item).strip()],
            "next_steps": [str(item) for item in next_steps if str(item).strip()],
            "citations": all_citations,
        }

    async def _synthesize_parallel_fields(
        self,
        *,
        question: str,
        outline: List[Dict[str, str]],
        sections: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
        coverage: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize title/summary/limitations/next_steps after parallel section drafting."""

        outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
        sections_json = json.dumps(self._compact_sections_for_synthesis(sections), ensure_ascii=False, indent=2)
        evidence_json = json.dumps(evidences, ensure_ascii=False, indent=2)
        coverage_json = json.dumps(coverage or {}, ensure_ascii=False, indent=2)

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
        raw = await self._call(messages, phase="parallel_synthesis")
        parsed = _safe_parse_json(raw, expected="dict")
        return parsed if isinstance(parsed, dict) else {}

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
        section: Dict[str, str],
        evidences: List[Dict[str, Any]],
        graph_chain: List[str],
    ) -> Dict[str, Any]:
        """Generate a single section using the section-specific prompt."""
        user_prompt = SECTION_WRITE_USER_PROMPT.format(
            question=question,
            section_title=section.get("title", ""),
            section_purpose=section.get("purpose", ""),
            evidence_json=json.dumps(evidences, ensure_ascii=False, indent=2),
            graph_chain_json=json.dumps(graph_chain, ensure_ascii=False, indent=2),
        )
        messages = [
            {"role": "system", "content": SECTION_WRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = await self._call(messages, phase=f"section:{section.get('title', 'unknown')}")
        data = _safe_parse_json(raw, expected="dict")
        if not data:
            return {"title": section.get("title", ""), "body_markdown": "", "citations": []}
        return {
            "title": data.get("title") or section.get("title", ""),
            "body_markdown": data.get("body_markdown") or "",
            "citations": data.get("citations") or [],
        }


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


__all__ = ["DeepSearchLLMReportWriter", "render_markdown_from_structured"]
