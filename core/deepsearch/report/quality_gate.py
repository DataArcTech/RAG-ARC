"""Quality gating + iteration signals for DeepSearch.

This module turns report post-processing artifacts (citations, consistency checks)
into structured decision signals that can drive a "research loop → quality gate → iterate"
workflow similar to Anthropic's multi-agent Research architecture.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from config.core.deepsearch import report_writer_defaults as report_defaults
from core.deepsearch.tools.base import call_llm_async
from core.prompts.deepsearch.report import JSON_REPAIR_USER_PROMPT
from core.prompts.deepsearch.quality_gate import QUALITY_GATE_SYSTEM_PROMPT, QUALITY_GATE_USER_PROMPT
from core.deepsearch.utils.language_policy import infer_user_language
from core.utils.json_extract import safe_json_loads
from core.utils.text_regex import BRACKET_CONTENT_RE, SENTENCE_SPLIT_RE


_BRACKET_RE = BRACKET_CONTENT_RE
_SENTENCE_SPLIT_RE = SENTENCE_SPLIT_RE


class QualityGateConfig(BaseModel):
    enabled: bool = Field(False, description="Enable the quality gate + iteration loop.")
    max_rounds: int = Field(2, ge=1, le=6, description="Maximum research iterations (including the first pass).")
    min_citation_sentence_coverage: float = Field(
        0.6, ge=0.0, le=1.0, description="Minimum fraction of report sentences that include a valid citation."
    )
    require_consistency: bool = Field(True, description="Fail when the consistency checker reports issues.")
    max_uncited_sentences: int = Field(6, ge=0, le=30, description="Maximum uncited sentences to surface for repair.")
    max_actions: int = Field(6, ge=0, le=20, description="Maximum follow-up actions to return.")
    enable_llm_judge: bool = Field(True, description="Use an LLM rubric judge to generate actions/scores.")
    judge_temperature: float = Field(0.0, ge=0.0, le=1.0)
    judge_max_retries: int = Field(1, ge=0, le=5)
    judge_max_evidence_items: int = Field(20, ge=1, le=80, description="Max evidence snippets forwarded to the LLM judge.")
    judge_max_evidence_chars: int = Field(900, ge=100, le=5000, description="Max characters per snippet for the LLM judge.")
    trigger_external_on_quality_failure: bool = Field(
        True, description="Allow the gate to request external search even if gap detection did not trigger it."
    )


class QualityGateAction(BaseModel):
    action: Literal["graph_search", "external_search", "rewrite"]
    query: Optional[str] = None
    rationale: str = Field(..., min_length=1)
    priority: int = Field(1, ge=1, le=5)


class QualityGateJudgeScores(BaseModel):
    factual_accuracy: float = Field(..., ge=0.0, le=1.0)
    citation_accuracy: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    source_quality: float = Field(..., ge=0.0, le=1.0)


class QualityGateJudgeResult(BaseModel):
    passed: bool = Field(..., alias="pass")
    overall: float = Field(..., ge=0.0, le=1.0)
    scores: QualityGateJudgeScores
    reasons: List[str] = Field(default_factory=list)
    missing_topics: List[str] = Field(default_factory=list)
    missing_claims: List[str] = Field(default_factory=list)
    next_actions: List[QualityGateAction] = Field(default_factory=list)


class QualityGateMetrics(BaseModel):
    citation_sentence_coverage: float = Field(..., ge=0.0, le=1.0)
    cited_sentence_count: int = Field(0, ge=0)
    sentence_count: int = Field(0, ge=0)
    uncited_sentences: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    evidence_source_counts: Dict[str, int] = Field(default_factory=dict)
    evidence_source_ratios: Dict[str, float] = Field(default_factory=dict)
    consistency_ok: Optional[bool] = None
    consistency_issue_count: int = Field(0, ge=0)
    missing_topics: List[str] = Field(default_factory=list)
    citation_mismatch_count: int = Field(0, ge=0)
    citation_mismatch_samples: List[Dict[str, Any]] = Field(default_factory=list)


class QualityGateResult(BaseModel):
    enabled: bool
    passed: bool
    should_iterate: bool
    metrics: QualityGateMetrics
    actions: List[QualityGateAction] = Field(default_factory=list)
    judge: Optional[QualityGateJudgeResult] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class DeepSearchQualityGate:
    """Turn report post-processing into actionable research loop signals."""

    def __init__(self, llm_connector: Any | None, *, config: Dict[str, Any] | QualityGateConfig | None = None) -> None:
        self.llm_connector = llm_connector
        self.config = self._coerce_config(config)

    async def evaluate(
        self,
        *,
        question: str,
        structured_report: Mapping[str, Any] | None,
        evidences: Sequence[Dict[str, Any]],
        gap_result: Optional[Dict[str, Any]] = None,
        external_allowed: bool = False,
    ) -> QualityGateResult:
        cfg = self.config
        if not cfg.enabled:
            return QualityGateResult(
                enabled=False,
                passed=True,
                should_iterate=False,
                metrics=QualityGateMetrics(
                    citation_sentence_coverage=1.0,
                    cited_sentence_count=0,
                    sentence_count=0,
                ),
                actions=[],
                diagnostics={"disabled": True},
            )

        sr = structured_report if isinstance(structured_report, Mapping) else {}
        evidence_lookup = {str(ev.get("chunk_id") or "").strip(): ev for ev in evidences if isinstance(ev, dict)}
        evidence_ids = set(evidence_lookup)
        evidence_ids.discard("")

        citations_text = self._collect_report_text(sr)
        cited_sentences, uncited_sentences, cited_ids, sentence_count = _citation_sentence_coverage(
            citations_text,
            known_ids=evidence_ids,
            max_uncited=cfg.max_uncited_sentences,
        )

        source_counts, source_ratios = _evidence_source_stats(evidences)
        missing_topics = _coerce_topics((gap_result or {}).get("missing_topics")) if gap_result else []

        consistency_ok, consistency_issue_count = _coerce_consistency(sr.get("consistency_check"))
        mismatch_count, mismatch_samples = _citation_integrity_checks(
            citations_text,
            evidence_lookup=evidence_lookup,
            known_ids=evidence_ids,
            max_samples=max(0, int(cfg.max_uncited_sentences)),
        )

        metrics = QualityGateMetrics(
            citation_sentence_coverage=round((cited_sentences / sentence_count), 4) if sentence_count else 1.0,
            cited_sentence_count=cited_sentences,
            sentence_count=sentence_count,
            uncited_sentences=uncited_sentences,
            cited_evidence_ids=sorted(cited_ids),
            evidence_source_counts=source_counts,
            evidence_source_ratios=source_ratios,
            consistency_ok=consistency_ok,
            consistency_issue_count=consistency_issue_count,
            missing_topics=missing_topics,
            citation_mismatch_count=mismatch_count,
            citation_mismatch_samples=mismatch_samples,
        )

        deterministic_pass = True
        reasons: List[str] = []
        if sentence_count and metrics.citation_sentence_coverage < cfg.min_citation_sentence_coverage:
            deterministic_pass = False
            reasons.append(
                f"citation_sentence_coverage={metrics.citation_sentence_coverage} < {cfg.min_citation_sentence_coverage}"
            )
        if cfg.require_consistency and consistency_ok is False:
            deterministic_pass = False
            reasons.append(f"consistency_check_failed (issues={consistency_issue_count})")
        if mismatch_count:
            deterministic_pass = False
            reasons.append(f"citation_integrity_failed (mismatches={mismatch_count})")

        generation = sr.get("generation") if isinstance(sr, Mapping) else None
        generation_mode = generation.get("mode") if isinstance(generation, Mapping) else None
        terminal_modes = {
            "deterministic_no_evidence",
            "deterministic_missing_named_files",
        }
        if isinstance(generation_mode, str) and generation_mode in terminal_modes:
            # These modes are explicit safe-stops (e.g., "no evidence" / "missing named files").
            # They should terminate the loop without requiring an LLM judge or follow-up actions.
            return QualityGateResult(
                enabled=True,
                passed=True,
                should_iterate=False,
                metrics=metrics,
                actions=[],
                judge=None,
                diagnostics={
                    "terminal_mode": generation_mode,
                    "deterministic_reasons": reasons,
                    "external_allowed": external_allowed,
                },
            )

        judge: QualityGateJudgeResult | None = None
        actions: List[QualityGateAction] = []
        if cfg.enable_llm_judge:
            if self.llm_connector is None:
                raise RuntimeError("Quality gate requires an LLM connector when enable_llm_judge is true.")
            try:
                judge = await self._maybe_judge_with_llm(
                    question=question,
                    sr=sr,
                    evidences=evidences,
                    metrics=metrics,
                    external_allowed=external_allowed,
                    cfg=cfg,
                )
                actions = list(judge.next_actions if judge else [])
            except Exception as exc:  # noqa: BLE001
                # Do not crash the run: fall back to deterministic signals and synthesized actions.
                should_iterate = bool((not deterministic_pass) and cfg.max_rounds > 1)
                return QualityGateResult(
                    enabled=True,
                    passed=bool(deterministic_pass),
                    should_iterate=should_iterate,
                    metrics=metrics,
                    actions=_synthesize_followups(
                        question=question,
                        metrics=metrics,
                        external_allowed=external_allowed,
                        cfg=cfg,
                    )
                    if should_iterate
                    else [],
                    judge=None,
                    diagnostics={
                        "deterministic_reasons": reasons,
                        "external_allowed": external_allowed,
                        "judge_error": str(exc) or exc.__class__.__name__,
                    },
                )
        else:
            if not deterministic_pass and cfg.max_rounds > 1:
                actions = _synthesize_followups(
                    question=question,
                    metrics=metrics,
                    external_allowed=external_allowed,
                    cfg=cfg,
                )

        passed = bool(deterministic_pass and (judge.passed if judge else True))
        should_iterate = (not passed) and cfg.max_rounds > 1
        if not cfg.trigger_external_on_quality_failure:
            actions = [a for a in actions if a.action != "external_search"]
        if not external_allowed:
            actions = [a for a in actions if a.action != "external_search"]
        normalized: List[QualityGateAction] = []
        seen: set[tuple[str, str]] = set()
        for action in actions:
            key = (action.action, (action.query or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(action)
            if len(normalized) >= cfg.max_actions:
                break
        actions = normalized
        if should_iterate and not actions:
            actions = _synthesize_followups(
                question=question,
                metrics=metrics,
                external_allowed=external_allowed,
                cfg=cfg,
            )
        if should_iterate and actions:
            has_graph = any(a.action == "graph_search" for a in actions)
            if not has_graph:
                extra = _synthesize_followups(
                    question=question,
                    metrics=metrics,
                    external_allowed=external_allowed,
                    cfg=cfg,
                )
                for action in extra:
                    if action.action != "graph_search" or not action.query:
                        continue
                    key = (action.action, (action.query or "").strip().lower())
                    if key in seen:
                        continue
                    actions.append(action)
                    seen.add(key)
                    if len(actions) >= cfg.max_actions:
                        break

        return QualityGateResult(
            enabled=True,
            passed=passed,
            should_iterate=should_iterate,
            metrics=metrics,
            actions=actions,
            judge=judge,
            diagnostics={"deterministic_reasons": reasons, "external_allowed": external_allowed},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_config(config: Dict[str, Any] | QualityGateConfig | None) -> QualityGateConfig:
        if config is None:
            raise ValueError("Quality gate requires an explicit config (no implicit defaults).")
        if isinstance(config, QualityGateConfig):
            return config
        if isinstance(config, dict):
            try:
                return QualityGateConfig.model_validate(config)
            except ValidationError as exc:
                raise ValueError(f"Invalid quality gate config: {exc}") from exc
        raise TypeError("Unsupported quality gate config type")

    @staticmethod
    def _collect_report_text(sr: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """Return (location, text) pairs for citation coverage analysis."""
        items: List[Tuple[str, str]] = []
        summary = sr.get("summary")
        if isinstance(summary, str) and summary.strip():
            items.append(("summary", summary.strip()))
        sections = sr.get("sections")
        if isinstance(sections, list):
            for idx, section in enumerate(sections, start=1):
                if not isinstance(section, dict):
                    continue
                title = str(section.get("title") or "").strip()
                body = str(section.get("body_markdown") or "").strip()
                if not body:
                    continue
                label = f"section:{idx}" if not title else f"section:{idx}:{title}"
                items.append((label, body))
        return items

    async def _maybe_judge_with_llm(
        self,
        *,
        question: str,
        sr: Mapping[str, Any],
        evidences: Sequence[Dict[str, Any]],
        metrics: QualityGateMetrics,
        external_allowed: bool,
        cfg: QualityGateConfig,
    ) -> QualityGateJudgeResult:
        if not cfg.enable_llm_judge or self.llm_connector is None:
            raise RuntimeError("LLM judge is required when quality gate is enabled.")

        summary = str(sr.get("summary") or "").strip()
        sections = sr.get("sections") if isinstance(sr.get("sections"), list) else []
        section_blocks: List[str] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip() or "Section"
            body = str(section.get("body_markdown") or "").strip()
            if body:
                section_blocks.append(f"## {title}\n{body}")
        sections_markdown = "\n\n".join(section_blocks)
        # Keep judge prompt bounded: the judge mostly needs coverage signals and a compact report snapshot.
        if len(sections_markdown) > 8000:
            sections_markdown = sections_markdown[:7999].rstrip() + "…"
        if len(summary) > 1200:
            summary = summary[:1199].rstrip() + "…"

        signals = {
            "citation_sentence_coverage": metrics.citation_sentence_coverage,
            "uncited_sentences": metrics.uncited_sentences,
            "citation_mismatch_count": metrics.citation_mismatch_count,
            "citation_mismatch_samples": metrics.citation_mismatch_samples,
            "consistency_ok": metrics.consistency_ok,
            "consistency_issue_count": metrics.consistency_issue_count,
            "missing_topics": metrics.missing_topics,
            "evidence_source_ratios": metrics.evidence_source_ratios,
        }
        user_prompt = QUALITY_GATE_USER_PROMPT.format(
            question=question,
            summary=summary,
            sections_markdown=sections_markdown,
            signals_json=json.dumps(signals, ensure_ascii=False, indent=2),
            evidence_json=json.dumps(
                _limit_evidences(evidences, max_items=cfg.judge_max_evidence_items, max_chars=cfg.judge_max_evidence_chars),
                ensure_ascii=False,
                indent=2,
            ),
            external_allowed=str(bool(external_allowed)).lower(),
        )
        output_language = infer_user_language(question)
        messages = [
            {"role": "system", "content": QUALITY_GATE_SYSTEM_PROMPT.replace("{output_language}", output_language)},
            {"role": "user", "content": user_prompt},
        ]

        last_exc: Exception | None = None
        repair_messages: List[Dict[str, str]] | None = None
        retries = max(cfg.judge_max_retries, 1)
        for attempt in range(retries):
            try:
                raw = await call_llm_async(
                    self.llm_connector,
                    (repair_messages or messages),
                    temperature=cfg.judge_temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

            parsed = safe_json_loads(raw, expected="dict")
            if parsed is None:
                last_exc = ValueError("LLM judge did not return valid JSON.")
                if attempt < retries - 1:
                    repair_prompt = JSON_REPAIR_USER_PROMPT.format(
                        expected_top_level="object",
                        error="invalid_json",
                        raw_snippet=_snippet(raw, limit=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS)),
                    )
                    repair_messages = messages + [
                        {"role": "assistant", "content": str(raw or "")},
                        {"role": "user", "content": repair_prompt},
                    ]
                    continue
                raise ValueError("LLM judge did not return valid JSON.") from None
            try:
                return QualityGateJudgeResult.model_validate(parsed)
            except ValidationError as exc:
                last_exc = exc
                if attempt < retries - 1:
                    repair_prompt = JSON_REPAIR_USER_PROMPT.format(
                        expected_top_level="object",
                        error="schema_validation_error",
                        raw_snippet=_snippet(raw, limit=int(report_defaults.DEFAULT_ERROR_SNIPPET_LIMIT_CHARS)),
                    )
                    repair_messages = messages + [
                        {"role": "assistant", "content": str(raw or "")},
                        {"role": "user", "content": repair_prompt},
                    ]
                    continue
                raise ValueError("LLM judge returned JSON with an unexpected schema.") from None

        raise RuntimeError(f"LLM judge failed to run: {last_exc}") from last_exc


def _snippet(text: str, *, limit: int = 500) -> str:
    value = str(text or "").strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _coerce_topics(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _coerce_consistency(consistency_payload: Any) -> tuple[Optional[bool], int]:
    if not isinstance(consistency_payload, dict):
        return None, 0
    ok = consistency_payload.get("is_consistent")
    issues = consistency_payload.get("issues")
    issue_count = len(issues) if isinstance(issues, list) else 0
    if isinstance(ok, bool):
        return ok, issue_count
    return None, issue_count


def _citation_sentence_coverage(
    texts: Sequence[Tuple[str, str]],
    *,
    known_ids: set[str],
    max_uncited: int,
) -> tuple[int, List[str], set[str], int]:
    cited_sentence_count = 0
    sentence_count = 0
    uncited: List[str] = []
    cited_ids: set[str] = set()

    for _, text in texts:
        for sentence in _iter_sentences(text):
            sentence_count += 1
            found, ids = _sentence_has_known_citation(sentence, known_ids=known_ids)
            if found:
                cited_sentence_count += 1
                cited_ids.update(ids)
                continue
            if max_uncited <= 0:
                continue
            if len(uncited) < max_uncited:
                uncited.append(sentence[:240].strip())

    return cited_sentence_count, uncited, cited_ids, sentence_count


def _iter_sentences(text: str) -> Iterable[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []
    # Split by lines first to respect bullets/markdown formatting.
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    sentences: List[str] = []
    for idx, line in enumerate(lines):
        # Skip markdown table separators and headers (these are structural, not claims).
        if re.fullmatch(r"[\s\|:\-]+", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            # A header row is typically followed by a separator row like: | --- | --- |
            if idx + 1 < len(lines) and re.fullmatch(r"[\s\|:\-]+", lines[idx + 1]):
                continue
        # Remove headings/list markers but keep content.
        line = re.sub(r"^#{1,6}\s+", "", line).strip()
        line = re.sub(r"^[-*]\s+", "", line).strip()
        if not line:
            continue
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(line) if part.strip()]
        if not parts:
            continue

        # Merge standalone citation fragments back into the previous sentence so patterns like
        # "Claim sentence. [ev1]" count as cited rather than dropping the short citation fragment.
        merged: List[str] = []
        for part in parts:
            if merged and _is_citation_only(part):
                merged[-1] = f"{merged[-1].rstrip()} {part}".strip()
            else:
                merged.append(part)

        for token in merged:
            # Drop tiny fragments (often connective text) unless they carry a real claim.
            claim_len = len(re.sub(_BRACKET_RE, "", token).strip())
            if claim_len < 20:
                continue
            sentences.append(token)
    return sentences


def _is_citation_only(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    # Allow one or more bracketed citation groups, possibly separated by whitespace.
    return bool(re.fullmatch(r"(?:\[[^\[\]]+\]\s*)+", stripped))


def _sentence_has_known_citation(sentence: str, *, known_ids: set[str]) -> tuple[bool, set[str]]:
    ids: set[str] = set()
    for raw in _BRACKET_RE.findall(sentence or ""):
        for candidate in re.split(r"[,\s]+", raw.strip()):
            token = candidate.strip()
            if token in known_ids:
                ids.add(token)
    return (len(ids) > 0), ids


def _evidence_source_stats(evidences: Sequence[Dict[str, Any]]) -> tuple[Dict[str, int], Dict[str, float]]:
    counts: Dict[str, int] = {}
    total = 0
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        source = str(ev.get("source") or "").strip() or "unknown"
        counts[source] = counts.get(source, 0) + 1
        total += 1
    ratios: Dict[str, float] = {}
    if total > 0:
        for key, value in counts.items():
            ratios[key] = round(value / total, 4)
    return counts, ratios


def _limit_evidences(evidences: Sequence[Dict[str, Any]], *, max_items: int, max_chars: int) -> List[Dict[str, Any]]:
    subset = list(evidences)[: max(0, max_items)]
    limited: List[Dict[str, Any]] = []
    for entry in subset:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content") or "")
        if max_chars > 0 and len(content) > max_chars:
            content = content[: max_chars].rstrip() + "…"
        prov = entry.get("provenance")
        filename = None
        if isinstance(prov, dict):
            meta = prov.get("metadata")
            if isinstance(meta, dict):
                filename = meta.get("filename") or meta.get("source_file_name") or meta.get("path") or meta.get("file_path")
                if not filename:
                    chunk_meta = meta.get("chunk_metadata")
                    if isinstance(chunk_meta, dict):
                        filename = (
                            chunk_meta.get("filename")
                            or chunk_meta.get("source_file_name")
                            or chunk_meta.get("path")
                            or chunk_meta.get("file_path")
                        )
            if not filename:
                filename = prov.get("path") or prov.get("file_path") or prov.get("source_path")
        limited.append(
            {
                "chunk_id": entry.get("chunk_id"),
                "source": entry.get("source"),
                "score": entry.get("score"),
                "filename": filename,
                "content": content,
            }
        )
    return limited


def _extract_anchors(text: str) -> List[str]:
    """Extract lightweight lexical anchors for citation integrity checks.

    Keep this language-agnostic and conservative: favor numeric anchors.
    (Free-form word tokens are often too generic and cause false positives.)
    """

    q = str(text or "").strip()
    if not q:
        return []
    anchors: list[str] = []
    for token in re.findall(r"\d[\d,\.%]*", q):
        token = token.strip()
        # Skip ultra-weak anchors like bare "1"/"2" that often appear in lists/years and create false mismatches.
        if len(token) < 2 and token.isdigit():
            continue
        if token and token not in anchors:
            anchors.append(token)
    # Mixed alnum tokens containing digits (e.g. "v2", "x86", "2023Q4").
    for token in re.findall(r"[^\W_]*\d[^\W_]*", q, flags=re.UNICODE):
        token = token.strip()
        if len(token) < 3:
            continue
        if token and token not in anchors:
            anchors.append(token)
    return anchors[:12]


def _citation_integrity_checks(
    texts: Sequence[Tuple[str, str]],
    *,
    evidence_lookup: Mapping[str, Dict[str, Any]],
    known_ids: set[str],
    max_samples: int,
) -> tuple[int, List[Dict[str, Any]]]:
    """Return (mismatch_count, samples[]) for sentences whose citations don't match evidence anchors."""

    mismatches = 0
    samples: List[Dict[str, Any]] = []
    for location, text in texts:
        for sentence in _iter_sentences(text):
            found, ids = _sentence_has_known_citation(sentence, known_ids=known_ids)
            if not found or not ids:
                continue
            # Anchors should come from the claim text, not from the citation tokens themselves.
            claim_text = re.sub(_BRACKET_RE, "", sentence or "").strip()
            anchors = _extract_anchors(claim_text)
            if not anchors:
                continue
            for ev_id in ids:
                ev = evidence_lookup.get(ev_id) or {}
                content = str(ev.get("content") or "")
                if not content.strip():
                    continue
                content_lower = content.lower()
                ok = any(anchor.lower() in content_lower for anchor in anchors if anchor)
                if ok:
                    continue
                mismatches += 1
                if max_samples > 0 and len(samples) < max_samples:
                    samples.append(
                        {
                            "location": location,
                            "sentence": sentence[:280],
                            "evidence_id": ev_id,
                            "anchors": anchors[:6],
                        }
                    )
    return mismatches, samples


def _synthesize_followups(
    *,
    question: str,
    metrics: QualityGateMetrics,
    external_allowed: bool,
    cfg: QualityGateConfig,
) -> List[QualityGateAction]:
    """Fallback follow-ups that guarantee the loop executes more retrieval before rewriting."""

    actions: List[QualityGateAction] = []
    max_actions = max(0, int(cfg.max_actions))

    def _append(action: QualityGateAction) -> None:
        if len(actions) >= max_actions:
            return
        actions.append(action)

    priority = 1
    for sample in metrics.citation_mismatch_samples:
        if len(actions) >= max_actions:
            break
        if not isinstance(sample, dict):
            continue
        anchors = sample.get("anchors") if isinstance(sample.get("anchors"), list) else []
        query = " ".join([str(a).strip() for a in anchors if str(a).strip()][:6]).strip()
        if not query:
            continue
        _append(
            QualityGateAction(
                action="graph_search",
                query=query[:220],
                rationale="Citations appear mismatched; retrieve supporting evidence for the anchored claim terms.",
                priority=priority,
            )
        )
        priority = min(5, priority + 1)

    for topic in metrics.missing_topics:
        if len(actions) >= max_actions:
            break
        token = str(topic).strip()
        if not token:
            continue
        _append(
            QualityGateAction(
                action="graph_search",
                query=token[:220],
                rationale="Key topic is missing; gather more internal evidence before rewriting.",
                priority=priority,
            )
        )
        priority = min(5, priority + 1)

    for sentence in metrics.uncited_sentences:
        if len(actions) >= max_actions:
            break
        seed = str(sentence).strip()
        if not seed:
            continue
        seed = re.sub(_BRACKET_RE, "", seed).strip()
        if not seed:
            continue
        _append(
            QualityGateAction(
                action="graph_search",
                query=seed[:220],
                rationale="The report contains uncited claims; retrieve evidence to cite or delete the claim.",
                priority=priority,
            )
        )
        priority = min(5, priority + 1)

    if external_allowed and len(actions) < max_actions:
        _append(
            QualityGateAction(
                action="external_search",
                query=question[:220],
                rationale="Internal sources appear insufficient; broaden search externally.",
                priority=min(5, priority),
            )
        )

    if len(actions) < max_actions:
        _append(
            QualityGateAction(
                action="rewrite",
                query=None,
                rationale="Rewrite the report to include only evidence-backed claims (delete unsupported content).",
                priority=5,
            )
        )

    return actions
