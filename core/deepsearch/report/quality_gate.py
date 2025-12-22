"""Quality gating + iteration signals for DeepSearch.

This module turns report post-processing artifacts (citations, consistency checks)
into structured decision signals that can drive a "research loop → quality gate → iterate"
workflow similar to Anthropic's multi-agent Research architecture.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from core.deepsearch.tools.base import call_llm_async
from core.prompts.deepsearch.quality_gate import QUALITY_GATE_SYSTEM_PROMPT, QUALITY_GATE_USER_PROMPT


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?\u3002\uff01\uff1f])\s+")


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
        evidence_ids = {str(ev.get("chunk_id") or "").strip() for ev in evidences if isinstance(ev, dict)}
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

        should_call_judge = bool(
            cfg.enable_llm_judge
            and self.llm_connector is not None
            and (not deterministic_pass or bool(missing_topics))
        )
        judge = None
        if should_call_judge:
            judge = await self._maybe_judge_with_llm(
                question=question,
                sr=sr,
                evidences=evidences,
                metrics=metrics,
                external_allowed=external_allowed,
                cfg=cfg,
            )

        passed = deterministic_pass and (judge.passed if judge else True)
        should_iterate = (not passed) and cfg.max_rounds > 1

        actions = self._synthesize_actions(
            question=question,
            metrics=metrics,
            judge=judge,
            external_allowed=external_allowed,
            cfg=cfg,
        )
        if not cfg.trigger_external_on_quality_failure:
            actions = [a for a in actions if a.action != "external_search"]

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
            return QualityGateConfig()
        if isinstance(config, QualityGateConfig):
            return config
        if isinstance(config, dict):
            try:
                return QualityGateConfig.model_validate(config)
            except ValidationError:
                return QualityGateConfig()
        return QualityGateConfig()

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
    ) -> Optional[QualityGateJudgeResult]:
        if not cfg.enable_llm_judge or self.llm_connector is None:
            return None

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

        signals = {
            "citation_sentence_coverage": metrics.citation_sentence_coverage,
            "uncited_sentences": metrics.uncited_sentences,
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
            evidence_json=json.dumps(_limit_evidences(evidences, max_items=20, max_chars=900), ensure_ascii=False, indent=2),
            external_allowed=str(bool(external_allowed)).lower(),
        )
        messages = [
            {"role": "system", "content": QUALITY_GATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_exc: Exception | None = None
        for _ in range(max(cfg.judge_max_retries, 1)):
            try:
                raw = await call_llm_async(self.llm_connector, messages, temperature=cfg.judge_temperature)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
            parsed = _safe_parse_json(raw)
            if parsed is None:
                return QualityGateJudgeResult(
                    **{
                        "pass": False,
                        "overall": 0.0,
                        "scores": {
                            "factual_accuracy": 0.0,
                            "citation_accuracy": 0.0,
                            "completeness": 0.0,
                            "source_quality": 0.0,
                        },
                        "reasons": ["LLM judge did not return valid JSON."],
                        "missing_topics": [],
                        "missing_claims": [],
                        "next_actions": [],
                    }
                )
            try:
                return QualityGateJudgeResult.model_validate(parsed)
            except ValidationError:
                return QualityGateJudgeResult(
                    **{
                        "pass": False,
                        "overall": 0.0,
                        "scores": {
                            "factual_accuracy": 0.0,
                            "citation_accuracy": 0.0,
                            "completeness": 0.0,
                            "source_quality": 0.0,
                        },
                        "reasons": ["LLM judge returned JSON with an unexpected schema."],
                        "missing_topics": [],
                        "missing_claims": [],
                        "next_actions": [],
                    }
                )

        if last_exc is not None:
            return QualityGateJudgeResult(
                **{
                    "pass": False,
                    "overall": 0.0,
                    "scores": {
                        "factual_accuracy": 0.0,
                        "citation_accuracy": 0.0,
                        "completeness": 0.0,
                        "source_quality": 0.0,
                    },
                    "reasons": [f"LLM judge failed to run: {last_exc}"],
                    "missing_topics": [],
                    "missing_claims": [],
                    "next_actions": [],
                }
            )
        return None

    def _synthesize_actions(
        self,
        *,
        question: str,
        metrics: QualityGateMetrics,
        judge: Optional[QualityGateJudgeResult],
        external_allowed: bool,
        cfg: QualityGateConfig,
    ) -> List[QualityGateAction]:
        actions: List[QualityGateAction] = []

        if judge and judge.next_actions:
            actions.extend(judge.next_actions)

        missing_topics = list(metrics.missing_topics)
        if judge and judge.missing_topics:
            missing_topics = list({*missing_topics, *judge.missing_topics})

        missing_claims: List[str] = []
        if judge and judge.missing_claims:
            missing_claims.extend([c for c in judge.missing_claims if isinstance(c, str) and c.strip()])
        if metrics.uncited_sentences:
            missing_claims.extend(metrics.uncited_sentences)

        if missing_topics:
            for topic in missing_topics[: max(cfg.max_actions, 1)]:
                query = f"{question} {topic}".strip()
                actions.append(
                    QualityGateAction(
                        action="graph_search",
                        query=query,
                        rationale=f"Fill missing topic: {topic}",
                        priority=1,
                    )
                )
                if external_allowed:
                    actions.append(
                        QualityGateAction(
                            action="external_search",
                            query=query,
                            rationale=f"Collect external sources for missing topic: {topic}",
                            priority=2,
                        )
                    )

        if missing_claims and metrics.citation_sentence_coverage < cfg.min_citation_sentence_coverage:
            for claim in missing_claims[: max(cfg.max_actions, 1)]:
                query = _normalize_query(claim)
                if not query:
                    continue
                actions.append(
                    QualityGateAction(
                        action="graph_search",
                        query=query,
                        rationale="Find evidence for an uncited claim and rewrite the report with citations.",
                        priority=2,
                    )
                )
                if external_allowed:
                    actions.append(
                        QualityGateAction(
                            action="external_search",
                            query=query,
                            rationale="Find authoritative external sources for an uncited claim.",
                            priority=3,
                        )
                    )

        # Ensure we always request a rewrite if we are missing citations or consistency failed.
        if (metrics.consistency_ok is False) or (metrics.sentence_count and metrics.uncited_sentences):
            actions.append(
                QualityGateAction(
                    action="rewrite",
                    query=None,
                    rationale="Rewrite the report to ensure all factual claims have valid inline citations and remove unsupported claims.",
                    priority=1,
                )
            )

        # Deduplicate + cap
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
        return normalized


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
    for line in lines:
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


def _normalize_query(text: str) -> str:
    token = (text or "").strip()
    token = re.sub(r"\[[^\[\]]+\]", "", token).strip()
    token = re.sub(r"\s+", " ", token).strip()
    if len(token) < 12:
        return ""
    # Cap length to keep tool queries sane.
    return token[:180].rstrip()


def _safe_parse_json(raw: str) -> Dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    extracted = _extract_first_json(text)
    if extracted is None:
        return None
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_first_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
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
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _limit_evidences(evidences: Sequence[Dict[str, Any]], *, max_items: int, max_chars: int) -> List[Dict[str, Any]]:
    subset = list(evidences)[: max(0, max_items)]
    limited: List[Dict[str, Any]] = []
    for entry in subset:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content") or "")
        if max_chars > 0 and len(content) > max_chars:
            content = content[: max_chars].rstrip() + "…"
        cloned = dict(entry)
        cloned["content"] = content
        limited.append(cloned)
    return limited
