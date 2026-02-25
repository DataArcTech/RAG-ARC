"""logic.check tool: review DeepSearch reasoning DAG for logic errors."""
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_TOOL_OUTPUT
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.prompts.deepsearch.tools import LOGIC_CHECK_SYSTEM_PROMPT_EN
from core.utils.llm_json import repair_json_from_raw_with_retry

from ..base import (
    GraphTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
    build_input_schema,
    call_llm_async,
    safe_json_loads,
)
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER
from ..explore.graph_ops.templates_utils import build_derived_evidence


_ISSUE_TYPES = {
    "evidence_gap",
    "conflict",
    "plan_gap",
    "computable_without_code",
    "reasoning_jump",
    "other",
}

_SEVERITIES = {"low", "medium", "high"}


class LogicAssertion(BaseModel):
    key: str = Field(..., min_length=1)
    value: Optional[str] = None
    polarity: str = Field("affirm")
    evidence_ids: List[str] = Field(default_factory=list)
    branch: Optional[str] = None

    @field_validator("polarity", mode="before")
    @classmethod
    def _normalize_polarity(cls, value: Any) -> str:
        token = str(value or "affirm").strip().lower()
        if token in {"deny", "false", "no"}:
            return "deny"
        return "affirm"

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class LogicCheckIssue(BaseModel):
    issue_type: str = Field("other")
    severity: str = Field("medium")
    message: str = Field(..., min_length=1)
    evidence_ids: List[str] = Field(default_factory=list)
    related_steps: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None

    @field_validator("issue_type", mode="before")
    @classmethod
    def _normalize_issue_type(cls, value: Any) -> str:
        token = str(value or "").strip().lower() or "other"
        return token if token in _ISSUE_TYPES else "other"

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, value: Any) -> str:
        token = str(value or "").strip().lower() or "medium"
        return token if token in _SEVERITIES else "medium"

    @field_validator("evidence_ids", "related_steps", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class LogicCheckResponse(BaseModel):
    summary: str = Field(..., min_length=1)
    ok: bool = Field(True)
    issues: List[LogicCheckIssue] = Field(default_factory=list)


class LogicCheckTool(GraphTool):
    """LLM-assisted review of the reasoning DAG with guardrail checks."""

    descriptor = ToolDescriptor(
        name="logic.check",
        channel="graph",
        description=(
            "Review the reasoning chain for logical errors: evidence gaps, contradictions, "
            "uncovered plan items, and missing deterministic verifications. "
            "Returns a list of issues found. Does NOT gather new evidence or evaluate report quality."
        ),
        speed="slow",
        cost="low",
        strategy_tags=("logic_check", "reflection", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_LLM),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.logic_check",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "assertions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Claim identifier (e.g. 'revenue_2024')."},
                            "value": {"type": "string", "description": "The claimed value or statement."},
                            "polarity": {"type": "string", "description": "Expected truth value: 'affirm' or 'deny'."},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "IDs of evidence chunks supporting this assertion.",
                            },
                            "branch": {"type": "string", "description": "Reasoning branch this assertion belongs to."},
                        },
                        "required": ["key"],
                    },
                    "description": "Structured claims to verify against collected evidence. Each assertion is checked for support and consistency.",
                },
                "max_issues": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Maximum number of issues to return. Defaults to 10.",
                },
            }
        ),
        example_args={
            "question": "Is Company A indirectly controlling Company C?",
            "plan_step": "plan_03",
            "extra": {
                "assertions": [
                    {"key": "control_path", "value": "A->B->C", "polarity": "affirm", "evidence_ids": ["graph.ops:plan_02:path_exists:abc123"]}
                ],
                "max_issues": 5,
            },
        },
    )

    def __init__(
        self,
        llm_connector=None,
        *,
        temperature: float = tool_defaults.LOGIC_CHECK_DEFAULT_TEMPERATURE,
        system_prompt: str = LOGIC_CHECK_SYSTEM_PROMPT_EN,
        json_repair_attempts: int = tool_defaults.LOGIC_CHECK_JSON_REPAIR_DEFAULT_ATTEMPTS,
        json_repair_temperature: float = tool_defaults.LOGIC_CHECK_JSON_REPAIR_DEFAULT_TEMPERATURE,
        json_repair_max_raw_chars: int = tool_defaults.LOGIC_CHECK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS,
    ) -> None:
        self.llm_connector = llm_connector
        self.temperature = float(temperature)
        self.system_prompt = system_prompt
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.json_repair_temperature = float(json_repair_temperature)
        max_raw_chars = int(json_repair_max_raw_chars)
        self.json_repair_max_raw_chars = (
            max_raw_chars if max_raw_chars > 0 else tool_defaults.LOGIC_CHECK_JSON_REPAIR_DEFAULT_MAX_RAW_CHARS
        )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        if self.llm_connector is None:
            raise RuntimeError("logic.check requires an LLM connector")

        extra = request.extra or {}
        assertions = self._parse_assertions(extra.get("assertions"))
        snapshot = self._coerce_snapshot(extra.get("runtime_snapshot"))
        evidence_ids = self._collect_evidence_ids(
            snapshot=snapshot,
            context_evidences=request.context_evidences,
            limit=tool_defaults.LOGIC_CHECK_EVIDENCE_ID_MAX,
        )
        tool_names = self._collect_tool_names(snapshot)
        plan_items = self._coerce_plan_items(snapshot.get("plan"))
        classification = self._extract_classification(request)
        deterministic_issues = self._deterministic_checks(
            assertions=assertions,
            evidence_ids=set(evidence_ids),
            tool_names=tool_names,
            plan_items=plan_items,
            classification=classification,
        )

        payload = self._build_prompt_payload(
            request=request,
            snapshot=snapshot,
            assertions=assertions,
            evidence_ids=evidence_ids,
            plan_items=plan_items,
            deterministic_issues=deterministic_issues,
            classification=classification,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._serialize_payload(payload)},
        ]
        raw = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
        parsed = await self._parse_or_repair_json(messages=messages, raw=raw)

        try:
            response = LogicCheckResponse.model_validate(parsed)
        except ValidationError as exc:
            raise RuntimeError(f"logic.check invalid response: {exc}") from exc

        max_issues = self._resolve_max_issues(extra.get("max_issues"))
        issues = self._merge_issues(response.issues, deterministic_issues, max_issues=max_issues)
        ok = bool(response.ok) and not deterministic_issues
        summary = response.summary.strip()
        if not summary:
            summary = "Logic check completed."
        if deterministic_issues and ok:
            ok = False
            summary = summary.rstrip() + " Deterministic guardrail issues detected."

        evidence = build_derived_evidence(
            tool_name=self.descriptor.name,
            plan_step=request.plan_step,
            label="logic_check",
            content=summary,
            provenance={
                "issue_count": len(issues),
                "issues": [issue.model_dump(exclude_none=True) for issue in issues],
                "tool_names": sorted(tool_names),
                "evidence_ids": evidence_ids,
                "classification": classification,
                "deterministic_issue_count": len(deterministic_issues),
            },
            kind=EVIDENCE_KIND_DERIVED,
            evidence_class=EVIDENCE_CLASS_TOOL_OUTPUT,
        )

        diagnostics = {
            "ok": ok,
            "issue_count": len(issues),
            "issues": [issue.model_dump(exclude_none=True) for issue in issues],
            "deterministic_issues": [issue.model_dump(exclude_none=True) for issue in deterministic_issues],
            "tool_names": sorted(tool_names),
            "evidence_ids": evidence_ids,
        }
        return ToolResult(summary=summary, evidences=[evidence], diagnostics=diagnostics)

    def _parse_assertions(self, raw: Any) -> List[LogicAssertion]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        assertions: List[LogicAssertion] = []
        for item in raw:
            try:
                assertions.append(LogicAssertion.model_validate(item))
            except ValidationError:
                continue
        return assertions[: tool_defaults.LOGIC_CHECK_MAX_ASSERTIONS]

    @staticmethod
    def _coerce_snapshot(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    @staticmethod
    def _collect_tool_names(snapshot: Dict[str, Any]) -> set[str]:
        raw = snapshot.get("tool_names")
        names: set[str] = set()
        if isinstance(raw, list):
            for item in raw:
                token = str(item or "").strip()
                if token:
                    names.add(token)
        return names

    @staticmethod
    def _coerce_plan_items(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        items: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                items.append(dict(item))
        return items

    @staticmethod
    def _extract_classification(request: ToolRunRequest) -> Dict[str, Any]:
        context = request.graph_context
        metadata = context.metadata if context is not None else {}
        if isinstance(metadata, dict):
            payload = metadata.get("question_classification")
            if isinstance(payload, dict):
                return dict(payload)
        return {}

    @staticmethod
    def _collect_evidence_ids(
        *,
        snapshot: Dict[str, Any],
        context_evidences: Sequence[Any],
        limit: int,
    ) -> List[str]:
        ids: List[str] = []
        seen: set[str] = set()
        def _add(value: Any) -> None:
            token = str(value or "").strip()
            if token and token not in seen:
                seen.add(token)
                ids.append(token)

        for item in context_evidences or []:
            chunk_id = getattr(item, "chunk_id", None) if not isinstance(item, dict) else item.get("chunk_id")
            _add(chunk_id)
            if len(ids) >= limit:
                return ids

        raw = snapshot.get("evidence_ids")
        if isinstance(raw, list):
            for item in raw:
                _add(item)
                if len(ids) >= limit:
                    return ids
        return ids

    def _deterministic_checks(
        self,
        *,
        assertions: Sequence[LogicAssertion],
        evidence_ids: set[str],
        tool_names: set[str],
        plan_items: Sequence[Dict[str, Any]],
        classification: Dict[str, Any],
    ) -> List[LogicCheckIssue]:
        issues: List[LogicCheckIssue] = []
        if assertions:
            missing_ids = self._missing_evidence_ids(assertions, evidence_ids)
            if missing_ids:
                issues.append(
                    LogicCheckIssue(
                        issue_type="evidence_gap",
                        severity="high",
                        message="Assertions reference evidence_ids not present in the runtime snapshot.",
                        evidence_ids=sorted(missing_ids),
                    )
                )
            conflicts = self._detect_conflicts(assertions)
            for key, details in conflicts:
                issues.append(
                    LogicCheckIssue(
                        issue_type="conflict",
                        severity="high",
                        message=f"Conflicting assertions detected for key '{key}'.",
                        related_steps=details,
                    )
                )

        if classification.get("is_computable") and "code.python" not in tool_names:
            issues.append(
                LogicCheckIssue(
                    issue_type="computable_without_code",
                    severity="medium",
                    message="Question classified as computable but no code.python run was observed.",
                    related_steps=sorted(tool_names),
                    suggested_fix="Use code.python to compute numeric/time results.",
                )
            )

        if plan_items and not tool_names:
            issues.append(
                LogicCheckIssue(
                    issue_type="plan_gap",
                    severity="medium",
                    message="Plan items exist but no tools were executed to support them.",
                )
            )
        return issues

    @staticmethod
    def _missing_evidence_ids(
        assertions: Sequence[LogicAssertion],
        evidence_ids: set[str],
    ) -> set[str]:
        missing: set[str] = set()
        for assertion in assertions:
            for evidence_id in assertion.evidence_ids:
                if evidence_id and evidence_id not in evidence_ids:
                    missing.add(evidence_id)
        return missing

    @staticmethod
    def _detect_conflicts(assertions: Sequence[LogicAssertion]) -> List[Tuple[str, List[str]]]:
        grouped: Dict[str, set[Tuple[str, str]]] = {}
        branches: Dict[str, set[str]] = {}
        for assertion in assertions:
            key = assertion.key.strip()
            if not key:
                continue
            value = str(assertion.value or "").strip()
            polarity = assertion.polarity
            grouped.setdefault(key, set()).add((polarity, value))
            if assertion.branch:
                branches.setdefault(key, set()).add(str(assertion.branch))

        conflicts: List[Tuple[str, List[str]]] = []
        for key, variants in grouped.items():
            if len(variants) <= 1:
                continue
            steps = sorted(branches.get(key, set()))
            conflicts.append((key, steps))
        return conflicts

    def _build_prompt_payload(
        self,
        *,
        request: ToolRunRequest,
        snapshot: Dict[str, Any],
        assertions: Sequence[LogicAssertion],
        evidence_ids: Sequence[str],
        plan_items: Sequence[Dict[str, Any]],
        deterministic_issues: Sequence[LogicCheckIssue],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        context_evidences = []
        for item in request.context_evidences or []:
            if isinstance(item, dict):
                context_evidences.append(item)
                continue
            context_evidences.append(
                {
                    "chunk_id": getattr(item, "chunk_id", None),
                    "source": getattr(item, "source", None),
                    "kind": getattr(item, "kind", None),
                    "content": getattr(item, "content", None),
                    "provenance": getattr(item, "provenance", None),
                }
            )
        return {
            "question": request.question,
            "plan_step": request.plan_step,
            "classification": classification,
            "plan": list(plan_items),
            "runtime_snapshot": snapshot,
            "assertions": [item.model_dump(exclude_none=True) for item in assertions],
            "evidence_ids": list(evidence_ids),
            "context_evidences": context_evidences,
            "deterministic_issues": [issue.model_dump(exclude_none=True) for issue in deterministic_issues],
        }

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def _resolve_max_issues(self, raw: Any) -> int:
        limit = int(tool_defaults.LOGIC_CHECK_MAX_ISSUES)
        try:
            value = int(raw) if raw is not None else limit
        except Exception:
            value = limit
        if value <= 0:
            value = limit
        return max(1, min(limit, value))

    @staticmethod
    def _merge_issues(
        issues: Sequence[LogicCheckIssue],
        deterministic: Sequence[LogicCheckIssue],
        *,
        max_issues: int,
    ) -> List[LogicCheckIssue]:
        merged: List[LogicCheckIssue] = []
        for issue in deterministic:
            merged.append(issue)
        for issue in issues:
            merged.append(issue)
        if max_issues > 0:
            return merged[:max_issues]
        return merged

    async def _parse_or_repair_json(self, *, messages: List[Dict[str, str]], raw: str) -> Dict[str, Any]:
        parsed = safe_json_loads(raw, expected="dict")
        if isinstance(parsed, dict):
            return parsed
        repaired = await self._attempt_json_repair(messages=messages, raw=raw)
        if isinstance(repaired, dict):
            return repaired
        snippet = (raw or "").strip().replace("\n", "\\n")
        if len(snippet) > self.json_repair_max_raw_chars:
            snippet = snippet[: self.json_repair_max_raw_chars] + "..."
        raise ValueError(f"logic.check returned non-JSON payload. raw_snippet={snippet}")

    async def _attempt_json_repair(self, *, messages: List[Dict[str, str]], raw: str) -> Any:
        if self.json_repair_attempts <= 0:
            return None
        try:
            repaired = await repair_json_from_raw_with_retry(
                llm_connector=self.llm_connector,
                messages=messages,
                raw=str(raw or ""),
                expected="dict",
                temperature=self.json_repair_temperature,
                attempts=self.json_repair_attempts,
                include_today_line=True,
                max_raw_chars=self.json_repair_max_raw_chars,
            )
        except Exception:
            return None
        return repaired if isinstance(repaired, dict) else None
