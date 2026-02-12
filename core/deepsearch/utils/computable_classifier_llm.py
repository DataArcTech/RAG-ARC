"""LLM-backed classification for computable questions.

Used to route finance/insurance questions that require deterministic handling
(amounts, dates, thresholds, rule checks) into deterministic tool paths.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from core.prompts.deepsearch.computable_classifier import (
    COMPUTABLE_CLASSIFIER_SYSTEM_PROMPT_V1_EN,
    COMPUTABLE_CLASSIFIER_USER_PROMPT_V1_EN,
)
from core.utils.llm_json import call_llm_json_with_retry


class ComputableLLMClassification(BaseModel):
    is_computable: bool = Field(False, description="Whether the question is computable.")
    reasons: List[str] = Field(default_factory=list, description="Brief reasons supporting the classification.")
    suggested_tools: List[str] = Field(default_factory=list, description="Suggested deterministic tool names when computable.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


async def aclassify_computable_question(
    llm: Any,
    *,
    question: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> ComputableLLMClassification:
    if llm is None:
        raise RuntimeError("LLM connector is required for computable classification.")
    normalized = (question or "").strip()
    if not normalized:
        return ComputableLLMClassification(is_computable=False, reasons=["empty_question"], suggested_tools=[])

    messages = [
        {"role": "system", "content": COMPUTABLE_CLASSIFIER_SYSTEM_PROMPT_V1_EN},
        {"role": "user", "content": COMPUTABLE_CLASSIFIER_USER_PROMPT_V1_EN.format(question=normalized)},
    ]

    llm_kwargs: Dict[str, Any] = {}
    if model:
        llm_kwargs["model"] = model
    payload = await call_llm_json_with_retry(
        llm_connector=llm,
        messages=messages,
        expected="dict",
        temperature=float(temperature),
        llm_kwargs=llm_kwargs or None,
    )
    if not isinstance(payload, dict):
        raise ValueError("Computable classifier returned non-JSON output.")
    try:
        return ComputableLLMClassification.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid computable classifier payload: {exc}") from exc
