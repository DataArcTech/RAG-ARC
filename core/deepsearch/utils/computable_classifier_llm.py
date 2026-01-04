"""LLM-backed classification for computable questions.

Used to route finance/insurance questions that require deterministic handling
(amounts, dates, thresholds, rule checks) into deterministic tool paths.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from core.prompts.deepsearch.computable_classifier import (
    COMPUTABLE_CLASSIFIER_SYSTEM_PROMPT_V1,
    COMPUTABLE_CLASSIFIER_USER_PROMPT_V1,
)
from core.utils.json_extract import safe_json_loads


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
        {"role": "system", "content": COMPUTABLE_CLASSIFIER_SYSTEM_PROMPT_V1},
        {"role": "user", "content": COMPUTABLE_CLASSIFIER_USER_PROMPT_V1.format(question=normalized)},
    ]

    async_chat = getattr(llm, "achat", None)
    if callable(async_chat):
        raw = await async_chat(messages, temperature=float(temperature), model=model) if model else await async_chat(messages, temperature=float(temperature))
    else:
        chat = getattr(llm, "chat", None)
        if not callable(chat):
            raise RuntimeError("LLM connector does not expose chat/achat methods.")
        raw = chat(messages, temperature=float(temperature), model=model) if model else chat(messages, temperature=float(temperature))

    payload = safe_json_loads(raw or "", expected="dict")
    if not isinstance(payload, dict):
        raise ValueError("Computable classifier returned non-JSON output.")
    try:
        return ComputableLLMClassification.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid computable classifier payload: {exc}") from exc

