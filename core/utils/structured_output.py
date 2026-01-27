"""
Structured (JSON) output helpers for LLM calls.

Goals
-----
- Reduce format noise by requiring JSON outputs.
- Validate outputs with Pydantic models (strong schema).
- Reuse shared JSON extraction/repair logic.
"""
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from core.utils.json_extract import safe_json_loads

TModel = TypeVar("TModel", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when an LLM structured output cannot be parsed/validated."""


def parse_pydantic_json_from_llm_text(raw: str, model_cls: type[TModel]) -> TModel:
    """
    Extract JSON from `raw` and validate it with `model_cls`.

    Raises StructuredOutputError with a compact message when parsing/validation fails.
    """
    payload = safe_json_loads(raw, expected="dict")
    if payload is None:
        raise StructuredOutputError("Failed to extract a JSON object from model output")
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        # Keep message readable for logs / error attachment.
        raise StructuredOutputError(f"Pydantic validation failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise StructuredOutputError(f"Unexpected validation error: {exc}") from exc


def as_json_dict(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model into a JSON-safe dict."""
    return model.model_dump(mode="json", exclude_none=True)

