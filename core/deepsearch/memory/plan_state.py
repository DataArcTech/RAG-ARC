"""Runtime plan memory for DeepSearch."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from pydantic import ValidationError

from encapsulation.data_model.deepsearch import PlanItem


def _normalize_plan_items(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        items = raw.get("items")
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    normalized: List[Dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            normalized.append({"text": text, "checked": False})
            continue
        try:
            model = PlanItem.model_validate(item)
        except ValidationError:
            continue
        text = model.text.strip()
        if not text:
            continue
        normalized.append({"text": text, "checked": bool(model.checked)})
    return normalized


def _render_plan_markdown(items: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        checked = bool(item.get("checked"))
        marker = "x" if checked else " "
        lines.append(f"- [{marker}] {text}")
    return "\n".join(lines)


@dataclass
class PlanState:
    items: List[Dict[str, Any]] = field(default_factory=list)
    markdown: str = ""
    version: int = 0

    def update(self, raw_items: Any) -> bool:
        normalized = _normalize_plan_items(raw_items)
        if normalized == self.items:
            return False
        self.items = normalized
        self.markdown = _render_plan_markdown(normalized)
        self.version += 1
        return True

    def to_payload(self) -> Dict[str, Any]:
        return {"items": list(self.items), "markdown": self.markdown, "version": self.version}


def update_plan_from_think_notes(
    plan_state: PlanState,
    *,
    think_notes: Sequence[Any],
) -> bool:
    if plan_state is None or not think_notes:
        return False
    extracted = None
    for note in reversed(list(think_notes)):
        metadata = None
        if isinstance(note, dict):
            metadata = note.get("metadata")
        else:
            metadata = getattr(note, "metadata", None)
        raw = metadata.get("raw") if isinstance(metadata, dict) else None
        if isinstance(raw, dict) and "plan" in raw:
            extracted = raw.get("plan")
            break
    if extracted is None:
        return False
    return plan_state.update(extracted)
