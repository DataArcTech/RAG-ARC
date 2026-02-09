"""Template mode for DeepSearch initial planning.

This module supports:
- Maintaining a small registry of plan templates (config-managed).
- Using a light LLM to select a template + fill slots.
- Instantiating an initial plan + initial tool calls to seed the think loop.

Notes:
- This is intentionally conservative. If the selector returns invalid JSON or an unknown template_id,
  the caller should fall back to the normal heavy initial think.
"""
import hashlib
import json
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config.core.deepsearch import plan_template_defaults
from config.core.deepsearch.plan_templates import PLAN_TEMPLATES, PLAN_TEMPLATES_VERSION
from core.deepsearch.utils.llm_json import call_llm_json_with_retry
from core.prompts.deepsearch.plan_templates import (
    PLAN_TEMPLATE_SELECTOR_SYSTEM_PROMPT_EN,
    PLAN_TEMPLATE_SELECTOR_USER_PROMPT_TEMPLATE_EN,
)


@dataclass(frozen=True)
class PlanTemplate:
    template_id: str
    label: str
    description: str
    slots: Dict[str, str]
    plan_items: List[str]
    initial_tool_calls: List[Dict[str, Any]]
    report_style: str


@dataclass(frozen=True)
class PlanTemplateSelection:
    use_template: bool
    template_id: str | None
    slots: Dict[str, str]
    report_needed: bool
    report_style: str
    reasoning: str
    raw: Dict[str, Any]


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_template_fingerprint() -> str:
    """Fingerprint used for cache busting when templates change."""

    blob = _stable_json_dumps({"version": PLAN_TEMPLATES_VERSION, "templates": PLAN_TEMPLATES})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def coerce_templates(raw: Sequence[Mapping[str, Any]] | None = None) -> Dict[str, PlanTemplate]:
    """Load templates from config into validated objects (best-effort)."""

    out: Dict[str, PlanTemplate] = {}
    for item in list(raw or PLAN_TEMPLATES):
        if not isinstance(item, Mapping):
            continue
        template_id = str(item.get("template_id") or "").strip()
        if not template_id:
            continue
        slots = item.get("slots") if isinstance(item.get("slots"), Mapping) else {}
        plan_items = item.get("plan_items") if isinstance(item.get("plan_items"), list) else []
        initial_tool_calls = item.get("initial_tool_calls") if isinstance(item.get("initial_tool_calls"), list) else []
        out[template_id] = PlanTemplate(
            template_id=template_id,
            label=str(item.get("label") or "").strip() or template_id,
            description=str(item.get("description") or "").strip(),
            slots={str(k): str(v) for k, v in slots.items() if str(k).strip()},
            plan_items=[str(x) for x in plan_items if str(x).strip()],
            initial_tool_calls=[dict(x) for x in initial_tool_calls if isinstance(x, Mapping)],
            report_style=str(item.get("report_style") or "deepsearch").strip().lower() or "deepsearch",
        )
    return out


def _render_text(template: str, mapping: Mapping[str, str]) -> str:
    """Render {placeholders} safely; missing keys become empty strings."""

    fmt = string.Formatter()
    pieces: List[str] = []
    for literal, field, spec, conv in fmt.parse(template):
        pieces.append(literal)
        if field is None:
            continue
        key = str(field)
        value = str(mapping.get(key, ""))
        # Ignore format spec / conversion for safety and predictability.
        pieces.append(value)
    return "".join(pieces).strip()


def _render_obj(obj: Any, mapping: Mapping[str, str]) -> Any:
    """Recursively render strings inside dict/list structures."""

    if isinstance(obj, str):
        return _render_text(obj, mapping)
    if isinstance(obj, list):
        return [_render_obj(x, mapping) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _render_obj(v, mapping) for k, v in obj.items()}
    return obj


def instantiate_template_plan(
    *,
    templates: Mapping[str, PlanTemplate],
    template_id: str,
    question: str,
    slots: Mapping[str, str] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Instantiate plan items + initial tool calls from a template."""

    template = templates.get(template_id)
    if template is None:
        raise KeyError(f"Unknown template_id: {template_id}")

    slot_map: Dict[str, str] = {}
    if isinstance(slots, Mapping):
        for key in template.slots.keys():
            val = str(slots.get(key, "") or "").strip()
            if val:
                slot_map[key] = val

    render_map = {"question": str(question or "").strip(), **slot_map}

    plan_items: List[Dict[str, Any]] = []
    for raw in template.plan_items:
        text = _render_text(str(raw), render_map)
        if text:
            plan_items.append({"text": text, "checked": False})

    tool_calls: List[Dict[str, Any]] = []
    for call in template.initial_tool_calls:
        if not isinstance(call, Mapping):
            continue
        tool_name = str(call.get("tool_name") or "").strip()
        if not tool_name:
            continue
        raw_args = call.get("tool_args") if isinstance(call.get("tool_args"), Mapping) else {}
        tool_args = _render_obj(raw_args, render_map)
        if not isinstance(tool_args, dict):
            tool_args = {}
        tool_calls.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "rationale": str(call.get("rationale") or "").strip() or None,
                "parallelizable": bool(call.get("parallelizable") or False),
            }
        )

    # Compact string for observability: a short template signature.
    signature = f"{template.template_id}@{PLAN_TEMPLATES_VERSION}"
    return plan_items, tool_calls, signature


async def select_plan_template(
    *,
    llm_connector: Any,
    question: str,
    templates: Mapping[str, PlanTemplate] | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attempts: int | None = None,
) -> PlanTemplateSelection | None:
    """Select a plan template + slots using a light LLM (strict JSON)."""

    if llm_connector is None:
        return None
    q = str(question or "").strip()
    if not q:
        return None

    registry = dict(templates or coerce_templates())
    if not registry:
        return None

    # Keep templates payload compact: only include fields needed for selection.
    template_rows: List[Dict[str, Any]] = []
    for tid, tmpl in registry.items():
        template_rows.append(
            {
                "template_id": tid,
                "label": tmpl.label,
                "description": tmpl.description,
                "slots": tmpl.slots,
            }
        )

    messages = [
        {"role": "system", "content": PLAN_TEMPLATE_SELECTOR_SYSTEM_PROMPT_EN},
        {
            "role": "user",
            "content": PLAN_TEMPLATE_SELECTOR_USER_PROMPT_TEMPLATE_EN.format(
                question=q,
                templates_json=_stable_json_dumps(template_rows),
            ),
        },
    ]

    if temperature is None:
        temperature = plan_template_defaults.DEFAULT_INITIAL_THINK_TEMPLATE_TEMPERATURE
    if max_tokens is None:
        max_tokens = plan_template_defaults.DEFAULT_INITIAL_THINK_TEMPLATE_MAX_TOKENS
    if attempts is None:
        attempts = plan_template_defaults.DEFAULT_INITIAL_THINK_TEMPLATE_JSON_ATTEMPTS

    llm_kwargs: Dict[str, Any] = {}
    if model_name is None:
        model_name = plan_template_defaults.DEFAULT_INITIAL_THINK_TEMPLATE_MODEL_NAME
    if isinstance(model_name, str) and model_name.strip():
        llm_kwargs["model"] = model_name.strip()

    payload = await call_llm_json_with_retry(
        llm_connector=llm_connector,
        messages=messages,
        expected="object",
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        attempts=int(attempts),
        llm_kwargs=llm_kwargs or None,
    )
    if not isinstance(payload, dict):
        return None

    use_template = bool(payload.get("use_template"))
    template_id = payload.get("template_id")
    template_id = str(template_id).strip() if isinstance(template_id, str) else None
    slots_raw = payload.get("slots") if isinstance(payload.get("slots"), dict) else {}
    slots: Dict[str, str] = {str(k): str(v) for k, v in slots_raw.items() if str(k).strip() and str(v).strip()}
    report_needed = payload.get("report_needed")
    report_needed = bool(report_needed) if report_needed is not None else True
    report_style = str(payload.get("report_style") or "deepsearch").strip().lower()
    if report_style not in {"deepsearch", "research"}:
        report_style = "deepsearch"
    reasoning = str(payload.get("reasoning") or "").strip()

    # Safety: unknown template_id -> treat as no-template.
    if use_template and template_id and template_id not in registry:
        use_template = False
        template_id = None

    return PlanTemplateSelection(
        use_template=use_template,
        template_id=template_id,
        slots=slots,
        report_needed=report_needed,
        report_style=report_style,
        reasoning=reasoning,
        raw=dict(payload),
    )
