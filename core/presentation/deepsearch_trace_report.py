from dataclasses import dataclass
from typing import Any, Iterable

from core.presentation.deepsearch_weaver_render import render_trace_payload, weaver_block


@dataclass(frozen=True)
class DeepSearchTraceEvent:
    tag: str
    content: str
    meta: dict[str, Any]


def render_weaver_blocks(*, run_id: str, trace_events: Iterable[DeepSearchTraceEvent]) -> list[str]:
    blocks: list[str] = []
    for event in trace_events:
        tag, rendered = render_trace_payload(trace_tag=str(event.tag), content=str(event.content), run_id=str(run_id))
        blocks.append(weaver_block(tag, rendered))
    return blocks


def build_debug_report_text(
    *,
    run_id: str,
    owner_id: str | None,
    question: str | None,
    answer: str | None,
    trace_events: Iterable[DeepSearchTraceEvent],
) -> str:
    lines: list[str] = []
    lines.append("DeepSearch Trace Report (human-readable)")
    lines.append("")
    if owner_id is not None:
        lines.append(f"owner_id: {owner_id}")
    lines.append(f"run_id: {run_id}")
    if question:
        lines.append("")
        lines.append("q:")
        lines.append(str(question))
    lines.append("")
    lines.append("---")
    lines.append("trace:")
    lines.append("")
    lines.extend(render_weaver_blocks(run_id=run_id, trace_events=trace_events))
    if answer:
        lines.append("")
        lines.append("---")
        lines.append("report:")
        lines.append("")
        lines.append(str(answer).rstrip())
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DeepSearchTraceEvent",
    "build_debug_report_text",
    "render_weaver_blocks",
]

