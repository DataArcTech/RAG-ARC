#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from api.routers.deepsearch_weaver_render import render_trace_payload, weaver_block


@dataclass(frozen=True)
class ExtractedTrace:
    run_id: str
    question: str | None
    owner_id: str | None
    answer: str | None
    trace_events: list[dict[str, Any]]


def _try_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_fix_mojibake(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except Exception:
        return text
    return repaired if repaired else text


def _extract_from_session_messages(payload: dict[str, Any], *, message_index: int | None) -> ExtractedTrace:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("session_messages.json must contain a non-empty `data` list")

    candidate_indices: list[int] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict) and item.get("deepsearch_trace"):
            candidate_indices.append(idx)

    if not candidate_indices:
        raise ValueError("No message contains `deepsearch_trace`")

    if message_index is None:
        chosen_index = candidate_indices[-1]
    else:
        chosen_index = message_index if message_index >= 0 else len(data) + message_index
        if chosen_index < 0 or chosen_index >= len(data):
            raise ValueError(f"message_index out of range: {message_index}")
        if chosen_index not in candidate_indices:
            raise ValueError(f"message_index={message_index} does not contain `deepsearch_trace`")

    message = data[chosen_index]
    trace = message.get("deepsearch_trace")
    if not isinstance(trace, dict):
        raise ValueError("deepsearch_trace must be an object")

    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    run_id = str(metadata.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("deepsearch_trace.metadata.run_id missing")

    question = _try_fix_mojibake(metadata.get("query") if isinstance(metadata.get("query"), str) else None)
    owner_id = None
    state_owner = (trace.get("deepsearch_result") or {}).get("state") if isinstance(trace.get("deepsearch_result"), dict) else None
    if isinstance(state_owner, dict) and state_owner.get("owner_id"):
        owner_id = str(state_owner.get("owner_id"))

    content = message.get("content") if isinstance(message.get("content"), dict) else {}
    answer = content.get("content") if isinstance(content.get("content"), str) else None

    trace_events = trace.get("trace_events")
    if not isinstance(trace_events, list):
        raise ValueError("deepsearch_trace.trace_events missing")

    return ExtractedTrace(
        run_id=run_id,
        question=question,
        owner_id=owner_id,
        answer=answer,
        trace_events=[ev for ev in trace_events if isinstance(ev, dict)],
    )


def _extract_from_trace_object(payload: dict[str, Any]) -> ExtractedTrace:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    run_id = str(metadata.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("trace json must contain metadata.run_id")
    question = _try_fix_mojibake(metadata.get("query") if isinstance(metadata.get("query"), str) else None)
    owner_id = None
    state = (payload.get("deepsearch_result") or {}).get("state") if isinstance(payload.get("deepsearch_result"), dict) else None
    if isinstance(state, dict) and state.get("owner_id"):
        owner_id = str(state.get("owner_id"))

    answer = None
    deepsearch_result = payload.get("deepsearch_result") if isinstance(payload.get("deepsearch_result"), dict) else {}
    report = deepsearch_result.get("report") if isinstance(deepsearch_result.get("report"), dict) else {}
    if isinstance(report.get("answer"), str):
        answer = report.get("answer")

    trace_events = payload.get("trace_events")
    if not isinstance(trace_events, list):
        raise ValueError("trace json must contain trace_events[]")

    return ExtractedTrace(
        run_id=run_id,
        question=question,
        owner_id=owner_id,
        answer=answer,
        trace_events=[ev for ev in trace_events if isinstance(ev, dict)],
    )


def extract_trace(path: Path, *, message_index: int | None) -> ExtractedTrace:
    payload = _try_load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("trace input must be a JSON object")
    if "trace_events" in payload and "metadata" in payload:
        return _extract_from_trace_object(payload)
    return _extract_from_session_messages(payload, message_index=message_index)


def _render_blocks(trace: ExtractedTrace) -> Iterable[str]:
    for item in trace.trace_events:
        tag = item.get("tag") or "progress"
        content = item.get("content") or ""
        rendered_tag, rendered = render_trace_payload(trace_tag=str(tag), content=str(content), run_id=trace.run_id)
        yield weaver_block(rendered_tag, rendered)


def write_report(
    *,
    trace: ExtractedTrace,
    output_path: Path,
    answer_override: str | None,
    question_override: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    question = question_override or trace.question
    answer = answer_override or trace.answer

    lines: list[str] = []
    lines.append("DeepSearch Trace Report (human-readable)")
    lines.append("")
    if trace.owner_id or trace.owner_id == "":
        lines.append(f"owner_id: {trace.owner_id}")
    if trace.run_id:
        lines.append(f"run_id: {trace.run_id}")
    if question:
        lines.append("")
        lines.append("q:")
        lines.append(question)
    lines.append("")
    lines.append("---")
    lines.append("trace:")
    lines.append("")
    lines.extend(_render_blocks(trace))
    if answer:
        lines.append("")
        lines.append("---")
        lines.append("report:")
        lines.append("")
        lines.append(answer.rstrip())

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render DeepSearch trace events into weaver blocks for docs.")
    parser.add_argument(
        "--trace-input",
        type=Path,
        required=True,
        help="Path to JSON file: either /session/{id}/messages response or a deepsearch_trace object JSON.",
    )
    parser.add_argument(
        "--message-index",
        type=int,
        default=None,
        help="If trace-input is session_messages.json, which message index to use (default: last with deepsearch_trace).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/test_deepsearch/deepsearch_trace_test_pdf.txt"),
        help="Output report path.",
    )
    parser.add_argument("--question", type=str, default=None, help="Override question rendered in the report header.")
    parser.add_argument("--answer-path", type=Path, default=None, help="Override answer markdown from a file path.")
    args = parser.parse_args()

    trace = extract_trace(args.trace_input, message_index=args.message_index)
    answer_override = args.answer_path.read_text(encoding="utf-8") if args.answer_path else None
    write_report(
        trace=trace,
        output_path=args.output,
        answer_override=answer_override,
        question_override=args.question,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

