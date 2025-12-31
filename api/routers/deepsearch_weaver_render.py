import json
from typing import Any, Dict

from config.output_limits import (
    DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS,
    DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT,
)

_WEAVER_ALLOWED_TAGS = {
    "all_tools",
    "think",
    "write_outline",
    "tool_call",
    "tool_response",
    "write",
    "progress",
    "terminate",
}


def _truncate_text(value: str, *, limit: int) -> str:
    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _try_parse_json(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not (text.startswith("{") or text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _render_tool_call(payload: Dict[str, Any]) -> str:
    tool_name = str(payload.get("tool_name") or payload.get("tool") or "").strip() or "unknown"
    call_id = str(payload.get("call_id") or "").strip()
    plan_step = str(payload.get("plan_step") or "").strip()
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}

    query = ""
    for key in ("query", "focus_query", "question", "prompt"):
        if isinstance(extra.get(key), str) and extra[key].strip():
            query = extra[key].strip()
            break

    channel = ""
    if isinstance(extra.get("channel"), str):
        channel = extra.get("channel") or ""

    lines: list[str] = []
    header = f"tool={tool_name}"
    if call_id:
        header += f" call_id={call_id}"
    if plan_step:
        header += f" plan_step={plan_step}"
    if channel:
        header += f" channel={channel}"
    lines.append(header)

    descriptor = payload.get("descriptor") if isinstance(payload.get("descriptor"), dict) else {}
    if descriptor:
        profile = descriptor.get("profile")
        determinism = descriptor.get("determinism")
        about = str(descriptor.get("description") or "").strip()
        if profile or determinism:
            lines.append(
                "tool_meta="
                + json.dumps(
                    {"profile": profile, "determinism": determinism},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
        if about:
            lines.append("about=" + _truncate_text(about, limit=200))

    if query:
        lines.append("query=" + _truncate_text(query, limit=280))

    routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
    if routing:
        lines.append(
            "routing="
            + json.dumps(
                {
                    "can_route_remote": routing.get("can_route_remote"),
                    "prefer_remote": routing.get("prefer_remote"),
                    "has_local": routing.get("has_local"),
                    "default_mcp_server": routing.get("default_mcp_server"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )

    coverage = payload.get("coverage_metrics") if isinstance(payload.get("coverage_metrics"), dict) else {}
    if coverage:
        lines.append(
            "coverage="
            + json.dumps(
                {
                    "evidence_count": coverage.get("evidence_count"),
                    "coverage_ratio": coverage.get("coverage_ratio"),
                    "coverage_score": coverage.get("coverage_score"),
                    "completed_steps": coverage.get("completed_steps"),
                    "total_steps": coverage.get("total_steps"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )

    return "\n".join([ln for ln in lines if str(ln).strip()])


def _evidence_preview(evidence: Dict[str, Any]) -> str:
    chunk_id = str(evidence.get("chunk_id") or evidence.get("evidence_id") or "").strip()
    source = str(evidence.get("source") or "").strip()
    score = evidence.get("score")

    prefix = f"- {chunk_id}" if chunk_id else "- (no_chunk_id)"
    if source:
        prefix += f" source={source}"
    if isinstance(score, (int, float)):
        prefix += f" score={float(score):.4f}"

    preview_limit = int(DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS)
    content = str(evidence.get("content") or "").strip().replace("\n", " ")
    if content:
        prefix += "\n  " + _truncate_text(content, limit=max(80, preview_limit))

    provenance = evidence.get("provenance") if isinstance(evidence.get("provenance"), dict) else None
    if provenance:
        meta = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
        path = (
            provenance.get("path")
            or provenance.get("file_path")
            or provenance.get("source_path")
            or meta.get("path")
            or meta.get("file_path")
            or meta.get("source_path")
        )
        url = provenance.get("url") or provenance.get("source_url") or meta.get("url") or meta.get("source_url")
        if path:
            prefix += f"\n  path={path}"
        if url:
            prefix += f"\n  url={url}"

    return prefix


def _render_tool_response(payload: Dict[str, Any]) -> str:
    tool_name = str(payload.get("tool_name") or payload.get("tool") or "").strip() or "unknown"
    call_id = str(payload.get("call_id") or "").strip()
    ok = payload.get("ok")
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}

    lines: list[str] = []
    header = f"tool={tool_name}"
    if call_id:
        header += f" call_id={call_id}"
    if ok is not None:
        header += f" ok={bool(ok)}"
    lines.append(header)

    error_message = extra.get("error") or extra.get("error_message")
    if isinstance(error_message, str) and error_message.strip():
        lines.append("error=" + _truncate_text(error_message.strip(), limit=600))

    result_obj = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    evidences = None
    if isinstance(result_obj, dict):
        evidences = result_obj.get("evidences")
    if not isinstance(evidences, list):
        evidences = payload.get("evidences")
    if isinstance(evidences, list) and evidences:
        sample_n = max(1, int(DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT))
        lines.append(f"evidence_count={len(evidences)} sample={min(len(evidences), sample_n)}")
        for item in evidences[:sample_n]:
            if isinstance(item, dict):
                lines.append(_evidence_preview(item))

        triple_samples: list[dict[str, Any]] = []
        for raw in evidences[:sample_n]:
            if not isinstance(raw, dict):
                continue
            prov = raw.get("provenance")
            if not isinstance(prov, dict):
                continue
            triples = prov.get("triples")
            if isinstance(triples, list):
                for triple in triples:
                    if (
                        isinstance(triple, dict)
                        and triple.get("head")
                        and triple.get("relation")
                        and triple.get("tail")
                    ):
                        triple_samples.append(triple)
            if len(triple_samples) >= 6:
                break
        if triple_samples:
            lines.append("triple_samples:")
            for triple in triple_samples[:6]:
                lines.append(f"- {triple.get('head')} -[{triple.get('relation')}]-> {triple.get('tail')}")

    # Intentionally avoid dumping raw result JSON here to keep the block human-readable and stable for tests.

    return "\n".join([ln for ln in lines if str(ln).strip()])


def render_trace_payload(*, trace_tag: str, content: str, run_id: str) -> tuple[str, str]:
    normalized = (trace_tag or "").strip().lower() or "think"
    text = str(content or "")
    parsed = _try_parse_json(text)

    if isinstance(parsed, dict) and "event" in parsed and "data" in parsed:
        event_type = str(parsed.get("event") or "event")
        payload = parsed.get("data")
        rendered = format_weaver_progress(event_type, payload, run_id=run_id)
        if event_type.strip().lower() in {"error", "done"}:
            return "terminate", rendered
        return "progress", rendered

    if normalized == "tool_call" and isinstance(parsed, dict):
        return normalized, _render_tool_call(parsed)
    if normalized == "tool_response" and isinstance(parsed, dict):
        return normalized, _render_tool_response(parsed)
    return normalized, text


def weaver_block(tag: str, content: str) -> str:
    normalized = (tag or "").strip().lower()
    if normalized not in _WEAVER_ALLOWED_TAGS:
        normalized = "progress"
    body = str(content or "")
    if body.endswith("\n"):
        body = body.rstrip("\n")
    if body:
        return f"<{normalized}>\n{body}\n</{normalized}>"
    return f"<{normalized}>\n</{normalized}>"


def format_weaver_progress(event_type: str, payload: Any, *, run_id: str) -> str:
    """Render non-trace task events as human-readable progress blocks."""

    normalized = (event_type or "").strip().lower() or "event"
    if normalized == "heartbeat":
        return f"heartbeat run_id={run_id}"

    if not isinstance(payload, dict):
        return "\n".join([f"event={normalized}", f"run_id={run_id}", f"payload={payload}"])

    stage = str(payload.get("stage") or "").strip()
    progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    stage_record = payload.get("stage_record") if isinstance(payload.get("stage_record"), dict) else {}
    errors = payload.get("errors") if isinstance(payload.get("errors"), dict) else {}
    stage_history = payload.get("stage_history") if isinstance(payload.get("stage_history"), list) else []

    lines: list[str] = [f"event={normalized}", f"run_id={run_id}"]
    if stage:
        lines.append(f"stage={stage}")
    if progress:
        lines.append(
            "progress.stage="
            + str(progress.get("stage"))
            + " progress.percent="
            + str(progress.get("percent"))
            + " progress.step="
            + str(progress.get("step_index"))
            + "/"
            + str(progress.get("step_total"))
        )
    if stage_record:
        lines.append(f"stage_record.stage={stage_record.get('stage')} stage_record.timestamp={stage_record.get('timestamp')}")
        rec_meta = stage_record.get("metadata")
        if isinstance(rec_meta, dict) and rec_meta:
            lines.append("stage_record.metadata=" + json.dumps(rec_meta, ensure_ascii=False, separators=(",", ":"), default=str))
    if errors:
        lines.append("errors=" + json.dumps(errors, ensure_ascii=False, separators=(",", ":"), default=str))
    if stage_history:
        lines.append("stage_history:")
        for item in stage_history:
            if not isinstance(item, dict):
                continue
            meta_text = ""
            hist_meta = item.get("metadata")
            if isinstance(hist_meta, dict) and hist_meta:
                meta_text = " " + json.dumps(hist_meta, ensure_ascii=False, separators=(",", ":"), default=str)
            lines.append(f"- {item.get('timestamp')} {item.get('stage')}{meta_text}")
    return "\n".join(lines)
