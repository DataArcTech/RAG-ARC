import json
from typing import Any, Dict, List, Optional

"""Public/dev artifact rendering for DeepSearch.

Policy: do NOT truncate tool outputs or evidence content. EvidencePool acts as external memory and
debugging requires full visibility into model actions and tool results.
"""


def _clamp_non_negative(value: Any, *, default: int) -> int:
    try:
        num = int(value)
    except Exception:
        return int(default)
    return max(0, num)


def _truncate_list(items: Any, *, limit: int) -> List[Any]:
    if not isinstance(items, list):
        return []
    return list(items)


def _truncate_text(value: str, *, limit: int) -> str:
    # No truncation.
    return str(value or "")


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
            lines.append("about=" + _truncate_text(about, limit=0))

    if query:
        lines.append("query=" + _truncate_text(query, limit=0))

    if tool_name == "code.python":
        runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        pyver = runtime.get("python_version")
        plat = runtime.get("platform")
        allowed = runtime.get("allowed_imports")
        packages = runtime.get("packages_available")
        if pyver or plat:
            lines.append(
                "env="
                + json.dumps(
                    {"python_version": pyver, "platform": plat},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
        if isinstance(allowed, list) and allowed:
            lines.append("allowed_imports=" + json.dumps(allowed, ensure_ascii=False, separators=(",", ":"), default=str))
        if isinstance(packages, dict) and packages:
            lines.append(
                "packages_available="
                + json.dumps(packages, ensure_ascii=False, separators=(",", ":"), default=str)
            )
        purpose = extra.get("purpose")
        if isinstance(purpose, str) and purpose.strip():
            lines.append("purpose=" + _truncate_text(purpose.strip(), limit=0))
        code = extra.get("code")
        if isinstance(code, str) and code.strip():
            preview = code.strip().replace("\n", " ")
            lines.append(
                "code_preview=" + _truncate_text(preview, limit=0)
            )

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

    content = str(evidence.get("content") or "").strip().replace("\n", " ")
    if content:
        prefix += "\n  " + content
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
        lines.append("error=" + _truncate_text(error_message.strip(), limit=0))

    top_error = payload.get("error")
    if isinstance(top_error, str) and top_error.strip():
        lines.append("error=" + _truncate_text(top_error.strip(), limit=0))

    result_obj = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if tool_name == "code.python" and isinstance(result_obj, dict):
        diagnostics = result_obj.get("diagnostics") if isinstance(result_obj.get("diagnostics"), dict) else {}
        exec_status = diagnostics.get("exec_status")
        if exec_status:
            lines.append(f"exec_status={exec_status}")
        purpose = diagnostics.get("purpose")
        if isinstance(purpose, str) and purpose.strip():
            lines.append("purpose=" + _truncate_text(purpose.strip(), limit=0))
        result_text = diagnostics.get("result_text")
        if isinstance(result_text, str) and result_text.strip():
            lines.append("result:")
            lines.append(result_text.rstrip())
        err = diagnostics.get("error")
        if isinstance(err, dict) and (err.get("message") or err.get("type")):
            message = str(err.get("message") or err.get("type"))
            lines.append(
                "exec_error=" + _truncate_text(message, limit=0)
            )
        return "\n".join([ln for ln in lines if str(ln).strip()])

    evidences = None
    if isinstance(result_obj, dict):
        evidences = result_obj.get("evidences")
    if not isinstance(evidences, list):
        evidences = payload.get("evidences")
    if isinstance(evidences, list) and evidences:
        lines.append(f"evidence_count={len(evidences)}")
        for item in evidences:
            if isinstance(item, dict):
                lines.append(_evidence_preview(item))
    return "\n".join([ln for ln in lines if str(ln).strip()])


def _render_public_trace_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_tag = str(event.get("tag") or "").strip().lower()
    if not raw_tag:
        return None

    tag_map = {
        "think": "think",
        "write_outline": "plan",
        "tool_call": "tool_call",
        "tool_response": "tool_response",
    }
    out_type = tag_map.get(raw_tag)
    if not out_type:
        return None

    content = str(event.get("content") or "")
    parsed = _try_parse_json(content)
    rendered = content
    if raw_tag == "tool_call" and isinstance(parsed, dict):
        rendered = _render_tool_call(parsed)
    elif raw_tag == "tool_response" and isinstance(parsed, dict):
        rendered = _render_tool_response(parsed)

    rendered = str(rendered or "").strip()
    if not rendered:
        return None
    return {"type": out_type, "content": rendered}


def build_ref(*, file: str, json_pointer: str | None = None, enabled: bool = True) -> Dict[str, Any]:
    if not str(file or "").strip():
        raise ValueError("ref file is required")
    if json_pointer is not None and json_pointer != "" and not str(json_pointer).startswith("/"):
        raise ValueError("json_pointer must start with '/' when provided")
    payload: Dict[str, Any] = {"file": str(file)}
    if json_pointer:
        payload["json_pointer"] = str(json_pointer)
    if enabled:
        return {"$ref": payload}
    return payload


def _artifact_map(
    *,
    refs_enabled: bool,
    files: Dict[str, str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, filename in (files or {}).items():
        entry_key = str(key or "").strip()
        entry_file = str(filename or "").strip()
        if not entry_key or not entry_file:
            continue
        payload[entry_key] = build_ref(file=entry_file, enabled=refs_enabled)
    return payload


def build_manifest_v2(
    *,
    snapshot: Dict[str, Any],
    stage_timings: Dict[str, Any],
    refs_enabled: bool,
    profile_files: Dict[str, str],
    artifact_files: Dict[str, str],
    artifact_version: int = 2,
) -> Dict[str, Any]:
    run_id = snapshot.get("run_id")
    if not str(run_id or "").strip():
        raise ValueError("snapshot.run_id is required")
    payload: Dict[str, Any] = {
        "artifact_version": int(artifact_version),
        "run_id": str(run_id),
        "config_fingerprint": snapshot.get("config_fingerprint"),
        "stage": snapshot.get("stage"),
        "stage_history": list(snapshot.get("stage_history") or []),
        "kpis": dict(snapshot.get("kpis") or {}),
        "error_summary": dict(snapshot.get("error_summary") or {}),
        "artifacts": _artifact_map(refs_enabled=refs_enabled, files=artifact_files),
        "profiles": dict(profile_files or {}),
    }
    if stage_timings:
        payload["stage_timings"] = dict(stage_timings)
    return payload


def build_dev_view_v2(
    *,
    snapshot: Dict[str, Any],
    stage_timings: Dict[str, Any],
    refs_enabled: bool,
    include_reasoning: bool,
    include_report: bool,
    include_plan_result: bool,
) -> Dict[str, Any]:
    run_id = snapshot.get("run_id")
    if not str(run_id or "").strip():
        raise ValueError("snapshot.run_id is required")

    plan_id = None
    plan_metadata = snapshot.get("plan_metadata")
    if isinstance(plan_metadata, dict):
        plan_id = plan_metadata.get("plan_id")
    plan_steps = snapshot.get("plan_steps")
    step_count = len(plan_steps) if isinstance(plan_steps, list) else 0

    payload: Dict[str, Any] = {
        "artifact_version": 2,
        "profile": "dev",
        "run_id": str(run_id),
        "config_fingerprint": snapshot.get("config_fingerprint"),
        "stage": snapshot.get("stage"),
        "stage_history": list(snapshot.get("stage_history") or []),
        "plan_summary": {"plan_id": plan_id, "step_count": step_count},
        "cost_telemetry": dict(snapshot.get("cost_telemetry") or {}),
        "errors": list(snapshot.get("errors") or []),
        "request_metadata": dict(snapshot.get("request_metadata") or {}),
        "kpis": dict(snapshot.get("kpis") or {}),
        "error_summary": dict(snapshot.get("error_summary") or {}),
    }
    if stage_timings:
        payload["stage_timings"] = dict(stage_timings)

    if include_plan_result:
        payload["plan_ref"] = build_ref(file="plan_result.json", json_pointer="/plan", enabled=refs_enabled)
    if include_reasoning:
        payload["reasoning_ref"] = build_ref(file="reasoning.json", enabled=refs_enabled)
    if include_report:
        payload["report_ref"] = build_ref(file="report.json", enabled=refs_enabled)
    return payload


def build_public_view_v2(
    *,
    snapshot: Dict[str, Any],
    stage_timings: Dict[str, Any],
    refs_enabled: bool,
    trace_events: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    # Public json is a minimal event stream (with XML-like tags).
    payload: Dict[str, Any] = {
        "dev_ref": build_ref(file="dev.json", enabled=refs_enabled),
        "events": [],
    }
    events_out: List[Dict[str, Any]] = []
    for raw in trace_events or []:
        if not isinstance(raw, dict):
            continue
        rendered = _render_public_trace_event(raw)
        if rendered is not None:
            events_out.append(rendered)
    payload["events"] = events_out
    return payload


def build_v2_artifact_documents(
    *,
    snapshot: Dict[str, Any],
    stage_timings: Dict[str, Any],
    artifacts_config: Dict[str, Any],
    artifacts_present: Dict[str, bool],
    trace_events: List[Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return filename -> payload for v2 artifacts (manifest/dev/public/state_snapshot)."""
    version = int(artifacts_config.get("version") or 2)
    refs_enabled = bool((artifacts_config.get("refs") or {}).get("enabled", True))
    profiles = artifacts_config.get("profiles") or []
    if not isinstance(profiles, list):
        profiles = []

    include_reasoning = bool(artifacts_present.get("reasoning"))
    include_report = bool(artifacts_present.get("report"))
    include_plan_result = bool(artifacts_present.get("plan_result", True))

    profile_files: Dict[str, str] = {}
    documents: Dict[str, Dict[str, Any]] = {}

    if "dev" in profiles:
        profile_files["dev"] = "dev.json"
        documents["dev.json"] = build_dev_view_v2(
            snapshot=snapshot,
            stage_timings=stage_timings,
            refs_enabled=refs_enabled,
            include_reasoning=include_reasoning,
            include_report=include_report,
            include_plan_result=include_plan_result,
        )
    if "public" in profiles:
        profile_files["public"] = "public.json"
        documents["public.json"] = build_public_view_v2(
            snapshot=snapshot,
            stage_timings=stage_timings,
            refs_enabled=refs_enabled,
            trace_events=trace_events,
        )

    artifact_files: Dict[str, str] = {}
    if include_plan_result:
        artifact_files["plan_result"] = "plan_result.json"
    if bool(artifacts_present.get("evidence_pool")):
        dedupe_cfg = artifacts_config.get("dedupe") if isinstance(artifacts_config, dict) else None
        if not isinstance(dedupe_cfg, dict):
            dedupe_cfg = {}
        filename = str(dedupe_cfg.get("evidence_pool_filename") or "evidence_pool.json").strip()
        if filename:
            artifact_files["evidence_pool"] = filename
    if include_reasoning:
        artifact_files["reasoning"] = "reasoning.json"
    if include_report:
        artifact_files["report"] = "report.json"
    if bool(artifacts_present.get("report_md")):
        artifact_files["report_md"] = "report.md"
    if bool(artifacts_present.get("stage_timings")):
        artifact_files["stage_timings"] = "stage_timings.json"
    for key, filename in profile_files.items():
        artifact_files[f"profile_{key}"] = filename

    documents["manifest.json"] = build_manifest_v2(
        snapshot=snapshot,
        stage_timings=stage_timings,
        refs_enabled=refs_enabled,
        profile_files=profile_files,
        artifact_files=artifact_files,
        artifact_version=version,
    )

    snapshot_mode = str(artifacts_config.get("state_snapshot_mode") or "manifest").strip().lower()
    if snapshot_mode == "legacy":
        documents["state_snapshot.json"] = dict(snapshot)
    else:
        documents["state_snapshot.json"] = dict(documents["manifest.json"])
    return documents
