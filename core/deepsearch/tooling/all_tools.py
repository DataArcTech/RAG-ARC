"""Helpers for reporting the full DeepSearch tool catalog in traces."""
from core.deepsearch.tooling import describe_available_tools
from core.deepsearch.tools import builtin_tool_descriptors


def render_all_tools_block(*, include_llm_tools: bool) -> str:
    """Render a stable, human-readable tool catalog for <all_tools> trace."""

    hints = describe_available_tools(include_llm_tools=include_llm_tools)
    hint_map = {str(h.get("name") or ""): h for h in hints if isinstance(h, dict)}
    descriptors = list(builtin_tool_descriptors())

    lines: list[str] = []
    lines.append("available_tools:")
    for desc in sorted(descriptors, key=lambda d: str(d.name)):
        hint = hint_map.get(desc.name) or {}
        profile = (hint.get("profile") or desc.profile or "").strip()
        determinism = (hint.get("determinism") or desc.determinism or "").strip()
        tags = hint.get("strategy_tags")
        tag_text = ""
        if isinstance(tags, list) and tags:
            tag_text = " tags=" + ",".join([str(t) for t in tags if str(t).strip()][:10])

        key_extras: list[str] = []
        try:
            extra_schema = (desc.input_schema or {}).get("properties", {}).get("extra", {})
            extra_props = extra_schema.get("properties", {}) if isinstance(extra_schema, dict) else {}
            if isinstance(extra_props, dict):
                key_extras = sorted([str(k) for k in extra_props.keys() if str(k).strip()])[:12]
        except Exception:
            key_extras = []
        extra_text = f" extra_keys={key_extras}" if key_extras else ""

        lines.append(
            " - "
            + desc.name
            + (f" profile={profile}" if profile else "")
            + (f" determinism={determinism}" if determinism else "")
            + f": {desc.description}{extra_text}{tag_text}"
        )
    return "\n".join(lines).strip()
