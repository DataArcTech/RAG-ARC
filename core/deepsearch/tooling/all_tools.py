"""Helpers for reporting the full DeepSearch tool catalog in traces."""
import os

from core.deepsearch.tooling import describe_available_tools
from core.deepsearch.tools import builtin_tool_descriptors
from core.deepsearch.tooling.registry import ToolHintRegistry


def render_all_tools_block(*, include_llm_tools: bool, registry: ToolHintRegistry | None = None) -> str:
    """Render a stable, human-readable tool catalog for <all_tools> trace.

    Default output is intentionally compact ("perfect is no more to remove").
    Set `DEEPSEARCH_WEAVER_ALL_TOOLS_MODE=full` to emit the full catalog.
    """

    mode = (str(os.getenv("DEEPSEARCH_WEAVER_ALL_TOOLS_MODE", "compact")) or "").strip().lower()

    hints = describe_available_tools(registry=registry, include_llm_tools=include_llm_tools)
    hint_map = {str(h.get("name") or ""): h for h in hints if isinstance(h, dict)}
    descriptors = list(builtin_tool_descriptors())
    disabled = registry.get_disabled_tool_names() if registry is not None else set()

    if mode != "full":
        profiles = {"F": 0, "X": 0, "H": 0}
        for desc in descriptors:
            hint = hint_map.get(desc.name) or {}
            profile = (hint.get("profile") or desc.profile or "").strip().upper() or "F"
            if profile not in profiles:
                continue
            profiles[profile] += 1
        total = len(descriptors)
        disabled_text = f"disabled_tools: {len(disabled)}" if disabled else "disabled_tools: 0"
        return "\n".join(
            [
                "available_tools:",
                f" - builtin_tools: {len(descriptors)} total={total}",
                f"profiles: F={profiles['F']} X={profiles['X']} H={profiles['H']}",
                disabled_text,
                "note: set DEEPSEARCH_WEAVER_ALL_TOOLS_MODE=full for full tool list",
            ]
        ).strip()

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
