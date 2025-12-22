"""Tool hint registry for DeepSearch planner prompts.

Avoid module-level mutable globals so multiple DeepSearchService instances (API/MCP/CLI)
can coexist in the same process with different configurations.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set


@dataclass
class ToolHintRegistry:
    """In-memory registry for extra tool hints + disabled tool names."""

    _additional_hints: List[Dict[str, str]] = field(default_factory=list)
    _disabled_tools: Set[str] = field(default_factory=set)
    _revision: int = 0

    def register_tool_hints(self, hints: Iterable[Dict[str, str]]) -> None:
        updated = False
        for hint in hints:
            if not isinstance(hint, dict) or "name" not in hint:
                continue
            normalized = self._normalize_hint(hint)
            if self._replace_or_append(normalized):
                updated = True
        if updated:
            self._bump_revision()

    def set_disabled_tools(self, names: Iterable[str]) -> None:
        normalized = {str(name) for name in names if str(name).strip()}
        if normalized == self._disabled_tools:
            return
        self._disabled_tools = normalized
        self._bump_revision()

    def clear(self) -> None:
        self._additional_hints.clear()
        self._disabled_tools.clear()
        self._bump_revision()

    def get_registered_hints(self) -> List[Dict[str, str]]:
        return list(self._additional_hints)

    def get_disabled_tool_names(self) -> Set[str]:
        return set(self._disabled_tools)

    def get_revision(self) -> int:
        return int(self._revision)

    # ------------------------------------------------------------------
    def _bump_revision(self) -> None:
        self._revision += 1

    @staticmethod
    def _normalize_hint(hint: Dict[str, str]) -> Dict[str, str]:
        name = str(hint["name"])
        return {
            "name": name,
            "channel": str(hint.get("channel", "")),
            "description": str(hint.get("description", "")),
            "profile": str(hint.get("profile", "")),
            "determinism": str(hint.get("determinism", "")),
            "namespace": str(hint.get("namespace", "")),
            "speed": str(hint.get("speed", "")),
            "cost": str(hint.get("cost", "")),
            "strategy_tags": list(hint.get("strategy_tags", [])),
        }

    def _replace_or_append(self, hint: Dict[str, str]) -> bool:
        for idx, existing in enumerate(self._additional_hints):
            if existing.get("name") == hint["name"]:
                if existing == hint:
                    return False
                self._additional_hints[idx] = hint
                return True
        self._additional_hints.append(hint)
        return True


DEFAULT_TOOL_HINT_REGISTRY = ToolHintRegistry()

