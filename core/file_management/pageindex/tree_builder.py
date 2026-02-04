import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from config import pageindex as pageindex_cfg
from core.file_management.pageindex.types import HeadingSignal, SectionNode
from core.file_management.pageindex.utils import (
    build_content_list_index,
    build_page_text_index,
    extract_numbering_level,
    iter_markdown_headings,
    max_page_index,
    normalize_heading_text,
    read_json,
    resolve_content_list_path,
)

logger = logging.getLogger(__name__)


@dataclass
class SectionLocator:
    offsets: List[int]
    section_ids: List[str]

    def resolve(self, offset: int) -> Optional[str]:
        if not self.offsets:
            return None
        idx = bisect_right(self.offsets, offset) - 1
        if idx < 0:
            return None
        if idx >= len(self.section_ids):
            return self.section_ids[-1]
        return self.section_ids[idx]


@dataclass
class SectionTreeBuildResult:
    nodes: List[SectionNode]
    locator: SectionLocator
    content_list: List[dict]
    page_texts: Dict[int, str]
    max_page_idx: Optional[int]
    level_conflict_ratio: float
    uniform_level_flattened: bool


def _pick_content_item(items: Sequence[dict]) -> Optional[dict]:
    if not items:
        return None
    scored: List[tuple[int, int, dict]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        text_level = item.get("text_level")
        page_score = int(page_idx) if isinstance(page_idx, int) else 1_000_000
        level_score = int(text_level) if isinstance(text_level, int) else 1_000_000
        scored.append((page_score, level_score, item))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def _resolve_level(
    heading: dict,
    *,
    content_index: Dict[str, List[dict]],
    max_level: int,
    numbering_enabled: bool,
    numbering_max: int,
) -> HeadingSignal:
    title = heading["title"]
    markdown_level = heading.get("markdown_level")
    numbering_level = None
    if numbering_enabled:
        numbering_level = extract_numbering_level(title, max_level=numbering_max)
    normalized = normalize_heading_text(title)
    matched = content_index.get(normalized, [])
    picked = _pick_content_item(matched)
    text_level = None
    page_idx = None
    if picked:
        tl = picked.get("text_level")
        if isinstance(tl, int):
            text_level = min(max(tl, 1), max_level)
        pi = picked.get("page_idx")
        if isinstance(pi, int):
            page_idx = pi

    candidates = []
    for token, source in (
        (numbering_level, "numbering"),
        (text_level, "text_level"),
        (markdown_level, "markdown"),
    ):
        if isinstance(token, int) and token > 0:
            candidates.append((token, source))

    if candidates:
        resolved_level, source = candidates[0]
    else:
        resolved_level, source = 1, "default"

    resolved_level = min(max(resolved_level, 1), max_level)

    return HeadingSignal(
        title=title,
        markdown_level=int(markdown_level) if markdown_level else None,
        numbering_level=numbering_level,
        text_level=text_level,
        page_idx=page_idx,
        line_index=heading["line_index"],
        char_start=heading["char_start"],
        char_end=heading["char_end"],
        resolved_level=resolved_level,
        level_source=source,
    )


def _apply_level_conflicts(
    signals: List[HeadingSignal],
    *,
    conflict_ratio: float,
    force_flat_if_uniform: bool,
) -> tuple[List[HeadingSignal], float, bool]:
    if not signals:
        return signals, 0.0, False
    conflict_count = 0
    for signal in signals:
        candidates = {
            level
            for level in (
                signal.numbering_level,
                signal.text_level,
                signal.markdown_level,
            )
            if isinstance(level, int) and level > 0
        }
        if len(candidates) > 1:
            conflict_count += 1
    ratio = conflict_count / max(len(signals), 1)
    uniform_flattened = False
    if ratio >= conflict_ratio:
        for signal in signals:
            signal.resolved_level = 1
            signal.level_source = "conflict_flatten"
        return signals, ratio, True

    levels = {signal.resolved_level for signal in signals}
    if force_flat_if_uniform and len(levels) == 1 and 1 not in levels:
        for signal in signals:
            signal.resolved_level = 1
            signal.level_source = "uniform_flatten"
        uniform_flattened = True
    return signals, ratio, uniform_flattened


def _assign_page_ranges(
    signals: List[HeadingSignal],
    *,
    max_page_idx: Optional[int],
) -> Dict[int, tuple[Optional[int], Optional[int]]]:
    if not signals:
        return {}
    ranges: Dict[int, tuple[Optional[int], Optional[int]]] = {}
    for idx, signal in enumerate(signals):
        start = signal.page_idx
        end = None
        if start is not None:
            next_page = None
            for future in signals[idx + 1 :]:
                if future.page_idx is not None:
                    next_page = future.page_idx
                    break
            if next_page is not None and next_page >= start:
                end = max(next_page - 1, start)
            elif max_page_idx is not None:
                end = max_page_idx
            else:
                end = start
        ranges[idx] = (start, end)
    return ranges


def build_section_tree(
    *,
    file_id: str,
    markdown: str,
    filename: Optional[str],
    md_path: Optional[str],
    output_dir: Optional[str],
) -> SectionTreeBuildResult:
    headings = iter_markdown_headings(markdown)
    content_list: List[dict] = []
    content_index: Dict[str, List[dict]] = {}
    page_texts: Dict[int, str] = {}
    max_page_idx = None
    content_path = resolve_content_list_path(md_path, output_dir)
    if content_path:
        try:
            loaded = read_json(content_path)
            if isinstance(loaded, list):
                content_list = loaded
                content_index = build_content_list_index(content_list)
                page_texts = build_page_text_index(content_list)
                max_page_idx = max_page_index(content_list)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read MinerU content_list %s: %s", content_path, exc)

    signals: List[HeadingSignal] = []
    max_level = pageindex_cfg.section_level_max()
    for heading in headings:
        signal = _resolve_level(
            heading,
            content_index=content_index,
            max_level=max_level,
            numbering_enabled=pageindex_cfg.section_numbering_enabled(),
            numbering_max=pageindex_cfg.section_numbering_max_level(),
        )
        signals.append(signal)

    signals, conflict_ratio, uniform_flattened = _apply_level_conflicts(
        signals,
        conflict_ratio=pageindex_cfg.section_level_conflict_ratio(),
        force_flat_if_uniform=pageindex_cfg.section_level_force_flat_if_uniform(),
    )

    nodes: List[SectionNode] = []
    offsets: List[int] = []
    ids: List[str] = []

    if not signals:
        title = filename or "Document"
        node = SectionNode(
            section_id=f"{file_id}:sec:1",
            file_id=file_id,
            title=title,
            path=title,
            level=1,
            parent_id=None,
            page_start=0 if max_page_idx is not None else None,
            page_end=max_page_idx,
            heading_start=0,
            heading_end=None,
            level_source="fallback",
        )
        nodes.append(node)
        offsets.append(0)
        ids.append(node.section_id)
        return SectionTreeBuildResult(
            nodes=nodes,
            locator=SectionLocator(offsets=offsets, section_ids=ids),
            content_list=content_list,
            page_texts=page_texts,
            max_page_idx=max_page_idx,
            level_conflict_ratio=conflict_ratio,
            uniform_level_flattened=uniform_flattened,
        )

    ranges = _assign_page_ranges(signals, max_page_idx=max_page_idx)

    stack: List[SectionNode] = []
    section_counter = 0
    path_delimiter = pageindex_cfg.section_path_delimiter()
    for idx, signal in enumerate(signals):
        while stack and signal.resolved_level <= stack[-1].level:
            stack.pop()
        parent = stack[-1] if stack else None
        section_counter += 1
        path = signal.title if parent is None else f"{parent.path}{path_delimiter}{signal.title}"
        page_start, page_end = ranges.get(idx, (None, None))
        node = SectionNode(
            section_id=f"{file_id}:sec:{section_counter}",
            file_id=file_id,
            title=signal.title,
            path=path,
            level=signal.resolved_level,
            parent_id=parent.section_id if parent else None,
            page_start=page_start,
            page_end=page_end,
            heading_start=signal.char_start,
            heading_end=signal.char_end,
            level_source=signal.level_source,
        )
        nodes.append(node)
        offsets.append(signal.char_start)
        ids.append(node.section_id)
        if parent:
            parent.children.append(node.section_id)
        stack.append(node)

    def _inherit_range(node: SectionNode, node_map: Dict[str, SectionNode]) -> tuple[Optional[int], Optional[int]]:
        start = node.page_start
        end = node.page_end
        for child_id in node.children:
            child = node_map[child_id]
            child_range = _inherit_range(child, node_map)
            child_start, child_end = child_range
            if child_start is not None:
                if start is None or child_start < start:
                    start = child_start
            if child_end is not None:
                if end is None or child_end > end:
                    end = child_end
        node.page_start = start
        node.page_end = end
        return start, end

    node_map = {node.section_id: node for node in nodes}
    roots = [node for node in nodes if node.parent_id is None]
    for root in roots:
        _inherit_range(root, node_map)

    return SectionTreeBuildResult(
        nodes=nodes,
        locator=SectionLocator(offsets=offsets, section_ids=ids),
        content_list=content_list,
        page_texts=page_texts,
        max_page_idx=max_page_idx,
        level_conflict_ratio=conflict_ratio,
        uniform_level_flattened=uniform_flattened,
    )
