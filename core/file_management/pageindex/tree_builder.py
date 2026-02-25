import json
import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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
from framework.virtual_paths import IO_PATH_PREFIX, io_key, is_io_path

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


def _normalize_levels(
    signals: List[HeadingSignal],
) -> tuple[List[HeadingSignal], float, bool]:
    """Min-level normalization: offset so min heading becomes L1."""
    if not signals:
        return signals, 0.0, False
    # Conflict ratio — diagnostic only, never flattens
    conflict_count = sum(
        1 for s in signals
        if len({l for l in (s.numbering_level, s.text_level, s.markdown_level)
                if isinstance(l, int) and l > 0}) > 1
    )
    ratio = conflict_count / max(len(signals), 1)
    # Min-level normalization
    min_level = min(s.resolved_level for s in signals)
    offset = min_level - 1
    if offset > 0:
        for s in signals:
            s.resolved_level = s.resolved_level - offset
    return signals, ratio, False


def _assign_page_ranges(
    signals: List[HeadingSignal],
    *,
    max_page_idx: Optional[int],
    max_span: int,
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
            # Clamp to max_span
            if end - start > max_span:
                end = start + max_span
        ranges[idx] = (start, end)
    return ranges


def build_section_tree(
    *,
    file_id: str,
    markdown: str,
    filename: Optional[str],
    md_path: Optional[str],
    output_dir: Optional[str],
    io_manager: Any = None,
) -> SectionTreeBuildResult:
    headings = iter_markdown_headings(markdown)
    content_list: List[dict] = []
    content_index: Dict[str, List[dict]] = {}
    page_texts: Dict[int, str] = {}
    max_page_idx = None

    def _load_content_list_from_io(io_dir: str, *, expected_basename: Optional[str]) -> List[dict]:
        """Load MinerU `*_content_list*.json` from an io:// directory using IOManager.

        NOTE: Do not resolve io:// to a local filesystem path here. The IOManager abstraction
        owns backend mapping (LocalDB now, MinIO later).
        """

        if io_manager is None:
            return []
        try:
            keys = io_manager.list_keys_path(io_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list io:// keys under %s: %s", io_dir, exc)
            return []

        def _basename(ref: str) -> str:
            token = str(ref or "").strip()
            return token.rsplit("/", 1)[-1] if "/" in token else token

        expected = (expected_basename or "").strip()
        expected_hit = None
        matches: List[str] = []
        for ref in keys:
            base = _basename(ref)
            if expected and base == expected:
                expected_hit = str(ref)
                break
            if base.endswith(".json") and "_content_list" in base:
                matches.append(str(ref))

        candidate = expected_hit or (sorted(matches)[0] if matches else None)
        if not candidate:
            return []
        try:
            raw = io_manager.get_text_path(candidate)
            if raw is None:
                return []
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, list) else []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read MinerU content_list %s: %s", candidate, exc)
            return []

    # MinerU outputs are persisted under io://..., so PageIndex must be able to load content_list
    # via IOManager (LocalDB/MinIO). For local filesystem paths we keep the glob-based behavior.
    md_path_token = str(md_path or "").strip()
    output_dir_token = str(output_dir or "").strip()
    expected_content_list_basename = None
    if md_path_token:
        md_base = md_path_token.rsplit("/", 1)[-1]
        if md_base.endswith(".md"):
            expected_content_list_basename = md_base[:-3] + "_content_list.json"

    loaded_list: List[dict] = []
    if output_dir_token and is_io_path(output_dir_token):
        loaded_list = _load_content_list_from_io(output_dir_token, expected_basename=expected_content_list_basename)
    elif md_path_token and is_io_path(md_path_token):
        parent_key = "/".join(io_key(md_path_token).split("/")[:-1])
        if parent_key:
            loaded_list = _load_content_list_from_io(f"{IO_PATH_PREFIX}{parent_key}", expected_basename=expected_content_list_basename)

    if not loaded_list:
        content_path = resolve_content_list_path(md_path, output_dir)
        if content_path:
            try:
                loaded = read_json(content_path)
                if isinstance(loaded, list):
                    loaded_list = loaded
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read MinerU content_list %s: %s", content_path, exc)

    if loaded_list:
        content_list = loaded_list
        content_index = build_content_list_index(content_list)
        page_texts = build_page_text_index(content_list)
        max_page_idx = max_page_index(content_list)

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

    signals, conflict_ratio, uniform_flattened = _normalize_levels(signals)

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

    max_span = pageindex_cfg.max_page_num_each_node()
    ranges = _assign_page_ranges(signals, max_page_idx=max_page_idx, max_span=max_span)

    stack: List[SectionNode] = []
    section_counter = 0
    path_delimiter = pageindex_cfg.section_path_delimiter()

    # Front-matter node: content before first heading
    if signals[0].char_start > 0:
        section_counter += 1
        first_page = ranges.get(0, (None, None))[0]
        fm_end = max(first_page - 1, 0) if isinstance(first_page, int) else None
        fm_node = SectionNode(
            section_id=f"{file_id}:sec:{section_counter}",
            file_id=file_id,
            title="Front Matter",
            path="Front Matter",
            level=1,
            parent_id=None,
            page_start=0 if max_page_idx is not None else None,
            page_end=fm_end,
            heading_start=0,
            heading_end=signals[0].char_start,
            level_source="front_matter",
        )
        nodes.append(fm_node)
        offsets.append(0)
        ids.append(fm_node.section_id)

    for idx, signal in enumerate(signals):
        # Level-jump capping: prevent jumps > 1 level deeper than parent
        if stack and signal.resolved_level > stack[-1].level + 1:
            signal.resolved_level = stack[-1].level + 1
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
            child_start, child_end = _inherit_range(child, node_map)
            if child_start is not None:
                if start is None or child_start < start:
                    start = child_start
            if child_end is not None:
                if end is None or child_end > end:
                    end = child_end
        # Clamp parent span
        if start is not None and end is not None and end - start > max_span:
            end = start + max_span
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
