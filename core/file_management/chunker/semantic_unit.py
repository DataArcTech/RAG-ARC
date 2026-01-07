import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.file_management.chunker.base import AbstractChunker
from core.file_management.atomic_units.fenced_code import (
    is_fence_line,
    parse_fenced_code_block,
    split_fenced_code_blocks_with_line_spans,
)
from core.file_management.atomic_units.markdown_image import extract_image_alts, extract_image_urls, line_contains_image
from core.file_management.atomic_units.markdown_table import is_markdown_table_start, parse_markdown_table
from core.file_management.chunker.semantic_unit_table_summary import summarize_tail_percent_rows
from core.utils.html_table import extract_html_table_rows, render_pipe_table

if TYPE_CHECKING:
    from config.core.file_management.chunker.chunker_config import SemanticUnitChunkerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Segment:
    kind: str  # "text" | "table" | "code" | "list" | "math" | "blockquote" | "image"
    content: str
    caption: str = ""
    info: Dict[str, Any] = field(default_factory=dict)


class SemanticUnitChunker(AbstractChunker):
    """
    SemanticUnitChunker implements a hierarchical (anchor/slice) chunking strategy.

    Phase A (basic) focuses on Markdown tables:
    - Small tables become a single anchor chunk.
    - Large tables become an anchor (caption + header) plus multiple slices (header + row windows).

    Phase B (standard/advanced) adds:
    - fenced code blocks (atomic anchors; no slicing)
    - markdown lists (anchor + slices by item windows)
    - display math blocks (atomic anchors)
    - blockquotes (atomic anchors; advanced only)

    Any non-recognized text falls back to a configured fallback chunker.
    """

    _BLOCKQUOTE_RE = re.compile(r"^\s*>\s?.*")
    _LIST_ITEM_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
    _ORDERED_ITEM_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
    _TASK_ITEM_RE = re.compile(r"^(\s*)([-*+])\s+\[( |x|X)\]\s+(.*)$")
    _MATH_BLOCK_START_RE = re.compile(r"^\s*(\$\$|\\\[)\s*$")
    _MATH_BLOCK_END_RE = re.compile(r"^\s*(\$\$|\\\])\s*$")
    _HTML_TABLE_START_RE = re.compile(r"<table\b", re.IGNORECASE)
    _HTML_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)

    def __init__(self, config: "SemanticUnitChunkerConfig"):
        super().__init__(config)
        self._fallback = self.config.fallback_chunker_config.build()

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        level = kwargs.get("level", getattr(self.config, "level", "basic"))
        if level == "disabled":
            chunks = self._fallback.chunk_text(text=text, metadata=metadata, **kwargs)
            for chunk in chunks:
                chunk_meta = chunk.get("metadata") or {}
                chunk_meta.setdefault("chunk_role", "text")
                chunk_meta.setdefault("semantic_unit_type", "text")
                chunk["metadata"] = chunk_meta
            return chunks

        enabled_units = self._enabled_units(level)
        segments = self._split_markdown(
            text,
            enable_tables=enabled_units["table"],
            enable_code=enabled_units["code"],
            enable_lists=enabled_units["list"],
            enable_math=enabled_units["math"],
            enable_blockquotes=enabled_units["blockquote"],
            enable_images=True,
        )
        source_file_id = ""
        if metadata:
            source_file_id = str(metadata.get("source_file_id") or "").strip()

        table_small_max_tokens = kwargs.get(
            "table_small_max_tokens", getattr(self.config, "table_small_max_tokens", 400)
        )
        table_slice_max_tokens = kwargs.get(
            "table_slice_max_tokens", getattr(self.config, "table_slice_max_tokens", 300)
        )
        table_slice_overlap_rows = kwargs.get(
            "table_slice_overlap_rows", getattr(self.config, "table_slice_overlap_rows", 2)
        )

        code_small_max_tokens = kwargs.get("code_small_max_tokens", getattr(self.config, "code_small_max_tokens", 400))
        code_slice_max_tokens = kwargs.get("code_slice_max_tokens", getattr(self.config, "code_slice_max_tokens", 300))
        code_slice_overlap_lines = kwargs.get(
            "code_slice_overlap_lines", getattr(self.config, "code_slice_overlap_lines", 3)
        )
        code_anchor_preview_lines = kwargs.get(
            "code_anchor_preview_lines", getattr(self.config, "code_anchor_preview_lines", 8)
        )

        list_small_max_tokens = kwargs.get("list_small_max_tokens", getattr(self.config, "list_small_max_tokens", 400))
        list_slice_max_tokens = kwargs.get("list_slice_max_tokens", getattr(self.config, "list_slice_max_tokens", 300))
        list_slice_overlap_items = kwargs.get(
            "list_slice_overlap_items", getattr(self.config, "list_slice_overlap_items", 1)
        )
        list_anchor_preview_items = kwargs.get(
            "list_anchor_preview_items", getattr(self.config, "list_anchor_preview_items", 8)
        )

        math_small_max_tokens = kwargs.get("math_small_max_tokens", getattr(self.config, "math_small_max_tokens", 2000))
        blockquote_small_max_tokens = kwargs.get(
            "blockquote_small_max_tokens", getattr(self.config, "blockquote_small_max_tokens", 600)
        )

        output: List[Dict[str, Any]] = []
        table_counter = 0
        code_counter = 0
        list_counter = 0
        math_counter = 0
        blockquote_counter = 0
        image_counter = 0

        for segment in segments:
            if segment.kind == "text":
                produced = self._fallback.chunk_text(text=segment.content, metadata=metadata, **kwargs)
                for chunk in produced:
                    chunk_meta = chunk.get("metadata") or {}
                    chunk_meta.setdefault("chunk_role", "text")
                    chunk_meta.setdefault("semantic_unit_type", "text")
                    chunk["metadata"] = chunk_meta
                output.extend(produced)
                continue

            caption = segment.caption.strip()

            if segment.kind == "table":
                table_counter += 1
                semantic_unit_id = f"{source_file_id}:table:{table_counter}"

                table_format = str((segment.info or {}).get("table_format") or "pipe").strip().lower()
                if table_format == "html":
                    html = segment.content.strip()
                    rows = extract_html_table_rows(html)
                    pipe_table = render_pipe_table(rows)
                    normalized_table = pipe_table or html
                    summary = summarize_tail_percent_rows(rows)
                    body = f"{summary}\n\n{normalized_table}".strip() if summary else normalized_table
                    content = self._compose_with_caption(caption, body)
                    output.append(
                        self._make_chunk_dict(
                            content=content,
                            source_metadata=metadata,
                            metadata={
                                "strategy": "semantic_unit",
                                "chunk_role": "anchor",
                                "semantic_unit_type": "table",
                                "semantic_unit_id": semantic_unit_id,
                                "parent_unit_id": semantic_unit_id,
                                "table_format": "pipe" if pipe_table else "html",
                                "index_text": content,
                                "token_count": self._estimate_tokens(content),
                                "anchor_is_summary": True,
                                "is_atomic": True,
                                "is_full_content": True,
                            },
                        )
                    )
                    continue

                header, separator, rows = parse_markdown_table(segment.content)
                table_header = "\n".join([line for line in (header, separator) if line]).strip()

                full_table = segment.content.strip()
                full_token_count = self._estimate_tokens(full_table)

                if full_token_count <= table_small_max_tokens:
                    content = self._compose_with_caption(caption, full_table)
                    output.append(
                        self._make_chunk_dict(
                            content=content,
                            source_metadata=metadata,
                            metadata={
                                "strategy": "semantic_unit",
                                "chunk_role": "anchor",
                                "semantic_unit_type": "table",
                                "semantic_unit_id": semantic_unit_id,
                                "parent_unit_id": semantic_unit_id,
                                "table_caption": caption,
                                "table_header": table_header,
                                "index_text": content,
                                "token_count": self._estimate_tokens(content),
                                "anchor_is_summary": False,
                                "is_atomic": True,
                                "is_full_content": True,
                            },
                        )
                    )
                    continue

                index_text = self._compose_with_caption(caption, table_header)
                anchor_content = index_text
                output.append(
                    self._make_chunk_dict(
                        content=anchor_content,
                        source_metadata=metadata,
                        metadata={
                            "strategy": "semantic_unit",
                            "chunk_role": "anchor",
                            "semantic_unit_type": "table",
                            "semantic_unit_id": semantic_unit_id,
                            "parent_unit_id": semantic_unit_id,
                            "table_caption": caption,
                            "table_header": table_header,
                            "index_text": index_text,
                            "token_count": self._estimate_tokens(anchor_content),
                            "anchor_is_summary": True,
                            "is_full_content": True,
                        },
                    )
                )

                if not rows:
                    continue

                row_groups = self._slice_table_rows(
                    caption=caption,
                    header=header,
                    separator=separator,
                    rows=rows,
                    slice_max_tokens=table_slice_max_tokens,
                    overlap_rows=table_slice_overlap_rows,
                )
                for slice_index, (row_start, row_end, slice_body) in enumerate(row_groups, start=1):
                    slice_index_text = self._compose_with_caption(
                        caption,
                        "\n".join([line for line in (header, separator, slice_body) if line]).strip(),
                    )
                    # Store each slice as an intact sub-table (header + row window).
                    slice_content = slice_index_text
                    output.append(
                        self._make_chunk_dict(
                            content=slice_content,
                            source_metadata=metadata,
                            metadata={
                                "strategy": "semantic_unit",
                                "chunk_role": "slice",
                                "semantic_unit_type": "table",
                                "semantic_unit_id": semantic_unit_id,
                                "parent_unit_id": semantic_unit_id,
                                "anchor_chunk_id": None,
                                "slice_index": slice_index,
                                "row_range": {"start": row_start, "end": row_end},
                                "table_caption": caption,
                                "table_header": table_header,
                                "index_text": slice_index_text,
                                "token_count": self._estimate_tokens(slice_content),
                                "is_full_content": True,
                            },
                        )
                    )
                continue

            if segment.kind == "image":
                image_counter += 1
                semantic_unit_id = f"{source_file_id}:image:{image_counter}"
                content = self._compose_with_caption(caption, segment.content.strip())
                urls: List[str] = []
                alts: List[str] = []
                for line in segment.content.splitlines():
                    urls.extend(extract_image_urls(line))
                    alts.extend(extract_image_alts(line))
                urls = [u for u in urls if u]
                alts = [a for a in alts if a]

                index_text_parts: List[str] = []
                if caption:
                    index_text_parts.append(caption)
                if alts:
                    index_text_parts.append("; ".join(alts))
                index_text = "\n".join(index_text_parts).strip() or "image"

                output.append(
                    self._make_chunk_dict(
                        content=content,
                        source_metadata=metadata,
                        metadata={
                            "strategy": "semantic_unit",
                            "chunk_role": "anchor",
                            "semantic_unit_type": "image",
                            "semantic_unit_id": semantic_unit_id,
                            "parent_unit_id": semantic_unit_id,
                            "image_urls": urls,
                            "image_alts": alts,
                            "image_count": len(urls),
                            "index_text": index_text,
                            "token_count": self._estimate_tokens(index_text),
                            "anchor_is_summary": True,
                            "is_atomic": True,
                            "is_full_content": True,
                        },
                    )
                )
                continue

            if segment.kind == "code":
                code_counter += 1
                semantic_unit_id = f"{source_file_id}:code:{code_counter}"
                fence, language, body_lines = self._split_fenced_code(segment.content)
                full_code = segment.content.strip()

                # Code blocks are low-priority in the current plan: keep each fenced block as a
                # single, atomic chunk so evidence/prompt never sees a broken fence.
                content = self._compose_with_caption(caption, full_code)

                # Keep indexing text reasonably bounded for embedding/graph extraction safety.
                # Note: we keep the stored content intact; only index_text may be summarized.
                index_text = content
                code_index_max_tokens = max(int(code_small_max_tokens), 2000)
                full_token_count = self._estimate_tokens(content)
                anchor_is_summary = False
                if full_token_count > code_index_max_tokens:
                    head_n = max(int(code_anchor_preview_lines), 0)
                    head_lines = body_lines[:head_n]
                    tail_lines = body_lines[-head_n:] if head_n and len(body_lines) > head_n else []
                    bridge = ["…"] if tail_lines else []
                    preview_lines = head_lines + bridge + tail_lines
                    preview = self._fence_wrap(fence=fence, language=language, body_lines=preview_lines)
                    index_text = self._compose_with_caption(caption, preview)
                    anchor_is_summary = True

                output.append(
                    self._make_chunk_dict(
                        content=content,
                        source_metadata=metadata,
                        metadata={
                            "strategy": "semantic_unit",
                            "chunk_role": "anchor",
                            "semantic_unit_type": "code",
                            "semantic_unit_id": semantic_unit_id,
                            "parent_unit_id": semantic_unit_id,
                            "code_language": language,
                            "index_text": index_text,
                            "token_count": self._estimate_tokens(index_text),
                            "anchor_is_summary": anchor_is_summary,
                            "is_atomic": True,
                            "is_full_content": True,
                        },
                    )
                )
                continue

            if segment.kind == "list":
                list_counter += 1
                semantic_unit_id = f"{source_file_id}:list:{list_counter}"
                items, list_type = self._parse_markdown_list_items(segment.content)
                full_list = segment.content.strip()
                full_token_count = self._estimate_tokens(self._compose_with_caption(caption, full_list))

                if full_token_count <= list_small_max_tokens or len(items) <= 1:
                    content = self._compose_with_caption(caption, full_list)
                    output.append(
                        self._make_chunk_dict(
                            content=content,
                            source_metadata=metadata,
                            metadata={
                                "strategy": "semantic_unit",
                                "chunk_role": "anchor",
                                "semantic_unit_type": "list",
                                "semantic_unit_id": semantic_unit_id,
                                "parent_unit_id": semantic_unit_id,
                                "list_type": list_type,
                                "index_text": content,
                                "token_count": self._estimate_tokens(content),
                                "anchor_is_summary": False,
                                "is_atomic": True,
                                "is_full_content": True,
                            },
                        )
                    )
                    continue

                preview_items = items[: max(int(list_anchor_preview_items), 0)]
                anchor_body = "\n".join(preview_items).strip()
                index_text = self._compose_with_caption(caption, anchor_body)
                anchor_content = index_text
                output.append(
                    self._make_chunk_dict(
                        content=anchor_content,
                        source_metadata=metadata,
                        metadata={
                            "strategy": "semantic_unit",
                            "chunk_role": "anchor",
                            "semantic_unit_type": "list",
                            "semantic_unit_id": semantic_unit_id,
                            "parent_unit_id": semantic_unit_id,
                            "list_type": list_type,
                            "index_text": index_text,
                            "token_count": self._estimate_tokens(anchor_content),
                            "anchor_is_summary": True,
                            "is_full_content": True,
                        },
                    )
                )

                item_groups = self._slice_list_items(
                    caption=caption,
                    items=items,
                    slice_max_tokens=list_slice_max_tokens,
                    overlap_items=list_slice_overlap_items,
                )
                for slice_index, (item_start, item_end, slice_body) in enumerate(item_groups, start=1):
                    slice_index_text = self._compose_with_caption(caption, slice_body)
                    slice_content = slice_index_text
                    output.append(
                        self._make_chunk_dict(
                            content=slice_content,
                            source_metadata=metadata,
                            metadata={
                                "strategy": "semantic_unit",
                                "chunk_role": "slice",
                                "semantic_unit_type": "list",
                                "semantic_unit_id": semantic_unit_id,
                                "parent_unit_id": semantic_unit_id,
                                "anchor_chunk_id": None,
                                "slice_index": slice_index,
                                "list_type": list_type,
                                "list_item_range": {"start": item_start, "end": item_end},
                                "index_text": slice_index_text,
                                "token_count": self._estimate_tokens(slice_content),
                                "is_full_content": True,
                            },
                        )
                    )
                continue

            if segment.kind == "math":
                math_counter += 1
                semantic_unit_id = f"{source_file_id}:math:{math_counter}"
                full_math = segment.content.strip()
                content = self._compose_with_caption(caption, full_math)
                is_atomic = self._estimate_tokens(content) <= math_small_max_tokens
                output.append(
                    self._make_chunk_dict(
                        content=content,
                        source_metadata=metadata,
                        metadata={
                            "strategy": "semantic_unit",
                            "chunk_role": "anchor",
                            "semantic_unit_type": "math",
                            "semantic_unit_id": semantic_unit_id,
                            "parent_unit_id": semantic_unit_id,
                            "index_text": content,
                            "token_count": self._estimate_tokens(content),
                            "anchor_is_summary": False,
                            "is_atomic": is_atomic,
                            "is_full_content": True,
                        },
                    )
                )
                continue

            if segment.kind == "blockquote":
                blockquote_counter += 1
                semantic_unit_id = f"{source_file_id}:blockquote:{blockquote_counter}"
                full_quote = segment.content.strip()
                content = self._compose_with_caption(caption, full_quote)
                is_atomic = self._estimate_tokens(content) <= blockquote_small_max_tokens
                output.append(
                    self._make_chunk_dict(
                        content=content,
                        source_metadata=metadata,
                        metadata={
                            "strategy": "semantic_unit",
                            "chunk_role": "anchor",
                            "semantic_unit_type": "blockquote",
                            "semantic_unit_id": semantic_unit_id,
                            "parent_unit_id": semantic_unit_id,
                            "index_text": content,
                            "token_count": self._estimate_tokens(content),
                            "anchor_is_summary": False,
                            "is_atomic": is_atomic,
                            "is_full_content": True,
                        },
                    )
                )
                continue

        logger.info(
            "SemanticUnitChunker produced %d chunks (tables=%d code=%d lists=%d math=%d quotes=%d)",
            len(output),
            table_counter,
            code_counter,
            list_counter,
            math_counter,
            blockquote_counter,
        )
        return output

    def get_chunker_info(self) -> Dict[str, Any]:
        return {
            "strategy": "semantic_unit",
            "supported_features": [
                "table_anchor_slice",
                "code_fence_atomic",
                "list_anchor_slice",
                "math_blocks",
                "blockquotes",
                "fallback_chunker",
            ],
            "parameters": {
                "level": getattr(self.config, "level", "basic"),
                "table_small_max_tokens": getattr(self.config, "table_small_max_tokens", 400),
                "table_slice_max_tokens": getattr(self.config, "table_slice_max_tokens", 300),
                "table_slice_overlap_rows": getattr(self.config, "table_slice_overlap_rows", 2),
                "code_small_max_tokens": getattr(self.config, "code_small_max_tokens", 400),
                "code_slice_max_tokens": getattr(self.config, "code_slice_max_tokens", 300),
                "code_slice_overlap_lines": getattr(self.config, "code_slice_overlap_lines", 3),
                "code_anchor_preview_lines": getattr(self.config, "code_anchor_preview_lines", 8),
                "list_small_max_tokens": getattr(self.config, "list_small_max_tokens", 400),
                "list_slice_max_tokens": getattr(self.config, "list_slice_max_tokens", 300),
                "list_slice_overlap_items": getattr(self.config, "list_slice_overlap_items", 1),
                "list_anchor_preview_items": getattr(self.config, "list_anchor_preview_items", 8),
                "math_small_max_tokens": getattr(self.config, "math_small_max_tokens", 2000),
                "blockquote_small_max_tokens": getattr(self.config, "blockquote_small_max_tokens", 600),
                "fallback_chunker": getattr(getattr(self.config, "fallback_chunker_config", None), "type", None),
            },
        }

    @staticmethod
    def _make_chunk_dict(
        *,
        content: str,
        metadata: Dict[str, Any],
        source_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chunk_dict: Dict[str, Any] = {
            "content": content,
            "metadata": metadata,
        }
        if source_metadata:
            chunk_dict["source_metadata"] = dict(source_metadata)
        return chunk_dict

    @staticmethod
    def _compose_with_caption(caption: str, body: str) -> str:
        caption = (caption or "").strip()
        body = (body or "").strip()
        if caption and body:
            return f"{caption}\n{body}"
        return caption or body

    def _estimate_tokens(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        try:
            import tiktoken

            model_name = getattr(self.config, "encoding_model_name", None) or getattr(
                getattr(self.config, "fallback_chunker_config", None), "model_name", None
            )
            encoding_name = getattr(
                getattr(self.config, "fallback_chunker_config", None), "encoding_name", "gpt2"
            )
            if model_name:
                enc = tiktoken.encoding_for_model(model_name)
            else:
                enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            word_tokens = len(re.findall(r"[A-Za-z0-9_]+", text))
            punct_tokens = len(re.findall(r"[^\sA-Za-z0-9_\u4e00-\u9fff]", text))
            return cjk_chars + word_tokens + punct_tokens

    @staticmethod
    def _enabled_units(level: str) -> dict[str, bool]:
        level = str(level or "").strip().lower()
        if level == "advanced":
            return {"table": True, "code": True, "list": True, "math": True, "blockquote": True}
        if level == "standard":
            return {"table": True, "code": True, "list": True, "math": True, "blockquote": False}
        if level == "basic":
            return {"table": True, "code": False, "list": False, "math": False, "blockquote": False}
        return {"table": False, "code": False, "list": False, "math": False, "blockquote": False}

    def _split_markdown(
        self,
        text: str,
        *,
        enable_tables: bool,
        enable_code: bool,
        enable_lists: bool,
        enable_math: bool,
        enable_blockquotes: bool,
        enable_images: bool,
    ) -> List[_Segment]:
        lines = text.splitlines()
        segments: List[_Segment] = []
        buffer: List[str] = []

        def flush_text() -> None:
            if not buffer:
                return
            content = "\n".join(buffer).strip("\n")
            if content.strip():
                segments.append(_Segment(kind="text", content=content))
            buffer.clear()

        for kind, span_start, span_end in split_fenced_code_blocks_with_line_spans(text):
            if kind == "code":
                if enable_code:
                    flush_text()
                    code_content = "\n".join(lines[span_start:span_end]).strip("\n")
                    caption = self._infer_block_caption(lines, span_start)
                    segments.append(_Segment(kind="code", content=code_content, caption=caption))
                else:
                    buffer.extend(lines[span_start:span_end])
                continue

            i = span_start
            while i < span_end:
                line = lines[i]

                if enable_math and self._is_math_block_start(line):
                    flush_text()
                    start = i
                    i += 1
                    while i < span_end and not self._is_math_block_end(lines[i]):
                        i += 1
                    if i < span_end:
                        i += 1
                    math_content = "\n".join(lines[start:i]).strip("\n")
                    caption = self._infer_block_caption(lines, start)
                    segments.append(_Segment(kind="math", content=math_content, caption=caption))
                    continue

                if enable_tables and i + 1 < span_end and is_markdown_table_start(lines, i):
                    flush_text()
                    start = i
                    i += 2
                    while i < span_end and lines[i].strip() and ("|" in lines[i]):
                        i += 1
                    table_content = "\n".join(lines[start:i]).strip("\n")
                    caption = self._infer_block_caption(lines, start)
                    segments.append(_Segment(kind="table", content=table_content, caption=caption, info={"table_format": "pipe"}))
                    continue

                if enable_tables and self._HTML_TABLE_START_RE.search(line or ""):
                    flush_text()
                    start = i
                    start_match = self._HTML_TABLE_START_RE.search(line or "")
                    assert start_match is not None
                    start_pos = start_match.start()
                    prefix = (line[:start_pos] or "").strip()
                    if prefix:
                        segments.append(_Segment(kind="text", content=prefix))

                    table_lines: List[str] = []
                    remainder = line[start_pos:]
                    same_line_end = self._HTML_TABLE_END_RE.search(remainder or "")
                    if same_line_end:
                        end_pos_abs = start_pos + same_line_end.end()
                        table_lines.append(line[start_pos:end_pos_abs])
                        suffix = (line[end_pos_abs:] or "").strip()
                        if suffix:
                            lines[i] = suffix
                        else:
                            i += 1
                        table_content = "\n".join(table_lines).strip("\n")
                        caption = self._infer_block_caption(lines, start)
                        segments.append(
                            _Segment(kind="table", content=table_content, caption=caption, info={"table_format": "html"})
                        )
                        continue

                    table_lines.append(remainder)
                    i += 1
                    while i < span_end:
                        candidate = lines[i]
                        end_match = self._HTML_TABLE_END_RE.search(candidate or "")
                        if not end_match:
                            table_lines.append(candidate)
                            i += 1
                            continue

                        end_pos = end_match.end()
                        table_lines.append(candidate[:end_pos])
                        suffix = (candidate[end_pos:] or "").strip()
                        if suffix:
                            lines[i] = suffix
                        else:
                            i += 1
                        break

                    table_content = "\n".join(table_lines).strip("\n")
                    caption = self._infer_block_caption(lines, start)
                    segments.append(_Segment(kind="table", content=table_content, caption=caption, info={"table_format": "html"}))
                    continue

                if enable_images and line_contains_image(line):
                    flush_text()
                    start = i
                    i += 1
                    while i < span_end and line_contains_image(lines[i]):
                        i += 1
                    image_content = "\n".join(lines[start:i]).strip("\n")
                    caption = self._infer_block_caption(lines, start)
                    segments.append(_Segment(kind="image", content=image_content, caption=caption))
                    continue

                if enable_lists and self._is_list_start(line):
                    flush_text()
                    start = i
                    i += 1
                    while i < span_end and lines[i].strip():
                        if self._is_list_line(lines[i]):
                            i += 1
                            continue
                        break
                    list_content = "\n".join(lines[start:i]).strip("\n")
                    caption = self._infer_block_caption(lines, start)
                    segments.append(_Segment(kind="list", content=list_content, caption=caption))
                    continue

                if enable_blockquotes and self._BLOCKQUOTE_RE.match(line or ""):
                    flush_text()
                    start = i
                    i += 1
                    while i < span_end and self._BLOCKQUOTE_RE.match(lines[i] or ""):
                        i += 1
                    quote_content = "\n".join(lines[start:i]).strip("\n")
                    caption = self._infer_block_caption(lines, start)
                    segments.append(_Segment(kind="blockquote", content=quote_content, caption=caption))
                    continue

                buffer.append(line)
                i += 1

        flush_text()
        return segments

    @staticmethod
    def _infer_block_caption(lines: List[str], block_start_index: int) -> str:
        j = block_start_index - 1
        while j >= 0:
            candidate = (lines[j] or "").strip()
            if not candidate:
                j -= 1
                continue
            if "|" in candidate:
                return ""
            lower = candidate.lower()
            if any(tag in lower for tag in ("<table", "</table", "<tr", "</tr", "<td", "</td", "<th", "</th", "<tbody", "</tbody", "<thead", "</thead")):
                return ""
            if is_fence_line(candidate):
                return ""
            if SemanticUnitChunker._BLOCKQUOTE_RE.match(candidate):
                return ""
            if SemanticUnitChunker._is_list_start(candidate):
                return ""
            if SemanticUnitChunker._is_math_block_start(candidate) or SemanticUnitChunker._is_math_block_end(candidate):
                return ""
            return candidate
        return ""

    @staticmethod
    def _is_math_block_start(line: str) -> bool:
        return bool(SemanticUnitChunker._MATH_BLOCK_START_RE.match(line or ""))

    @staticmethod
    def _is_math_block_end(line: str) -> bool:
        return bool(SemanticUnitChunker._MATH_BLOCK_END_RE.match(line or ""))

    @staticmethod
    def _is_list_start(line: str) -> bool:
        if not isinstance(line, str):
            return False
        stripped = line.strip()
        if not stripped:
            return False
        return bool(
            SemanticUnitChunker._TASK_ITEM_RE.match(line)
            or SemanticUnitChunker._LIST_ITEM_RE.match(line)
            or SemanticUnitChunker._ORDERED_ITEM_RE.match(line)
        )

    @staticmethod
    def _is_list_line(line: str) -> bool:
        if not isinstance(line, str) or not line.strip():
            return False
        if SemanticUnitChunker._is_list_start(line):
            return True
        return bool(re.match(r"^\s+\S+", line))

    def _split_fenced_code(self, content: str) -> tuple[str, str, List[str]]:
        return parse_fenced_code_block(content)

    @staticmethod
    def _fence_wrap(*, fence: str, language: str, body_lines: List[str]) -> str:
        fence = (fence or "```").strip()
        language = (language or "").strip()
        header = f"{fence}{language}".rstrip()
        footer = fence
        body = "\n".join(body_lines or []).rstrip("\n")
        if body:
            return "\n".join([header, body, footer])
        return "\n".join([header, footer])

    def _slice_lines(
        self,
        body_lines: List[str],
        *,
        slice_max_tokens: int,
        overlap_lines: int,
        prefix_caption: str,
        fence: str,
        language: str,
    ) -> List[tuple[int, int, str]]:
        overlap_lines = max(int(overlap_lines), 0)
        slice_max_tokens = max(int(slice_max_tokens), 1)

        slices: List[tuple[int, int, str]] = []
        start = 0
        while start < len(body_lines):
            end = start
            chosen: List[str] = []
            while end < len(body_lines):
                candidate_lines = chosen + [body_lines[end]]
                candidate_body = "\n".join(candidate_lines)
                candidate_fenced = self._fence_wrap(fence=fence, language=language, body_lines=candidate_body.splitlines())
                candidate = self._compose_with_caption(prefix_caption, candidate_fenced)
                if chosen and self._estimate_tokens(candidate) > slice_max_tokens:
                    break
                chosen.append(body_lines[end])
                end += 1

                if self._estimate_tokens(candidate) >= slice_max_tokens:
                    break

            if not chosen:
                chosen.append(body_lines[start])
                end = start + 1

            line_start = start + 1
            line_end = end
            slices.append((line_start, line_end, "\n".join(chosen)))

            if end >= len(body_lines):
                break
            start = max(end - overlap_lines, start + 1)

        return slices

    def _parse_markdown_list_items(self, content: str) -> tuple[List[str], str]:
        lines = [line.rstrip() for line in (content or "").splitlines() if line.strip()]
        if not lines:
            return [], "unknown"

        first = lines[0]
        if self._TASK_ITEM_RE.match(first):
            list_type = "task"
        elif self._ORDERED_ITEM_RE.match(first):
            list_type = "ordered"
        else:
            list_type = "unordered"

        items: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            if self._is_list_start(line):
                if current:
                    items.append(current)
                current = [line]
                continue
            if current and re.match(r"^\s+\S+", line):
                current.append(line)
                continue
            if current:
                current.append(line)
        if current:
            items.append(current)

        rendered = ["\n".join(item).strip() for item in items if "\n".join(item).strip()]
        return rendered, list_type

    def _slice_list_items(
        self,
        *,
        caption: str,
        items: List[str],
        slice_max_tokens: int,
        overlap_items: int,
    ) -> List[tuple[int, int, str]]:
        overlap_items = max(int(overlap_items), 0)
        slice_max_tokens = max(int(slice_max_tokens), 1)

        slices: List[tuple[int, int, str]] = []
        start = 0
        while start < len(items):
            end = start
            chosen: List[str] = []
            while end < len(items):
                candidate_items = chosen + [items[end]]
                candidate_body = "\n".join(candidate_items).strip()
                candidate = self._compose_with_caption(caption, candidate_body)
                if chosen and self._estimate_tokens(candidate) > slice_max_tokens:
                    break
                chosen.append(items[end])
                end += 1

                if self._estimate_tokens(candidate) >= slice_max_tokens:
                    break

            if not chosen:
                chosen.append(items[start])
                end = start + 1

            item_start = start + 1
            item_end = end
            slices.append((item_start, item_end, "\n".join(chosen).strip()))

            if end >= len(items):
                break
            start = max(end - overlap_items, start + 1)

        return slices

    def _slice_table_rows(
        self,
        *,
        caption: str,
        header: str,
        separator: str,
        rows: List[str],
        slice_max_tokens: int,
        overlap_rows: int,
    ) -> List[tuple[int, int, str]]:
        overlap_rows = max(int(overlap_rows), 0)
        slice_max_tokens = max(int(slice_max_tokens), 1)

        slices: List[tuple[int, int, str]] = []
        start = 0
        while start < len(rows):
            end = start
            chosen: List[str] = []
            while end < len(rows):
                candidate_rows = chosen + [rows[end]]
                candidate_body = "\n".join(candidate_rows)
                candidate_table = self._compose_with_caption(
                    caption,
                    "\n".join([line for line in (header, separator, candidate_body) if line]).strip(),
                )
                if chosen and self._estimate_tokens(candidate_table) > slice_max_tokens:
                    break
                chosen.append(rows[end])
                end += 1

                if self._estimate_tokens(candidate_table) >= slice_max_tokens:
                    break

            if not chosen:
                chosen.append(rows[start])
                end = start + 1

            row_start = start + 1
            row_end = end
            slices.append((row_start, row_end, "\n".join(chosen)))

            if end >= len(rows):
                break
            start = max(end - overlap_rows, start + 1)

        return slices
