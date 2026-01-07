import re
from typing import List, Tuple

from core.file_management.atomic_units.fenced_code import split_fenced_code_blocks_with_line_spans
from core.file_management.atomic_units.markdown_image import line_contains_image
from core.file_management.atomic_units.markdown_table import is_markdown_table_start


_HTML_TABLE_START_RE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)


def split_markdown_atomic_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Split markdown into atomic blocks:
    - fenced code blocks (```...```)
    - markdown pipe tables
    - HTML tables (<table>...</table>)
    - image lines containing ![](...) or <img ...>
    - everything else as text

    Returned kinds: "code" | "table" | "image" | "text"
    """
    lines = (text or "").splitlines()
    segments: List[Tuple[str, str]] = []
    buffer: List[str] = []

    def flush_text() -> None:
        if not buffer:
            return
        content = "\n".join(buffer).strip("\n")
        if content.strip():
            segments.append(("text", content))
        buffer.clear()

    for kind, span_start, span_end in split_fenced_code_blocks_with_line_spans(text or ""):
        if kind == "code":
            flush_text()
            code = "\n".join(lines[span_start:span_end]).strip("\n")
            if code.strip():
                segments.append(("code", code))
            continue

        i = span_start
        while i < span_end:
            line = lines[i]

            if i + 1 < span_end and is_markdown_table_start(lines, i):
                flush_text()
                start = i
                i += 2
                while i < span_end and lines[i].strip() and ("|" in lines[i]):
                    i += 1
                table = "\n".join(lines[start:i]).strip("\n")
                if table.strip():
                    segments.append(("table", table))
                continue

            if _HTML_TABLE_START_RE.search(line or ""):
                flush_text()
                start = i
                start_match = _HTML_TABLE_START_RE.search(line or "")
                assert start_match is not None
                start_pos = start_match.start()

                # Keep any leading text before <table> as normal text.
                prefix = (line[:start_pos] or "").strip()
                if prefix:
                    segments.append(("text", prefix))

                table_lines: List[str] = []
                remainder = line[start_pos:]
                same_line_end = _HTML_TABLE_END_RE.search(remainder or "")
                if same_line_end:
                    end_pos_abs = start_pos + same_line_end.end()
                    table_lines.append(line[start_pos:end_pos_abs])
                    suffix = (line[end_pos_abs:] or "").strip()
                    if suffix:
                        lines[i] = suffix
                    else:
                        i += 1
                    table = "\n".join(table_lines).strip("\n")
                    if table.strip():
                        segments.append(("table", table))
                    continue

                table_lines.append(remainder)

                i += 1
                while i < span_end:
                    candidate = lines[i]
                    end_match = _HTML_TABLE_END_RE.search(candidate or "")
                    if not end_match:
                        table_lines.append(candidate)
                        i += 1
                        continue

                    end_pos = end_match.end()
                    table_lines.append(candidate[:end_pos])

                    # If there is trailing content after </table>, keep it for re-processing
                    # (e.g., MinerU sometimes emits </table> immediately followed by ![](...)).
                    suffix = (candidate[end_pos:] or "").strip()
                    if suffix:
                        lines[i] = suffix
                    else:
                        i += 1
                    break

                table = "\n".join(table_lines).strip("\n")
                if table.strip():
                    segments.append(("table", table))
                continue

            if line_contains_image(line):
                flush_text()
                start = i
                i += 1
                while i < span_end and line_contains_image(lines[i]):
                    i += 1
                images = "\n".join(lines[start:i]).strip("\n")
                if images.strip():
                    segments.append(("image", images))
                continue

            buffer.append(line)
            i += 1

    flush_text()
    return segments
