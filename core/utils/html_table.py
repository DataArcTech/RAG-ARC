from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import List


@dataclass
class _RowAccumulator:
    cells: List[str] = field(default_factory=list)


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._rows: List[_RowAccumulator] = []

    @property
    def rows(self) -> List[List[str]]:
        return [row.cells for row in self._rows if row.cells]

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        token = (tag or "").lower()
        if token == "table":
            self._in_table = True
            return
        if not self._in_table:
            return
        if token == "tr":
            self._in_row = True
            self._rows.append(_RowAccumulator())
            return
        if token in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell_parts = []
            return
        if token == "br" and self._in_cell:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        token = (tag or "").lower()
        if token == "table":
            self._in_table = False
            self._in_row = False
            self._in_cell = False
            self._cell_parts = []
            return
        if not self._in_table:
            return
        if token in {"td", "th"} and self._in_cell:
            text = unescape("".join(self._cell_parts))
            normalized = " ".join(text.replace("\r", "\n").split())
            if self._rows:
                self._rows[-1].cells.append(normalized)
            self._in_cell = False
            self._cell_parts = []
            return
        if token == "tr":
            self._in_row = False
            self._in_cell = False
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_table and self._in_cell and data:
            self._cell_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._in_table and self._in_cell and name:
            self._cell_parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._in_table and self._in_cell and name:
            self._cell_parts.append(f"&#{name};")


def extract_html_table_rows(table_html: str) -> List[List[str]]:
    """Extract row/cell text from a <table>...</table> HTML fragment."""
    parser = _HTMLTableParser()
    parser.feed(table_html or "")
    parser.close()
    return parser.rows


def render_pipe_table(rows: List[List[str]]) -> str:
    """Render rows as a markdown pipe table (best-effort)."""
    if not rows:
        return ""
    width = max((len(r) for r in rows if r), default=0)
    if width <= 0:
        return ""

    def _cell(value: str) -> str:
        text = str(value or "")
        text = text.replace("|", "\\|").strip()
        return text

    normalized: List[List[str]] = []
    for row in rows:
        padded = list(row) + [""] * max(0, width - len(row))
        normalized.append([_cell(c) for c in padded[:width]])

    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []
    lines: List[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines).strip()

