from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class HeadingSignal:
    title: str
    markdown_level: Optional[int]
    numbering_level: Optional[int]
    text_level: Optional[int]
    page_idx: Optional[int]
    line_index: int
    char_start: int
    char_end: int
    resolved_level: int
    level_source: str


@dataclass
class SectionNode:
    section_id: str
    file_id: str
    title: str
    path: str
    level: int
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    summary: Optional[str] = None
    heading_start: Optional[int] = None
    heading_end: Optional[int] = None
    level_source: Optional[str] = None


@dataclass
class DocDescription:
    file_id: str
    title: str
    description: str


@dataclass
class DocProfile:
    """Structured identifiers used for cross-document disambiguation in doc routing."""

    company: Optional[str] = None
    product: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    doc_type: Optional[str] = None
    language: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
