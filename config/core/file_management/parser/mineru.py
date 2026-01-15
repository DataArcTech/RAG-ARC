import os
from typing import Literal, Optional

from pydantic import Field

from framework.config import AbstractConfig
from core.file_management.parser.mineru import MinerUParser


def _default_mineru_server_url() -> str:
    return str(os.getenv("MINERU_SERVER_URL", "") or "").strip()

def _default_reuse_cache() -> bool:
    # Default to true so re-indexing can reuse existing MinerU markdown artifacts without hitting the service.
    return str(os.getenv("MINERU_REUSE_CACHE", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}

def _default_timeout_s() -> int:
    raw = str(os.getenv("MINERU_TIMEOUT_S", "") or "").strip()
    if not raw:
        return 900
    try:
        return max(1, int(raw))
    except Exception:
        return 900


def _default_start_page() -> int:
    raw = str(os.getenv("MINERU_START_PAGE", "") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except Exception:
        return 0


def _default_end_page() -> Optional[int]:
    raw = str(os.getenv("MINERU_END_PAGE", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
        return value if value >= 0 else None
    except Exception:
        return None


class MinerUParserConfig(AbstractConfig):
    type: Literal["mineru_parser"] = "mineru_parser"

    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory for MinerU artifacts. When using ParserCombinator this is set automatically.",
    )

    server_url: str = Field(default_factory=_default_mineru_server_url, description="MinerU server base URL.")
    timeout_s: int = Field(default_factory=_default_timeout_s, description="HTTP timeout (seconds) for remote MinerU parsing.")
    reuse_cache: bool = Field(
        default_factory=_default_reuse_cache,
        description=(
            "When true, reuse existing MinerU markdown artifacts under output_dir/<source_file_id>/ when present, "
            "skipping remote MinerU calls. Controlled by MINERU_REUSE_CACHE=1/0."
        ),
    )

    # MinerU parse defaults (can be overridden per-call via kwargs)
    backend: str = Field(default="vlm-transformers")
    parse_method: str = Field(default="auto")
    lang: str = Field(default="ch")
    formula_enable: bool = Field(default=True)
    table_enable: bool = Field(default=True)
    start_page: int = Field(default_factory=_default_start_page)
    end_page: Optional[int] = Field(default_factory=_default_end_page)
    output_format: str = Field(default="mm_md")

    def build(self) -> MinerUParser:
        if not str(self.server_url or "").strip():
            raise ValueError("MinerUParser requires MINERU_SERVER_URL (or set server_url in config).")
        return MinerUParser(self)
