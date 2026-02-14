import os
from pathlib import Path
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


def _default_poll_interval_s() -> int:
    raw = str(os.getenv("MINERU_POLL_INTERVAL_S", "") or "").strip()
    if not raw:
        return 5
    try:
        return max(1, int(raw))
    except Exception:
        return 5


def _default_poll_timeout_s() -> int:
    raw = str(os.getenv("MINERU_POLL_TIMEOUT_S", "") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except Exception:
        return 0


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


def _default_backend() -> str:
    raw = str(os.getenv("MINERU_BACKEND", "") or "").strip()
    return raw or "vlm-transformers"

def _default_http_max_retries() -> int:
    raw = str(os.getenv("MINERU_HTTP_MAX_RETRIES", "") or "").strip()
    if not raw:
        return 3
    try:
        return max(0, int(raw))
    except Exception:
        return 3

def _default_http_retry_backoff_s() -> float:
    raw = str(os.getenv("MINERU_HTTP_RETRY_BACKOFF_S", "") or "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except Exception:
        return 1.0

def _default_http_retry_max_backoff_s() -> float:
    raw = str(os.getenv("MINERU_HTTP_RETRY_MAX_BACKOFF_S", "") or "").strip()
    if not raw:
        return 8.0
    try:
        return max(0.0, float(raw))
    except Exception:
        return 8.0

def _default_shared_cache_enabled() -> bool:
    return str(os.getenv("MINERU_SHARED_CACHE_ENABLED", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}

def _default_shared_cache_dir() -> str:
    raw = str(os.getenv("MINERU_SHARED_CACHE_DIR", "") or "").strip()
    if raw:
        if not raw.startswith("io://"):
            raise ValueError("MINERU_SHARED_CACHE_DIR must be an io:// virtual path")
        return raw
    base = str(os.getenv("PARSER_OUTPUT_DIR", "io://parsed_files") or "io://parsed_files").strip()
    if not base.startswith("io://"):
        raise ValueError("PARSER_OUTPUT_DIR must be an io:// virtual path")
    base = base.rstrip("/")
    return f"{base}/mineru/_shared"

def _default_shared_cache_mode() -> str:
    raw = str(os.getenv("MINERU_SHARED_CACHE_MODE", "") or "").strip().lower()
    return raw or "symlink"


class MinerUParserConfig(AbstractConfig):
    type: Literal["mineru_parser"] = "mineru_parser"

    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory for MinerU artifacts. When using ParserCombinator this is set automatically.",
    )

    server_url: str = Field(default_factory=_default_mineru_server_url, description="MinerU server base URL.")
    timeout_s: int = Field(default_factory=_default_timeout_s, description="HTTP timeout (seconds) for remote MinerU parsing.")
    poll_interval_s: int = Field(
        default_factory=_default_poll_interval_s,
        description="Polling interval (seconds) for MinerU async parse status.",
    )
    poll_timeout_s: int = Field(
        default_factory=_default_poll_timeout_s,
        description="Max seconds to wait for MinerU parse; <=0 means no limit.",
    )
    http_max_retries: int = Field(
        default_factory=_default_http_max_retries,
        description="Max retry attempts for transient MinerU HTTP errors (0 disables retries).",
    )
    http_retry_backoff_s: float = Field(
        default_factory=_default_http_retry_backoff_s,
        description="Base backoff seconds for MinerU HTTP retries.",
    )
    http_retry_max_backoff_s: float = Field(
        default_factory=_default_http_retry_max_backoff_s,
        description="Max backoff seconds for MinerU HTTP retries.",
    )
    reuse_cache: bool = Field(
        default_factory=_default_reuse_cache,
        description=(
            "When true, reuse existing MinerU markdown artifacts under output_dir/<source_file_id>/ when present, "
            "skipping remote MinerU calls. Controlled by MINERU_REUSE_CACHE=1/0."
        ),
    )
    shared_cache_enabled: bool = Field(
        default_factory=_default_shared_cache_enabled,
        description=(
            "When true, reuse MinerU artifacts across uploads if the file bytes are identical (sha256 match), "
            "even across different owners/tenants. Controlled by MINERU_SHARED_CACHE_ENABLED=1/0."
        ),
    )
    shared_cache_dir: str = Field(
        default_factory=_default_shared_cache_dir,
        description=(
            "Directory for shared MinerU artifacts cache. Defaults to `${PARSER_OUTPUT_DIR}/mineru/_shared`. "
            "Controlled by MINERU_SHARED_CACHE_DIR."
        ),
    )
    shared_cache_mode: str = Field(
        default_factory=_default_shared_cache_mode,
        description="How to materialize shared cache into per-file dirs: `symlink` (preferred) or `copy`.",
    )

    # MinerU parse defaults (can be overridden per-call via kwargs)
    backend: str = Field(default_factory=_default_backend)
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
