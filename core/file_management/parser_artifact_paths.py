from pathlib import Path
from typing import Optional, Tuple


def _norm_posix(path: Path) -> str:
    # Persist as POSIX for cross-platform stability (API + DB).
    return path.as_posix()


def relativize_under_base(path: str | None, *, base_dir: str | None) -> Optional[str]:
    """Return `path` as a relative POSIX path when it sits under `base_dir`.

    If `path` is empty or not under `base_dir`, return the original string (stripped).
    This keeps metadata useful even when operators change output roots.
    """
    token = str(path or "").strip()
    if not token:
        return None

    base = str(base_dir or "").strip()
    if not base:
        return token

    try:
        p = Path(token).expanduser().resolve()
        b = Path(base).expanduser().resolve()
        rel = p.relative_to(b)
        return _norm_posix(rel)
    except Exception:
        return token


def resolve_under_base(path: str | None, *, base_dir: str | None) -> Optional[str]:
    """Resolve a stored artifact path back to an absolute path.

    - Absolute stored paths are returned as-is.
    - Relative stored paths are joined under `base_dir` when provided.
    """
    token = str(path or "").strip()
    if not token:
        return None
    p = Path(token).expanduser()
    if p.is_absolute():
        return str(p)
    base = str(base_dir or "").strip()
    if not base:
        return str(p)
    return str((Path(base).expanduser().resolve() / p).resolve())


def extract_md_path_and_output_dir(parse_result: object) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of parser artifact paths from a single parse result dict."""
    if not isinstance(parse_result, dict):
        return None, None
    md_path = parse_result.get("md_content_path")
    meta = parse_result.get("metadata") if isinstance(parse_result.get("metadata"), dict) else {}
    output_dir = meta.get("output_dir")
    md_path = str(md_path).strip() if isinstance(md_path, str) and md_path.strip() else None
    output_dir = str(output_dir).strip() if isinstance(output_dir, str) and output_dir.strip() else None
    return md_path, output_dir

