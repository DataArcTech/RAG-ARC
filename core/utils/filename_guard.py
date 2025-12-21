from pathlib import Path, PurePosixPath


def project_root_dir() -> Path:
    """Return repository root directory as inferred from this module location."""

    # core/utils/filename_guard.py -> core/utils -> core -> repo root
    return Path(__file__).resolve().parents[2]


def normalize_project_filename(filename: str, *, root: Path | None = None) -> str:
    """
    Normalize a user-provided filename/path into a safe, repo-root-prefixed relative path.

    Always returns: "<repo_root_dir_name>/<relative_path_or_basename>".

    Notes:
    - Any absolute path, Windows drive prefix, or traversal segments are stripped/sanitized.
    - If filename is empty after sanitization, falls back to "upload".
    """

    root = root or project_root_dir()
    root_name = root.name

    raw = str(filename or "").strip().replace("\\", "/")
    # Drop URL query/fragment-like suffixes if any (defensive).
    raw = raw.split("?", 1)[0].split("#", 1)[0].strip()
    raw = raw.lstrip("/")

    # Strip Windows drive prefix (e.g., C:/path/file.txt) if present.
    if len(raw) >= 2 and raw[1] == ":":
        raw = raw[2:].lstrip("/")

    parts = [part for part in PurePosixPath(raw).parts if part not in {"", ".", ".."}]
    if parts and parts[0] == root_name:
        parts = parts[1:]

    if not parts:
        return f"{root_name}/upload"

    normalized = str(PurePosixPath(*parts))
    return f"{root_name}/{normalized}"


def project_relative_path(path: Path, *, root: Path | None = None) -> str:
    """
    Return "<repo_root_dir_name>/<path_relative_to_repo_root>".

    Raises ValueError when the path is outside the repo root.
    """

    root = root or project_root_dir()
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        rel = resolved.relative_to(root_resolved)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"path must be under project root: {root_resolved}") from exc
    rel_str = rel.as_posix().lstrip("/")
    return f"{root.name}/{rel_str}" if rel_str else f"{root.name}/"

