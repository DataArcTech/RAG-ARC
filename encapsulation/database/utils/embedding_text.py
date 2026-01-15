from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def normalize_filename_for_embedding(filename: object, *, root: str | None) -> str:
    """
    Normalize a filename/path string for embedding.

    - Uses forward slashes for stability across OSes.
    - If `root` is provided and present in the path, trims to start at that root token.
      (E.g. "/x/y/RAG-ARC/local/user_files/a.pdf" -> "RAG-ARC/local/user_files/a.pdf")
    """
    value = str(filename or "").strip()
    if not value:
        return ""
    value = value.replace("\\", "/")
    if root:
        root_token = str(root).strip().replace("\\", "/")
        if root_token:
            idx = value.find(root_token)
            if idx >= 0:
                value = value[idx:]
    return value


def build_embedding_text(
    *,
    base_text: str,
    metadata: Mapping[str, Any],
    prefix_keys: Sequence[str],
    filename_root: str | None,
    separator: str = "\n",
) -> str:
    """
    Build the final embedding input text by optionally prefixing metadata fields.

    This is intentionally conservative:
    - If no prefix fields are available, returns base_text unchanged.
    - Prefix fields are appended in the order provided by `prefix_keys`.
    """
    base = str(base_text or "").strip()
    if not base:
        return ""

    parts: list[str] = []
    for key in prefix_keys or []:
        token = str(key or "").strip()
        if not token:
            continue
        raw = metadata.get(token)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if token == "filename":
            text = normalize_filename_for_embedding(text, root=filename_root)
        parts.append(text)

    if not parts:
        return base

    sep = str(separator or "\n")
    return (sep.join(parts) + sep + base).strip()


def build_prefix_keys(value: object) -> list[str]:
    """
    Coerce a config value into a list of prefix keys.

    Accepts:
    - list/tuple of strings
    - comma-separated string
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out
    raw = str(value).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]

