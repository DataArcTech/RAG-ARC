from pathlib import Path
from urllib.parse import quote


def build_attachment_content_disposition(filename: str, *, fallback_basename: str = "download") -> str:
    """
    Build a latin-1-safe Content-Disposition header for arbitrary UTF-8 filenames.

    Notes:
    - ASGI/Starlette headers are encoded with latin-1; non-ascii filenames must be
      carried via RFC 5987 `filename*` (percent-encoded UTF-8).
    - Always includes a conservative ASCII `filename="..."` fallback for compatibility.
    """
    name = Path(str(filename or "")).name or fallback_basename
    name = name.replace('"', "")

    fallback = "".join(
        ch if (32 <= ord(ch) < 127 and ch not in {'"', "\\", ";"}) else "_"
        for ch in name
    ).strip()
    if not fallback:
        fallback = fallback_basename

    quoted = quote(name, safe="")
    return f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quoted}'

