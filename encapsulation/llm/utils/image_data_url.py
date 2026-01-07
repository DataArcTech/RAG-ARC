import base64
import mimetypes
from pathlib import Path


def file_to_data_url(path: str | Path) -> str:
    """Convert a local image file into a data URL usable by OpenAI-compatible vision models."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"

