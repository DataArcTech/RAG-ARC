import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.file_management.atomic_units.markdown_image import extract_image_urls
from core.utils.path_guard import safe_leaf_name

logger = logging.getLogger(__name__)


def extract_image_urls_from_markdown(markdown: str) -> List[str]:
    urls: List[str] = []
    for line in (markdown or "").splitlines():
        urls.extend(extract_image_urls(line))
    return [u for u in urls if u]


def resolve_local_image_paths_for_mineru(
    *,
    filename: str,
    image_urls: Sequence[str],
    source_file_id: str | None = None,
    parser_output_dir: str | None = None,
) -> List[Path]:
    """Resolve MinerU relative image URLs (images/...) into local file paths under PARSER_OUTPUT_DIR."""
    base = (parser_output_dir or os.getenv("PARSER_OUTPUT_DIR") or "./data/parsed_files").strip() or "./data/parsed_files"
    base_dir = Path(base).expanduser().resolve()
    stem = Path(filename).stem or "document"
    stem_dir = (base_dir / "mineru" / safe_leaf_name(stem, default="document")).resolve()
    file_dir = None
    if source_file_id:
        file_dir = (base_dir / "mineru" / safe_leaf_name(source_file_id, default=stem)).resolve()
    doc_dirs = [d for d in [file_dir, stem_dir] if isinstance(d, Path)]

    resolved: List[Path] = []
    for url in image_urls:
        token = str(url or "").strip()
        if not token:
            continue
        candidate = Path(token)
        if candidate.is_absolute() and candidate.exists():
            resolved.append(candidate)
            continue
        rel = token.lstrip("/").lstrip("\\")
        for doc_dir in doc_dirs:
            path = (doc_dir / rel).resolve()
            try:
                path.relative_to(doc_dir)
            except Exception:
                continue
            if path.exists() and path.is_file():
                resolved.append(path)
                break
    return resolved


def collect_image_paths_from_chunk_payloads(
    chunks: Iterable[Any],
    *,
    max_images: Optional[int] = None,
) -> List[Path]:
    """
    Collect local image file paths from chunk-like payloads.

    Supports:
    - `Chunk` objects with `.content` and `.metadata`
    - dicts with `content` and `metadata`
    """
    seen: set[str] = set()
    images: List[Path] = []

    for chunk in chunks:
        content = ""
        metadata: Dict[str, Any] = {}

        if isinstance(chunk, dict):
            content = str(chunk.get("content") or "")
            if isinstance(chunk.get("metadata"), dict):
                metadata = dict(chunk["metadata"])
        else:
            content = str(getattr(chunk, "content", "") or "")
            raw_meta = getattr(chunk, "metadata", None)
            if isinstance(raw_meta, dict):
                metadata = dict(raw_meta)

        if not content.strip():
            continue

        urls = extract_image_urls_from_markdown(content)
        if not urls:
            continue

        filename = str(metadata.get("filename") or metadata.get("source_file_name") or "").strip()
        if not filename:
            # DeepSearch evidence provenance stores chunk metadata under `chunk_metadata`.
            filename = str(metadata.get("chunk_metadata", {}).get("filename") or "").strip() if isinstance(metadata.get("chunk_metadata"), dict) else ""

        if not filename:
            continue

        source_file_id = str(metadata.get("source_file_id") or "").strip() or None
        if not source_file_id and isinstance(metadata.get("chunk_metadata"), dict):
            source_file_id = str(metadata["chunk_metadata"].get("source_file_id") or "").strip() or None

        paths = resolve_local_image_paths_for_mineru(
            filename=filename,
            image_urls=urls,
            source_file_id=source_file_id,
        )
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            images.append(path)
            if max_images is not None and len(images) >= max_images:
                return images

    return images


def collect_image_paths_from_deepsearch_evidences(
    evidences: Sequence[Dict[str, Any]],
    *,
    max_images: Optional[int] = None,
) -> List[Path]:
    """
    Collect local MinerU image paths from DeepSearch evidence payloads.

    Evidence payload shape (dict):
      - content: str
      - provenance.metadata.chunk_metadata.filename: str
    """
    wrapped: List[Dict[str, Any]] = []
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        content = str(evidence.get("content") or "")
        provenance = evidence.get("provenance")
        chunk_meta: Dict[str, Any] = {}
        if isinstance(provenance, dict):
            meta = provenance.get("metadata")
            if isinstance(meta, dict):
                cm = meta.get("chunk_metadata")
                if isinstance(cm, dict):
                    chunk_meta = dict(cm)
        wrapped.append({"content": content, "metadata": {"chunk_metadata": chunk_meta}})
    return collect_image_paths_from_chunk_payloads(wrapped, max_images=max_images)
