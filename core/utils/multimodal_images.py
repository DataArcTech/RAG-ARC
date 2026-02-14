import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.file_management.atomic_units.markdown_image import extract_image_urls
from core.utils.path_guard import safe_leaf_name
from core.utils.path_guard import require_writable_dir

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
    backend = str(os.getenv("IO_STORE_BACKEND", "localdb") or "localdb").strip().lower()
    if backend == "minio":
        raise RuntimeError("MinerU image path resolution to local filesystem is disabled when IO_STORE_BACKEND=minio")
    base = (parser_output_dir or os.getenv("PARSER_OUTPUT_DIR") or "io://parsed_files").strip() or "io://parsed_files"
    if not base.startswith("io://"):
        raise ValueError("PARSER_OUTPUT_DIR must be an io:// virtual path")
    base_dir = Path(require_writable_dir(base)).expanduser().resolve()
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
            # Some pipelines store image URLs only in metadata (e.g. semantic_unit_type="image").
            # Keep going so we can still resolve images via `image_urls`.
            content = ""

        urls = extract_image_urls_from_markdown(content)
        # Fallback: metadata-driven image URLs (e.g. semantic_unit_chunker image segments).
        meta_urls = metadata.get("image_urls")
        if isinstance(meta_urls, list):
            urls.extend([str(u).strip() for u in meta_urls if isinstance(u, str) and str(u).strip()])
        urls = [u for u in urls if u]
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
        meta: Dict[str, Any] = {}
        if isinstance(provenance, dict):
            raw_meta = provenance.get("metadata")
            if isinstance(raw_meta, dict):
                # Two supported shapes:
                # 1) provenance.metadata.chunk_metadata (older/explicit)
                # 2) provenance.metadata directly contains chunk metadata (current DeepSearch composer)
                cm = raw_meta.get("chunk_metadata")
                if isinstance(cm, dict) and cm:
                    meta.update(dict(cm))
                else:
                    meta.update(dict(raw_meta))
            # Back-compat: some evidence records keep filename at provenance.file_name.
            if "filename" not in meta:
                token = provenance.get("file_name")
                if isinstance(token, str) and token.strip():
                    meta["filename"] = token.strip()
        wrapped.append({"content": content, "metadata": meta})
    return collect_image_paths_from_chunk_payloads(wrapped, max_images=max_images)
