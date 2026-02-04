import logging
import os
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from core.file_management.parser.base import AbstractParser
from core.utils.path_guard import require_writable_dir, safe_leaf_name
from framework.thread_pool import get_thread_pool

from encapsulation.remote_services.mineru_service_client import MinerUServiceClient

if TYPE_CHECKING:
    from config.core.file_management.parser.mineru import MinerUParserConfig

logger = logging.getLogger(__name__)


class MinerUParser(AbstractParser):
    """
    Remote MinerU parser via HTTP service.

    Produces a local Markdown + assets mirror under config.output_dir/<doc_name>/...
    so downstream RAG-ARC pipeline reads markdown from a stable local path.
    """

    def __init__(self, config: "MinerUParserConfig"):
        super().__init__(config)
        server_url = str(getattr(self.config, "server_url", "") or "").strip()
        timeout_s = int(getattr(self.config, "timeout_s", 900) or 900)
        poll_interval_s = int(getattr(self.config, "poll_interval_s"))
        poll_timeout_s = int(getattr(self.config, "poll_timeout_s"))
        http_max_retries = int(getattr(self.config, "http_max_retries", 0) or 0)
        http_retry_backoff_s = float(getattr(self.config, "http_retry_backoff_s", 1.0) or 1.0)
        http_retry_max_backoff_s = float(getattr(self.config, "http_retry_max_backoff_s", 8.0) or 8.0)
        self.client = MinerUServiceClient(
            base_url=server_url,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            poll_timeout_s=poll_timeout_s,
            http_max_retries=http_max_retries,
            http_retry_backoff_s=http_retry_backoff_s,
            http_retry_max_backoff_s=http_retry_max_backoff_s,
        )

    def get_supported_extensions(self) -> List[str]:
        return [".pdf", ".jpg", ".jpeg", ".png"]

    async def parse_file(self, file_data: bytes, filename: str, **kwargs: Any) -> List[Dict[str, Any]]:
        base_filename = Path(filename).stem or "document"
        file_ext = Path(filename).suffix.lower()
        if file_ext not in set(self.get_supported_extensions()):
            supported = ", ".join(self.get_supported_extensions())
            raise ValueError(f"Unsupported file type '{file_ext}'. Supported types: {supported}")

        output_dir = getattr(self.config, "output_dir", None)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("MinerUParser requires config.output_dir (no implicit env defaults).")
        output_dir = require_writable_dir(output_dir)
        source_file_id = kwargs.get("source_file_id") or kwargs.get("file_id")
        doc_key = safe_leaf_name(str(source_file_id or ""), default=base_filename)
        doc_dir = Path(output_dir) / doc_key
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Cache reuse: if a previous MinerU run already produced local markdown for this file_id,
        # allow re-indexing to reuse it without calling the remote MinerU service again.
        reuse_cache = bool(kwargs.get("reuse_cache", getattr(self.config, "reuse_cache", False)))
        md_local_path = doc_dir / f"{base_filename}.md"
        if reuse_cache and md_local_path.exists():
            md_text = md_local_path.read_text(encoding="utf-8", errors="ignore")
            if md_text.strip():
                logger.info("Reusing cached MinerU markdown: %s", md_local_path)
                return [
                    {
                        "md_content_path": str(md_local_path),
                        "text": md_text,
                        "metadata": {
                            "source_file_name": filename,
                            "mineru_cache_reused": True,
                            "output_dir": str(doc_dir),
                        },
                    }
                ]

        # If the filename stem changed between runs, fall back to any markdown file under doc_dir.
        if reuse_cache and not md_local_path.exists():
            md_candidates = sorted(doc_dir.glob("*.md"))
            if md_candidates:
                candidate = md_candidates[0]
                md_text = candidate.read_text(encoding="utf-8", errors="ignore")
                if md_text.strip():
                    logger.info("Reusing cached MinerU markdown (fallback): %s", candidate)
                    return [
                        {
                            "md_content_path": str(candidate),
                            "text": md_text,
                            "metadata": {
                                "source_file_name": filename,
                                "mineru_cache_reused": True,
                                "output_dir": str(doc_dir),
                            },
                        }
                    ]

        backend = str(kwargs.get("backend") or getattr(self.config, "backend", "vlm-transformers"))
        parse_method = str(kwargs.get("parse_method") or getattr(self.config, "parse_method", "auto"))
        lang = str(kwargs.get("lang") or getattr(self.config, "lang", "ch"))
        formula_enable = bool(kwargs.get("formula_enable", getattr(self.config, "formula_enable", True)))
        table_enable = bool(kwargs.get("table_enable", getattr(self.config, "table_enable", True)))
        start_page = int(kwargs.get("start_page", getattr(self.config, "start_page", 0)))
        end_page = kwargs.get("end_page", getattr(self.config, "end_page", None))
        end_page = int(end_page) if end_page is not None else None
        output_format = str(kwargs.get("output_format") or getattr(self.config, "output_format", "mm_md"))

        parse_result = await get_thread_pool().run_blocking(
            self.client.parse_bytes,
            file_bytes=file_data,
            filename=filename,
            backend=backend,
            parse_method=parse_method,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page=start_page,
            end_page=end_page,
            output_format=output_format,
        )

        task_id = str(parse_result.get("task_id") or "").strip()
        if not task_id or parse_result.get("status") != "success":
            raise RuntimeError(f"MinerU parsing failed: {parse_result}")

        # Mirror primary artifacts to a stable local layout:
        # - <doc_dir>/<base_filename>.md
        # - <doc_dir>/*_content_list*.json
        # - <doc_dir>/images/...
        md_local_path = doc_dir / f"{base_filename}.md"
        content_list_local_path = doc_dir / f"{base_filename}_content_list.json"
        asset_manifest_local_path = doc_dir / "asset_manifest.json"

        def _download(rel_path: str, dst: Path) -> None:
            self.client.download_task_file(task_id, rel_path, dst)

        md_rel = parse_result.get("markdown_rel_path")
        if md_rel:
            await get_thread_pool().run_blocking(_download, str(md_rel), md_local_path)

        content_rel = parse_result.get("content_list_rel_path")
        if content_rel:
            await get_thread_pool().run_blocking(_download, str(content_rel), content_list_local_path)

        manifest_rel = parse_result.get("asset_manifest_rel_path")
        if manifest_rel:
            await get_thread_pool().run_blocking(_download, str(manifest_rel), asset_manifest_local_path)

        images_meta = parse_result.get("images_metadata") or []
        for image in images_meta:
            task_rel = image.get("task_rel_path")
            rel_path = str(image.get("relative_path") or "").strip()
            filename_only = str(image.get("filename") or (Path(rel_path).name if rel_path else "")).strip()
            if not task_rel or not filename_only:
                continue
            rel_dir = Path(rel_path).parent
            subdir = rel_dir.as_posix() if rel_dir.as_posix() not in ("", ".") else "images"
            dst = doc_dir / subdir / filename_only
            await get_thread_pool().run_blocking(_download, str(task_rel), dst)

        try:
            import json

            (doc_dir / "mineru_parse_result.json").write_text(
                json.dumps(parse_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write mineru_parse_result.json: %s", exc)

        if not md_local_path.exists():
            raise RuntimeError(f"MinerU markdown download missing: {md_local_path}")

        md_text = md_local_path.read_text(encoding="utf-8", errors="ignore")
        return [
            {
                "md_content_path": str(md_local_path),
                "text": md_text,
                "metadata": {
                    "source_file_name": filename,
                    "mineru_task_id": task_id,
                    "backend": backend,
                    "parse_method": parse_method,
                    "lang": lang,
                    "output_format": output_format,
                    "output_dir": str(doc_dir),
                },
            }
        ]
