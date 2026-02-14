import logging
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from core.file_management.parser.base import AbstractParser
from core.utils.path_guard import safe_leaf_name
from framework.thread_pool import get_thread_pool

from encapsulation.remote_services.mineru_service_client import MinerUServiceClient
from core.file_management.parser.mineru_shared_cache import (
    MinerUSharedCacheKey,
    build_parser_fingerprint,
    materialize_shared_cache,
    publish_to_shared_cache,
    resolve_shared_cache_dir,
    sha256_hex,
    shared_cache_hit,
)

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
        import app_registration

        io_manager = app_registration.registrator.get_object("io_manager")
        if io_manager is None:
            raise RuntimeError("io_manager is required for MinerUParser")

        # `force_reparse` bypasses all cache reuse (local + shared) and refreshes the shared-cache
        # entry for the same (bytes_sha256, parser_fingerprint) key after a successful parse.
        #
        # Use this when the operator explicitly wants to regenerate artifacts for identical bytes
        # (e.g., MinerU upgraded or parse parameters changed outside the cache fingerprint).
        force_reparse = bool(kwargs.get("force_reparse", False))
        base_filename = Path(filename).stem or "document"
        file_ext = Path(filename).suffix.lower()
        if file_ext not in set(self.get_supported_extensions()):
            supported = ", ".join(self.get_supported_extensions())
            raise ValueError(f"Unsupported file type '{file_ext}'. Supported types: {supported}")

        output_dir = getattr(self.config, "output_dir", None)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("MinerUParser requires config.output_dir (no implicit env defaults).")
        output_dir_virtual = str(output_dir).strip()
        if not output_dir_virtual.startswith("io://"):
            raise ValueError("MinerUParser config.output_dir must be an io:// virtual path")
        source_file_id = kwargs.get("source_file_id") or kwargs.get("file_id")
        doc_key = safe_leaf_name(str(source_file_id or ""), default=base_filename)
        doc_dir_virtual = f"{output_dir_virtual.rstrip('/')}/{doc_key}"

        # Cache reuse: if a previous MinerU run already produced local markdown for this file_id,
        # allow re-indexing to reuse it without calling the remote MinerU service again.
        reuse_cache = False if force_reparse else bool(kwargs.get("reuse_cache", getattr(self.config, "reuse_cache", False)))
        md_virtual_path = f"{doc_dir_virtual}/{base_filename}.md"
        if reuse_cache:
            md_text = io_manager.get_text_path(md_virtual_path) or ""
            if md_text.strip():
                logger.info("Reusing cached MinerU markdown: %s", md_virtual_path)
                return [
                    {
                        "md_content_path": md_virtual_path,
                        "text": md_text,
                        "metadata": {
                            "source_file_name": filename,
                            "mineru_cache_reused": True,
                            "output_dir": doc_dir_virtual,
                        },
                    }
                ]

        # If the filename stem changed between runs, fall back to any markdown file under doc_dir.
        if reuse_cache:
            try:
                candidates = [ref for ref in io_manager.list_keys_path(doc_dir_virtual, limit=2000) if str(ref).lower().endswith(".md")]
            except Exception:
                candidates = []
            if candidates:
                candidate_ref = str(sorted(candidates)[0])
                md_text = io_manager.get_text_path(candidate_ref) or ""
                if md_text.strip():
                    logger.info("Reusing cached MinerU markdown (fallback): %s", candidate_ref)
                    return [
                        {
                            "md_content_path": candidate_ref,
                            "text": md_text,
                            "metadata": {
                                "source_file_name": filename,
                                "mineru_cache_reused": True,
                                "output_dir": doc_dir_virtual,
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

        # Shared-cache reuse (cross-owner/tenant): if file bytes are identical and parse params match,
        # materialize shared artifacts under this file_id dir and skip MinerU remote calls.
        shared_enabled = bool(getattr(self.config, "shared_cache_enabled", False)) and not force_reparse
        shared_root = resolve_shared_cache_dir(base_dir=getattr(self.config, "shared_cache_dir", None)) if shared_enabled else None
        shared_mode = str(getattr(self.config, "shared_cache_mode", "symlink") or "symlink")
        shared_key: MinerUSharedCacheKey | None = None
        if shared_root is not None:
            try:
                bytes_sha = sha256_hex(file_data)
                fp = build_parser_fingerprint(
                    params={
                        "backend": backend,
                        "parse_method": parse_method,
                        "lang": lang,
                        "formula_enable": bool(formula_enable),
                        "table_enable": bool(table_enable),
                        "start_page": int(start_page),
                        "end_page": None if end_page is None else int(end_page),
                        "output_format": output_format,
                    }
                )
                shared_key = MinerUSharedCacheKey(bytes_sha256=bytes_sha, parser_fingerprint=fp)
                shared_dir = shared_cache_hit(io_manager, shared_root, shared_key)
                if shared_dir is not None:
                    mat = materialize_shared_cache(io_manager=io_manager, shared_dir=shared_dir, dest_dir=doc_dir_virtual, mode=shared_mode)
                    if mat.get("ok"):
                        md_text = io_manager.get_text_path(md_virtual_path) or ""
                        if md_text.strip():
                            logger.info("Reusing shared MinerU artifacts: %s -> %s", shared_dir, doc_dir_virtual)
                            return [
                                {
                                    "md_content_path": md_virtual_path,
                                    "text": md_text,
                                    "metadata": {
                                        "source_file_name": filename,
                                        "mineru_shared_cache_reused": True,
                                        "mineru_shared_cache_sha256": bytes_sha,
                                        "mineru_shared_cache_fingerprint": fp,
                                        "mineru_shared_cache_mode": mat.get("mode"),
                                        "output_dir": doc_dir_virtual,
                                    },
                                }
                            ]
            except Exception as exc:
                logger.warning("MinerU shared-cache reuse failed; will fall back: %s", exc)
                shared_key = None

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

        # Persist primary artifacts into io:// layout (no local files):
        # - <doc_dir>/<base_filename>.md
        # - <doc_dir>/*_content_list*.json
        # - <doc_dir>/images/...
        md_rel = parse_result.get("markdown_rel_path")
        if md_rel:
            md_bytes = await get_thread_pool().run_blocking(self.client.download_task_file_bytes, task_id, str(md_rel))
            io_manager.put_bytes_path(md_virtual_path, payload=md_bytes, content_type="text/markdown; charset=utf-8")

        content_rel = parse_result.get("content_list_rel_path")
        if content_rel:
            content_bytes = await get_thread_pool().run_blocking(self.client.download_task_file_bytes, task_id, str(content_rel))
            io_manager.put_bytes_path(
                f"{doc_dir_virtual}/{base_filename}_content_list.json",
                payload=content_bytes,
                content_type="application/json; charset=utf-8",
            )

        manifest_rel = parse_result.get("asset_manifest_rel_path")
        if manifest_rel:
            manifest_bytes = await get_thread_pool().run_blocking(self.client.download_task_file_bytes, task_id, str(manifest_rel))
            io_manager.put_bytes_path(
                f"{doc_dir_virtual}/asset_manifest.json",
                payload=manifest_bytes,
                content_type="application/json; charset=utf-8",
            )

        images_meta = parse_result.get("images_metadata") or []
        for image in images_meta:
            task_rel = image.get("task_rel_path")
            rel_path = str(image.get("relative_path") or "").strip()
            filename_only = str(image.get("filename") or (Path(rel_path).name if rel_path else "")).strip()
            if not task_rel or not filename_only:
                continue
            rel_dir = Path(rel_path).parent
            subdir = rel_dir.as_posix() if rel_dir.as_posix() not in ("", ".") else "images"
            image_bytes = await get_thread_pool().run_blocking(self.client.download_task_file_bytes, task_id, str(task_rel))
            io_manager.put_bytes_path(
                f"{doc_dir_virtual}/{subdir}/{filename_only}",
                payload=image_bytes,
                content_type=None,
            )

        try:
            io_manager.put_json_path(f"{doc_dir_virtual}/mineru_parse_result.json", payload=parse_result)
        except Exception as exc:
            logger.warning("Failed to persist mineru_parse_result.json: %s", exc)

        md_text = io_manager.get_text_path(md_virtual_path) or ""
        if not md_text.strip():
            raise RuntimeError(f"MinerU markdown missing or empty: {md_virtual_path}")
        # Publish to shared cache:
        # - normal: publish on miss (no overwrite)
        # - force_reparse: overwrite to refresh the shared entry for the same bytes+fingerprint key
        if force_reparse and bool(getattr(self.config, "shared_cache_enabled", False)):
            try:
                shared_root_force = resolve_shared_cache_dir(base_dir=getattr(self.config, "shared_cache_dir", None))
                if shared_root_force is not None:
                    bytes_sha = sha256_hex(file_data)
                    fp = build_parser_fingerprint(
                        params={
                            "backend": backend,
                            "parse_method": parse_method,
                            "lang": lang,
                            "formula_enable": bool(formula_enable),
                            "table_enable": bool(table_enable),
                            "start_page": int(start_page),
                            "end_page": None if end_page is None else int(end_page),
                            "output_format": output_format,
                        }
                    )
                    publish_to_shared_cache(
                        io_manager=io_manager,
                        shared_root=shared_root_force,
                        key=MinerUSharedCacheKey(bytes_sha256=bytes_sha, parser_fingerprint=fp),
                        src_dir=doc_dir_virtual,
                        overwrite=True,
                    )
            except Exception:
                pass
        elif shared_root is not None and shared_key is not None:
            try:
                publish_to_shared_cache(io_manager=io_manager, shared_root=shared_root, key=shared_key, src_dir=doc_dir_virtual)
            except Exception:
                pass
        return [
            {
                "md_content_path": md_virtual_path,
                "text": md_text,
                "metadata": {
                    "source_file_name": filename,
                    "mineru_task_id": task_id,
                    "backend": backend,
                    "parse_method": parse_method,
                    "lang": lang,
                    "output_format": output_format,
                    "output_dir": doc_dir_virtual,
                },
            }
        ]
