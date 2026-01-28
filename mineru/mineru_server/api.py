import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from mineru_server.captioning import build_captioner
from mineru_server.config import (
    DEFAULT_CONFIG_PATH,
    PARSE_STATUS_FILENAME,
    ServerConfig,
    apply_runtime_environment,
    copy_upload_to_temp,
    ensure_mineru_import_path,
    load_config,
)
from mineru_server.env import load_dotenv
from mineru_server.postprocess import AssetPostProcessor
from mineru_server.runner import run_mineru_parse


class ParseOptions(BaseModel):
    backend: str = Field(default="vlm-transformers")
    parse_method: str = Field(default="auto")
    lang: str = Field(default="ch")
    formula_enable: bool = Field(default=True)
    table_enable: bool = Field(default=True)
    start_page: int = Field(default=0)
    end_page: Optional[int] = Field(default=None)
    output_format: str = Field(default="mm_md")  # mm_md | md_only | content_list


class ParseResult(BaseModel):
    task_id: str
    filename: str
    original_filename: Optional[str] = None
    status: str

    task_root: Optional[str] = None
    output_dir: Optional[str] = None
    markdown_path: Optional[str] = None
    content_list_path: Optional[str] = None
    images_dir: Optional[str] = None
    asset_manifest_path: Optional[str] = None

    output_dir_rel: Optional[str] = None
    markdown_rel_path: Optional[str] = None
    content_list_rel_path: Optional[str] = None
    images_dir_rel: Optional[str] = None
    asset_manifest_rel_path: Optional[str] = None

    images_metadata: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class BatchParseResult(BaseModel):
    total: int
    success: int
    failed: int
    results: List[ParseResult]


def build_app(cfg: ServerConfig) -> FastAPI:
    output_root = cfg.output_dir_path()
    temp_root = cfg.temp_dir_path()
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    parse_semaphore = asyncio.Semaphore(max(1, int(cfg.max_jobs_per_worker)))
    pending_limit = max(0, int(cfg.max_pending_jobs))
    pending_count = 0
    pending_lock = asyncio.Lock()

    captioner = build_captioner(
        caption_mode=cfg.caption_mode,
        chat_api_base_url=cfg.chat_api_base_url,
        chat_api_key=cfg.chat_api_key,
        chat_api_key_file=cfg.chat_api_key_file,
        chat_model=cfg.chat_model,
        chat_timeout_s=cfg.chat_timeout_s,
        caption_context=cfg.caption_context,
        caption_context_file=cfg.caption_context_file,
    )
    post = AssetPostProcessor(
        caption_mode=cfg.caption_mode,
        captioner=captioner,
        caption_max_images=cfg.caption_max_images,
        caption_up_tokens=cfg.caption_up_tokens,
        caption_down_tokens=cfg.caption_down_tokens,
    )

    app = FastAPI(
        title="MinerU API Server",
        description="Multimodal parsing service (MinerU) producing Markdown + assets.",
        version="4.0.0",
    )

    async def _inc_pending(slots: int) -> None:
        nonlocal pending_count
        if slots <= 0:
            return
        async with pending_lock:
            pending_count += slots

    async def _dec_pending(slots: int) -> None:
        nonlocal pending_count
        if slots <= 0:
            return
        async with pending_lock:
            pending_count = max(0, pending_count - slots)

    async def _get_pending_count() -> int:
        async with pending_lock:
            return pending_count

    async def try_acquire_capacity(slots: int = 1) -> bool:
        nonlocal pending_count
        if pending_limit <= 0 or slots <= 0:
            return True
        async with pending_lock:
            if pending_count + slots > pending_limit:
                return False
            pending_count += slots
            return True

    async def release_capacity(slots: int = 1) -> None:
        if pending_limit <= 0 or slots <= 0:
            return
        await _dec_pending(slots)

    def rel_to_task(task_root: Path, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            return path.relative_to(task_root).as_posix()
        except Exception:
            return None

    def _status_path(task_root: Path) -> Path:
        return task_root / PARSE_STATUS_FILENAME

    def _write_status(task_root: Path, payload: Dict[str, Any]) -> None:
        path = _status_path(task_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _make_status_result(
        *,
        task_id: str,
        doc_name: str,
        original_filename: str,
        task_root: Path,
        status: str,
        error: Optional[str] = None,
        processing_time: float = 0.0,
    ) -> ParseResult:
        return ParseResult(
            task_id=task_id,
            filename=doc_name,
            original_filename=original_filename,
            status=status,
            task_root=str(task_root),
            error=error,
            processing_time=processing_time,
        )

    async def parse_one(
        *,
        temp_file: Path,
        doc_name: str,
        options: ParseOptions,
        task_id: str,
        original_filename: str,
    ) -> ParseResult:
        started = time.perf_counter()
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)

        await parse_semaphore.acquire()
        try:
            method_dir, md_path, content_list_path = await run_mineru_parse(
                cfg=cfg,
                task_dir=task_root,
                file_path=temp_file,
                doc_name=doc_name,
                backend=options.backend,
                parse_method=options.parse_method,
                lang=options.lang,
                formula_enable=options.formula_enable,
                table_enable=options.table_enable,
                start_page=options.start_page,
                end_page=options.end_page,
                output_format=options.output_format,
            )

            asset_manifest, images_meta = post.process(
                task_id=task_id,
                task_root=task_root,
                doc_name=doc_name,
                method_dir=method_dir,
                markdown_path=md_path,
                content_list_path=content_list_path,
            )

            images_dir = method_dir / "images"
            if not images_dir.exists():
                images_dir = None  # type: ignore[assignment]

            return ParseResult(
                task_id=task_id,
                filename=doc_name,
                original_filename=original_filename,
                status="success",
                task_root=str(task_root),
                output_dir=str(method_dir),
                markdown_path=str(md_path) if md_path else None,
                content_list_path=str(content_list_path) if content_list_path else None,
                images_dir=str(images_dir) if images_dir else None,
                asset_manifest_path=str(asset_manifest) if asset_manifest else None,
                output_dir_rel=rel_to_task(task_root, method_dir),
                markdown_rel_path=rel_to_task(task_root, md_path),
                content_list_rel_path=rel_to_task(task_root, content_list_path),
                images_dir_rel=rel_to_task(task_root, images_dir) if images_dir else None,
                asset_manifest_rel_path=rel_to_task(task_root, asset_manifest),
                images_metadata=images_meta or None,
                metadata={
                    "backend": options.backend,
                    "parse_method": options.parse_method,
                    "lang": options.lang,
                    "formula_enabled": options.formula_enable,
                    "table_enabled": options.table_enable,
                    "output_format": options.output_format,
                    "output_root": str(task_root),
                },
                processing_time=(time.perf_counter() - started),
            )
        except Exception as exc:
            logger.exception("Failed to parse document")
            return ParseResult(
                task_id=task_id,
                filename=doc_name,
                original_filename=original_filename,
                status="failed",
                task_root=str(task_root),
                error=str(exc),
                processing_time=(time.perf_counter() - started),
            )
        finally:
            parse_semaphore.release()

    async def _run_parse_job(
        *,
        temp_file: Path,
        doc_name: str,
        options: ParseOptions,
        task_id: str,
        original_filename: str,
        task_root: Path,
        release_slot: bool = False,
    ) -> None:
        try:
            running = _make_status_result(
                task_id=task_id,
                doc_name=doc_name,
                original_filename=original_filename,
                task_root=task_root,
                status="running",
            )
            _write_status(task_root, running.model_dump())
            result = await parse_one(
                temp_file=temp_file,
                doc_name=doc_name,
                options=options,
                task_id=task_id,
                original_filename=original_filename,
            )
            _write_status(task_root, result.model_dump())
        except Exception as exc:
            failed = _make_status_result(
                task_id=task_id,
                doc_name=doc_name,
                original_filename=original_filename,
                task_root=task_root,
                status="failed",
                error=str(exc),
            )
            _write_status(task_root, failed.model_dump())
        finally:
            if release_slot:
                await release_capacity(1)
            temp_file.unlink(missing_ok=True)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "host": cfg.host,
            "port": cfg.port,
            "workers": cfg.workers,
            "max_jobs_per_worker": cfg.max_jobs_per_worker,
            "max_pending_jobs": cfg.max_pending_jobs,
            "pending_jobs": await _get_pending_count(),
            "output_dir": str(output_root),
            "temp_dir": str(temp_root),
            "model_source": cfg.model_source,
            "device_mode": cfg.device_mode,
            "backend_default": cfg.backend,
            "caption_mode": cfg.caption_mode,
        }

    @app.get("/config")
    async def config() -> Dict[str, Any]:
        return {**asdict(cfg), "chat_api_key": None}

    @app.post("/parse", response_model=ParseResult)
    async def parse_single(
        file: UploadFile = File(..., description="File to parse (PDF or image)"),
        backend: Optional[str] = Form(default=None),
        parse_method: Optional[str] = Form(default=None),
        lang: Optional[str] = Form(default=None),
        formula_enable: bool = Form(default=True),
        table_enable: bool = Form(default=True),
        start_page: int = Form(default=0),
        end_page: Optional[int] = Form(default=None),
        output_format: str = Form(default="mm_md"),
        wait: bool = Form(default=False, description="If true, wait for parsing to finish before returning."),
    ) -> ParseResult:
        capacity_acquired = False
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        from mineru_server.config import secure_filename

        if not await try_acquire_capacity(1):
            raise HTTPException(status_code=503, detail="Server is at maximum capacity. Please try again later.")
        capacity_acquired = True
        safe_name, safe_stem = secure_filename(file.filename)
        temp_file = temp_root / task_id / safe_name
        copy_upload_to_temp(file.file, temp_file)
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        try:
            options = ParseOptions(
                backend=backend or cfg.backend,
                parse_method=parse_method or cfg.parse_method,
                lang=lang or cfg.lang,
                formula_enable=bool(formula_enable),
                table_enable=bool(table_enable),
                start_page=int(start_page),
                end_page=int(end_page) if end_page is not None else None,
                output_format=str(output_format),
            )
            pending = _make_status_result(
                task_id=task_id,
                doc_name=safe_stem,
                original_filename=file.filename,
                task_root=task_root,
                status="pending",
            )
            _write_status(task_root, pending.model_dump())
            if wait:
                try:
                    result = await parse_one(
                        temp_file=temp_file,
                        doc_name=safe_stem,
                        options=options,
                        task_id=task_id,
                        original_filename=file.filename,
                    )
                    _write_status(task_root, result.model_dump())
                    temp_file.unlink(missing_ok=True)
                    return result
                finally:
                    if capacity_acquired:
                        await release_capacity(1)
                        capacity_acquired = False
            asyncio.create_task(
                _run_parse_job(
                    temp_file=temp_file,
                    doc_name=safe_stem,
                    options=options,
                    task_id=task_id,
                    original_filename=file.filename,
                    task_root=task_root,
                    release_slot=capacity_acquired,
                )
            )
            return pending
        except Exception:
            if capacity_acquired:
                await release_capacity(1)
            temp_file.unlink(missing_ok=True)
            raise

    @app.post("/parse/batch", response_model=BatchParseResult)
    async def parse_batch(
        files: List[UploadFile] = File(..., description="Multiple files to parse"),
        backend: Optional[str] = Form(default=None),
        parse_method: Optional[str] = Form(default=None),
        lang: Optional[str] = Form(default=None),
        formula_enable: bool = Form(default=True),
        table_enable: bool = Form(default=True),
        start_page: int = Form(default=0),
        end_page: Optional[int] = Form(default=None),
        output_format: str = Form(default="mm_md"),
    ) -> BatchParseResult:
        capacity_acquired = False
        capacity_slots = len(files)
        if not await try_acquire_capacity(capacity_slots):
            raise HTTPException(status_code=503, detail="Server is at maximum capacity. Please try again later.")
        capacity_acquired = True
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
        options = ParseOptions(
            backend=backend or cfg.backend,
            parse_method=parse_method or cfg.parse_method,
            lang=lang or cfg.lang,
            formula_enable=bool(formula_enable),
            table_enable=bool(table_enable),
            start_page=int(start_page),
            end_page=int(end_page) if end_page is not None else None,
            output_format=str(output_format),
        )

        temp_entries: List[Tuple[Path, str, str, str]] = []
        from mineru_server.config import secure_filename

        try:
            for idx, upload in enumerate(files):
                task_id = f"{batch_id}_{idx}"
                safe_name, safe_stem = secure_filename(upload.filename)
                temp_file = temp_root / task_id / safe_name
                copy_upload_to_temp(upload.file, temp_file)
                temp_entries.append((temp_file, safe_stem, upload.filename, task_id))

            tasks = [
                parse_one(
                    temp_file=temp_file,
                    doc_name=doc_name,
                    options=options,
                    task_id=task_id,
                    original_filename=orig_name,
                )
                for temp_file, doc_name, orig_name, task_id in temp_entries
            ]
            results = await asyncio.gather(*tasks)
            success_count = sum(1 for r in results if r.status == "success")
            return BatchParseResult(
                total=len(results),
                success=success_count,
                failed=len(results) - success_count,
                results=results,
            )
        finally:
            if capacity_acquired:
                await release_capacity(capacity_slots)
            for temp_file, *_ in temp_entries:
                temp_file.unlink(missing_ok=True)

    @app.get("/parse/status/{task_id}", response_model=ParseResult)
    async def parse_status(task_id: str) -> ParseResult:
        task_root = output_root / task_id
        status_path = _status_path(task_root)
        if not status_path.exists():
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            return ParseResult.model_validate(payload)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read parse status: {exc}")

    @app.get("/task/{task_id}/manifest")
    async def task_manifest(task_id: str) -> Dict[str, Any]:
        task_root = output_root / task_id
        if not task_root.exists():
            raise HTTPException(status_code=404, detail="Task not found")
        files: List[Dict[str, Any]] = []
        for root, _, filenames in os.walk(task_root):
            root_path = Path(root)
            for name in filenames:
                file_path = root_path / name
                try:
                    rel_path = file_path.relative_to(task_root).as_posix()
                except Exception:
                    continue
                try:
                    size = file_path.stat().st_size
                except Exception:
                    size = None
                files.append({"path": rel_path, "size": size})
        files.sort(key=lambda x: x["path"])
        return {"task_id": task_id, "file_count": len(files), "files": files}

    @app.get("/task/{task_id}/file/{rel_path:path}")
    async def task_file(task_id: str, rel_path: str):
        if not rel_path or rel_path.startswith(("/", "\\")) or ".." in Path(rel_path).parts:
            raise HTTPException(status_code=400, detail="Invalid path")
        task_root = output_root / task_id
        if not task_root.exists():
            raise HTTPException(status_code=404, detail="Task not found")
        target = (task_root / rel_path).resolve()
        try:
            target.relative_to(task_root.resolve())
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=str(target), filename=Path(rel_path).name, media_type="application/octet-stream")

    @app.get("/download/{task_id}/{filename}")
    async def download(task_id: str, filename: str):
        if "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        from mineru_server.config import secure_filename

        safe_name, _ = secure_filename(filename)
        if safe_name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        base = output_root / task_id
        if not base.exists():
            raise HTTPException(status_code=404, detail="Task not found")

        for root, _, files in os.walk(base):
            if filename in files:
                file_path = Path(root) / filename
                return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="File not found")

    return app


def create_app() -> FastAPI:
    cfg = load_config(DEFAULT_CONFIG_PATH) or ServerConfig()

    # Load `.env` for worker processes (does not override existing env by default).
    from mineru_server.config import SERVER_DIR

    load_dotenv(repo_dir=SERVER_DIR)

    env = os.environ
    updates = {}
    if not cfg.chat_api_base_url and env.get("CHAT_API_BASE_URL"):
        updates["chat_api_base_url"] = env.get("CHAT_API_BASE_URL")
    if not cfg.chat_api_key and env.get("CHAT_API_KEY"):
        updates["chat_api_key"] = env.get("CHAT_API_KEY")
    if not cfg.chat_api_key_file and env.get("CHAT_API_KEY_FILE"):
        updates["chat_api_key_file"] = env.get("CHAT_API_KEY_FILE")
    if env.get("OPENAI_CHAT_MODEL") and (not cfg.chat_model or cfg.chat_model == "gemini-2.5-flash"):
        updates["chat_model"] = env.get("OPENAI_CHAT_MODEL")
    if not cfg.caption_context and env.get("CAPTION_CONTEXT"):
        updates["caption_context"] = env.get("CAPTION_CONTEXT")
    if not cfg.caption_context_file and env.get("CAPTION_CONTEXT_FILE"):
        updates["caption_context_file"] = env.get("CAPTION_CONTEXT_FILE")
    if env.get("CAPTION_UP") and cfg.caption_up_tokens == 500:
        try:
            updates["caption_up_tokens"] = int(env.get("CAPTION_UP") or "500")
        except Exception:
            pass
    if env.get("CAPTION_DOWN") and cfg.caption_down_tokens == 500:
        try:
            updates["caption_down_tokens"] = int(env.get("CAPTION_DOWN") or "500")
        except Exception:
            pass
    if updates:
        cfg = ServerConfig(**{**asdict(cfg), **updates})

    ensure_mineru_import_path(cfg)
    apply_runtime_environment(cfg)
    return build_app(cfg)
