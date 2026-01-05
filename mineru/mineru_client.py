#!/usr/bin/env python3
"""
Simple MinerU HTTP client + CLI.

Usage example:
    python mineru_client.py parse --file demo/pdfs/demo3.pdf --output-dir ./mineru_outputs
Environment variables:
    MINERU_SERVER_URL: override default server URL (http://127.0.0.1:8899)
"""
import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

LOGGER = logging.getLogger("mineru_client")


def _bool_to_form_value(flag: bool) -> str:
    return "true" if flag else "false"


class MinerUClient:
    """Minimal HTTP client wrapper for interacting with MinerU server."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        LOGGER.debug("Initialized MinerUClient base_url=%s timeout=%s", self.base_url, timeout)

    def parse_file(
        self,
        file_path: Path,
        backend: str = "vlm-transformers",
        parse_method: str = "auto",
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        start_page: int = 0,
        end_page: Optional[int] = None,
        output_format: str = "mm_md",
    ) -> Dict:
        """Upload a document and trigger parsing."""
        url = f"{self.base_url}/parse"
        payload = {
            "backend": backend,
            "parse_method": parse_method,
            "lang": lang,
            "formula_enable": _bool_to_form_value(formula_enable),
            "table_enable": _bool_to_form_value(table_enable),
            "start_page": str(start_page),
            "output_format": output_format,
        }
        if end_page is not None:
            payload["end_page"] = str(end_page)

        LOGGER.info("Uploading %s to %s", file_path, url)
        with file_path.open("rb") as file_stream:
            files = {"file": (file_path.name, file_stream, "application/octet-stream")}
            response = self.session.post(url, data=payload, files=files, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        LOGGER.info(
            "MinerU parse success: task_id=%s status=%s processing_time=%s",
            result.get("task_id"),
            result.get("status"),
            result.get("processing_time"),
        )
        return result

    def download_file(self, task_id: str, filename: str, destination: Path) -> None:
        """Download a single artifact from MinerU server."""
        url = f"{self.base_url}/download/{task_id}/{filename}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Downloading %s -> %s", url, destination)
        with self.session.get(url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            with destination.open("wb") as outfile:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        outfile.write(chunk)

    def get_task_manifest(self, task_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/task/{task_id}/manifest"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def download_task_file(self, task_id: str, rel_path: str, destination: Path) -> None:
        rel_path = str(rel_path).lstrip("/")
        url = f"{self.base_url}/task/{task_id}/file/{rel_path}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.session.get(url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            with destination.open("wb") as outfile:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        outfile.write(chunk)

    def sync_task_outputs(self, task_id: str, output_root: Path) -> Path:
        """
        Sync full server-side `mineru_outputs/{task_id}/...` to local:
            {output_root}/mineru_outputs/{task_id}/...
        """
        manifest = self.get_task_manifest(task_id)
        files = manifest.get("files", [])
        local_task_root = output_root / "mineru_outputs" / task_id
        for entry in files:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            destination = local_task_root / rel_path
            self.download_task_file(task_id, rel_path, destination)
        return local_task_root

    def download_artifacts(self, parse_result: Dict, output_root: Path) -> Dict[str, List[str]]:
        """Download markdown/content list/images to local output directory."""
        task_id = parse_result.get("task_id")
        doc_name = parse_result.get("filename") or parse_result.get("original_filename") or "document"
        doc_dir = output_root / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []

        def _download_by_rel(rel_path: Optional[str], *, subdir: Optional[str] = None) -> None:
            if not rel_path:
                return
            filename = Path(rel_path).name
            target_dir = doc_dir if not subdir else doc_dir / subdir
            destination = target_dir / filename
            self.download_task_file(str(task_id), rel_path, destination)
            saved_files.append(str(destination))

        def _download_by_filename(path_str: Optional[str], *, subdir: Optional[str] = None) -> None:
            if not path_str:
                return
            filename = Path(path_str).name
            target_dir = doc_dir if not subdir else doc_dir / subdir
            destination = target_dir / filename
            self.download_file(str(task_id), filename, destination)
            saved_files.append(str(destination))

        def _download_primary(primary_abs_path: Optional[str], primary_rel_path: Optional[str]) -> None:
            # Prefer the collision-free /task/<id>/file/<rel_path> endpoint when available.
            if primary_rel_path:
                _download_by_rel(primary_rel_path)
            else:
                _download_by_filename(primary_abs_path)

        _download_primary(parse_result.get("markdown_path"), parse_result.get("markdown_rel_path"))
        _download_primary(parse_result.get("content_list_path"), parse_result.get("content_list_rel_path"))
        _download_primary(parse_result.get("asset_manifest_path"), parse_result.get("asset_manifest_rel_path"))

        images_meta = parse_result.get("images_metadata") or []
        if images_meta:
            for image in images_meta:
                task_rel = image.get("task_rel_path")
                rel_path = image.get("relative_path", "")
                filename = image.get("filename") or Path(rel_path).name
                if not filename or not task_id:
                    continue
                # Prefer task_rel_path when the server provides it; else fall back to legacy filename download.
                if task_rel:
                    rel_dir = Path(rel_path).parent
                    subdir = str(rel_dir) if str(rel_dir) not in ("", ".") else "images"
                    self.download_task_file(str(task_id), str(task_rel), doc_dir / subdir / filename)
                    saved_files.append(str(doc_dir / subdir / filename))
                else:
                    rel_dir = Path(rel_path).parent
                    subdir = str(rel_dir) if str(rel_dir) not in ("", ".") else "images"
                    self.download_file(str(task_id), filename, doc_dir / subdir / filename)
                    saved_files.append(str(doc_dir / subdir / filename))

        meta_path = doc_dir / "parse_result.json"
        meta_path.write_text(json.dumps(parse_result, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_files.append(str(meta_path))

        return {"document_dir": str(doc_dir), "files": saved_files}

    def parse_and_download(
        self,
        file_path: Path,
        output_root: Path,
        backend: str,
        parse_method: str,
        lang: str,
        formula_enable: bool,
        table_enable: bool,
        start_page: int,
        end_page: Optional[int],
        output_format: str,
        *,
        download_artifacts: bool,
        sync_task: bool,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        parse_result = self.parse_file(
            file_path,
            backend=backend,
            parse_method=parse_method,
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page=start_page,
            end_page=end_page,
            output_format=output_format,
        )
        document_dir = None
        if download_artifacts:
            download_summary = self.download_artifacts(parse_result, output_root)
            document_dir = download_summary.get("document_dir")
        if sync_task and parse_result.get("task_id"):
            try:
                self.sync_task_outputs(str(parse_result["task_id"]), output_root)
            except Exception as exc:
                LOGGER.warning("Failed to sync mineru_outputs for task_id=%s: %s", parse_result.get("task_id"), exc)
        ended = time.perf_counter()
        return {
            "file": str(file_path),
            "task_id": parse_result.get("task_id"),
            "status": parse_result.get("status"),
            "server_processing_time": parse_result.get("processing_time"),
            "wall_time": ended - started,
            "document_dir": document_dir,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinerU client helper.")
    parser.add_argument("command", choices=["parse", "bench"], help="Supported command.")
    parser.add_argument("--file", type=Path, required=True, help="Path to the document to parse.")
    parser.add_argument("--files", type=Path, nargs="*", help="Multiple files for bench mode.")
    parser.add_argument("--output-dir", type=Path, default=Path("./mineru_client_outputs"), help="Directory to store downloads.")
    parser.add_argument("--base-url", default=None, help="MinerU server base URL (default: env MINERU_SERVER_URL or http://127.0.0.1:8899).")
    parser.add_argument("--backend", default="vlm-transformers", help="MinerU backend (recommended: vlm-transformers).")
    parser.add_argument("--parse-method", default="auto", help="Parse method.")
    parser.add_argument("--lang", default="ch", help="Language hint.")
    parser.add_argument("--formula-enable", action="store_true", default=True, help="Enable formula recognition.")
    parser.add_argument("--no-formula", action="store_false", dest="formula_enable", help="Disable formula recognition.")
    parser.add_argument("--table-enable", action="store_true", default=True, help="Enable table recognition.")
    parser.add_argument("--no-table", action="store_false", dest="table_enable", help="Disable table recognition.")
    parser.add_argument("--start-page", type=int, default=0, help="Start page (0-based).")
    parser.add_argument("--end-page", type=int, default=None, help="End page (inclusive).")
    parser.add_argument("--output-format", default="mm_md", choices=["mm_md", "md_only", "content_list"], help="Output format.")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds.")
    parser.add_argument("--concurrency", type=int, default=2, help="Bench concurrency level.")
    parser.add_argument("--sync-task", action="store_true", default=True, help="Sync server mineru_outputs/<task_id>/ to local output-dir.")
    parser.add_argument("--no-sync-task", action="store_false", dest="sync_task", help="Disable syncing full task outputs.")
    parser.add_argument("--download", action="store_true", default=True, help="Download primary artifacts (md/content_list/images).")
    parser.add_argument("--no-download", action="store_false", dest="download", help="Skip downloading artifacts (useful for benchmarking).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Client log level.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level)
    LOGGER.setLevel(log_level)

    base_url = args.base_url or os.environ.get("MINERU_SERVER_URL") or "http://127.0.0.1:8899"
    client = MinerUClient(base_url=base_url, timeout=args.timeout)
    if args.command == "parse":
        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        parse_result = client.parse_file(
            args.file,
            backend=args.backend,
            parse_method=args.parse_method,
            lang=args.lang,
            formula_enable=args.formula_enable,
            table_enable=args.table_enable,
            start_page=args.start_page,
            end_page=args.end_page,
            output_format=args.output_format,
        )
        if args.download:
            download_summary = client.download_artifacts(parse_result, args.output_dir)
            LOGGER.info("Artifacts saved under %s", download_summary["document_dir"])
        if args.sync_task and parse_result.get("task_id"):
            task_root = client.sync_task_outputs(str(parse_result["task_id"]), args.output_dir)
            LOGGER.info("Synced mineru outputs to %s", task_root)
        return

    if args.command == "bench":
        file_list = list(args.files or [])
        if not file_list:
            if args.file:
                file_list = [args.file]
        if not file_list:
            parser.error("bench requires --files or --file")
        for path in file_list:
            if not path.exists():
                parser.error(f"File not found: {path}")

        concurrency = max(1, int(args.concurrency))
        started = time.perf_counter()
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    client.parse_and_download,
                    path,
                    args.output_dir,
                    args.backend,
                    args.parse_method,
                    args.lang,
                    args.formula_enable,
                    args.table_enable,
                    args.start_page,
                    args.end_page,
                    args.output_format,
                    download_artifacts=bool(args.download),
                    sync_task=bool(args.sync_task),
                )
                for path in file_list
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    LOGGER.error("Bench task failed: %s", exc)
                    results.append({"status": "failed", "error": str(exc)})

        ended = time.perf_counter()
        summary = {
            "base_url": args.base_url,
            "backend": args.backend,
            "concurrency": concurrency,
            "total_files": len(file_list),
            "wall_time": ended - started,
            "results": results,
        }
        out_path = args.output_dir / "bench_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Bench done: total=%s concurrency=%s wall_time=%.2fs summary=%s", len(file_list), concurrency, summary["wall_time"], out_path)
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()