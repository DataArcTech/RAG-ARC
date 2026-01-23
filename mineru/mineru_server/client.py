import time
from pathlib import Path
from typing import Any, Dict

import requests

from mineru_server.polling import load_polling_config


class MinerUServerClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: int = 600,
        poll_interval_s: int | None = None,
        poll_timeout_s: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = int(timeout_s)
        default_interval, default_timeout = load_polling_config()
        self.poll_interval_s = max(1, int(poll_interval_s) if poll_interval_s is not None else default_interval)
        self.poll_timeout_s = max(0, int(poll_timeout_s) if poll_timeout_s is not None else default_timeout)
        self.session = requests.Session()

    def parse(
        self,
        *,
        file_path: Path,
        backend: str,
        parse_method: str,
        lang: str,
        formula_enable: bool,
        table_enable: bool,
        start_page: int,
        end_page: int | None,
        output_format: str,
        wait: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/parse"
        data = {
            "backend": backend,
            "parse_method": parse_method,
            "lang": lang,
            "formula_enable": "true" if formula_enable else "false",
            "table_enable": "true" if table_enable else "false",
            "start_page": str(start_page),
            "output_format": output_format,
            "wait": "false",
        }
        if end_page is not None:
            data["end_page"] = str(end_page)
        with file_path.open("rb") as f:
            resp = self.session.post(
                url,
                data=data,
                files={"file": (file_path.name, f, "application/octet-stream")},
                timeout=self.timeout_s,
            )
        resp.raise_for_status()
        result = resp.json()
        status = str(result.get("status") or "").lower()
        if status in {"success", "failed"} or not wait:
            return result
        task_id = str(result.get("task_id") or "").strip()
        if not task_id:
            return result
        return self.wait_for_parse(task_id)

    def get_parse_status(self, task_id: str) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/parse/status/{task_id}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def wait_for_parse(self, task_id: str) -> Dict[str, Any]:
        deadline = None
        if self.poll_timeout_s > 0:
            deadline = time.monotonic() + float(self.poll_timeout_s)
        while True:
            payload = self.get_parse_status(task_id)
            status = str(payload.get("status") or "").lower()
            if status in {"success", "failed"}:
                return payload
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"MinerU parse timed out after {self.poll_timeout_s}s (task_id={task_id})")
            time.sleep(self.poll_interval_s)

    def get_manifest(self, task_id: str) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/task/{task_id}/manifest", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def download_task_file(self, task_id: str, rel_path: str, dst: Path) -> None:
        rel_path = str(rel_path).lstrip("/")
        resp = self.session.get(
            f"{self.base_url}/task/{task_id}/file/{rel_path}",
            timeout=self.timeout_s,
            stream=True,
        )
        resp.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)

    def sync_task(self, task_id: str, dst_root: Path) -> Path:
        manifest = self.get_manifest(task_id)
        files = manifest.get("files", [])
        local_task_root = dst_root / "mineru_outputs" / task_id
        for entry in files:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            self.download_task_file(task_id, rel_path, local_task_root / rel_path)
        return local_task_root
