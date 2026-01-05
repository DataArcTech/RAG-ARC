import io
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class MinerUServiceClient:
    def __init__(self, base_url: str, timeout_s: int = 600):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_s = int(timeout_s)
        self.session = requests.Session()

    def parse_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        backend: str,
        parse_method: str,
        lang: str,
        formula_enable: bool,
        table_enable: bool,
        start_page: int,
        end_page: Optional[int],
        output_format: str,
    ) -> Dict[str, Any]:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
        url = f"{self.base_url}/parse"
        data = {
            "backend": str(backend),
            "parse_method": str(parse_method),
            "lang": str(lang),
            "formula_enable": "true" if formula_enable else "false",
            "table_enable": "true" if table_enable else "false",
            "start_page": str(int(start_page)),
            "output_format": str(output_format),
        }
        if end_page is not None:
            data["end_page"] = str(int(end_page))

        file_obj = io.BytesIO(file_bytes)
        files = {"file": (str(filename or "document"), file_obj, "application/octet-stream")}
        resp = self.session.post(url, data=data, files=files, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def get_manifest(self, task_id: str) -> Dict[str, Any]:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
        resp = self.session.get(f"{self.base_url}/task/{task_id}/manifest", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def download_task_file(self, task_id: str, rel_path: str, dst: Path) -> None:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
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

