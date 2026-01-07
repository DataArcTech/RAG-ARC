import io
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from urllib.parse import urlparse, urlunparse


class MinerUServiceClient:
    def __init__(self, base_url: str, timeout_s: int = 600):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_s = int(timeout_s)
        self.session = requests.Session()
        self._validated_base_url = False

    def _is_mineru_openapi(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        info = payload.get("info")
        title = ""
        if isinstance(info, dict):
            title = str(info.get("title") or "")
        paths = payload.get("paths")
        if not isinstance(paths, dict):
            return False
        return ("MinerU" in title) and ("/parse" in paths) and ("/health" in paths)

    def _get_openapi(self, base_url: str) -> Optional[Dict[str, Any]]:
        url = f"{str(base_url).rstrip('/')}/openapi.json"
        try:
            resp = self.session.get(url, timeout=min(max(self.timeout_s, 1), 8))
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _candidate_base_urls(self) -> list[str]:
        """
        When users set MINERU_SERVER_URL=http://localhost:8899 under WSL with SSH tunnels,
        `localhost` may resolve to an IPv4 listener that is not MinerU (port conflicts).
        Provide deterministic candidates for validation.
        """
        raw = str(self.base_url or "").strip()
        if not raw:
            return []
        parsed = urlparse(raw)
        host = (parsed.hostname or "").strip().lower()
        port = parsed.port
        if host not in {"localhost", "127.0.0.1"} or port is None:
            return [raw]

        # Prefer deterministic loopback addresses over `localhost` to avoid
        # flakey resolver ordering between IPv4/IPv6 during long-running processes.
        candidates = []
        netloc_ipv6 = f"[::1]:{port}"
        candidates.append(urlunparse(parsed._replace(netloc=netloc_ipv6)))
        netloc_ipv4 = f"127.0.0.1:{port}"
        candidates.append(urlunparse(parsed._replace(netloc=netloc_ipv4)))
        netloc_localhost = f"localhost:{port}"
        candidates.append(urlunparse(parsed._replace(netloc=netloc_localhost)))
        candidates.append(raw)

        out: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
        return out

    def _ensure_valid_base_url(self) -> None:
        if self._validated_base_url:
            return
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")

        candidates = self._candidate_base_urls()
        last_err: str | None = None
        for candidate in candidates:
            openapi = self._get_openapi(candidate)
            if self._is_mineru_openapi(openapi):
                if candidate != self.base_url:
                    import logging

                    logging.getLogger(__name__).warning(
                        "MINERU_SERVER_URL=%s does not appear to be MinerU; switching to %s based on /openapi.json",
                        self.base_url,
                        candidate,
                    )
                self.base_url = candidate.rstrip("/")
                self._validated_base_url = True
                return
            last_err = f"openapi invalid or missing for {candidate}"

        raise ValueError(
            f"MinerU server validation failed for MINERU_SERVER_URL={self.base_url}; "
            f"tried={candidates}; last_error={last_err}"
        )

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
        self._ensure_valid_base_url()
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
        self._ensure_valid_base_url()
        resp = self.session.get(f"{self.base_url}/task/{task_id}/manifest", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def download_task_file(self, task_id: str, rel_path: str, dst: Path) -> None:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
        self._ensure_valid_base_url()
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
