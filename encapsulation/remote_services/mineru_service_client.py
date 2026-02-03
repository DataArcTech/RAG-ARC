import io
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from urllib.parse import urlparse, urlunparse
import logging

logger = logging.getLogger(__name__)


class MinerUServiceClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: int = 600,
        poll_interval_s: int = 5,
        poll_timeout_s: int = 0,
        http_max_retries: int = 0,
        http_retry_backoff_s: float = 1.0,
        http_retry_max_backoff_s: float = 8.0,
    ):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_s = int(timeout_s)
        self.poll_interval_s = max(1, int(poll_interval_s))
        self.poll_timeout_s = max(0, int(poll_timeout_s))
        self.http_max_retries = max(0, int(http_max_retries))
        self.http_retry_backoff_s = max(0.0, float(http_retry_backoff_s))
        self.http_retry_max_backoff_s = max(0.0, float(http_retry_max_backoff_s))
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

        # Prefer explicit IPv4 loopback first for SSH tunnels that bind to 127.0.0.1.
        # Then try IPv6 loopback, and finally preserve `localhost`.
        candidates = [raw]
        netloc_ipv4 = f"127.0.0.1:{port}"
        candidates.append(urlunparse(parsed._replace(netloc=netloc_ipv4)))
        netloc_ipv6 = f"[::1]:{port}"
        candidates.append(urlunparse(parsed._replace(netloc=netloc_ipv6)))
        netloc_localhost = f"localhost:{port}"
        candidates.append(urlunparse(parsed._replace(netloc=netloc_localhost)))

        out: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
        return out

    def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """
        Best-effort retry wrapper for transient network errors.

        We retry only a bounded number of times to avoid hiding persistent failures.
        """
        max_retries = int(self.http_max_retries)
        attempt = 0
        while True:
            try:
                resp = self.session.request(method, url, **kwargs)
                # Retry on common transient gateway errors.
                if resp.status_code in {502, 503, 504} and attempt < max_retries:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                return resp
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
                requests.HTTPError,
            ) as exc:
                if attempt >= max_retries:
                    raise
                backoff = min(self.http_retry_backoff_s * (2**attempt), self.http_retry_max_backoff_s)
                attempt += 1
                logger.warning(
                    "MinerU request retrying (%s %s) attempt=%s/%s backoff_s=%.2f error=%s",
                    method,
                    url,
                    attempt,
                    max_retries,
                    backoff,
                    str(exc),
                )
                time.sleep(backoff)
            except requests.RequestException as exc:
                # Non-transient request errors: do not retry by default.
                raise exc

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
                    logger.info("MinerU base_url selected: %s (input=%s)", candidate, self.base_url)
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
            "wait": "false",
        }
        if end_page is not None:
            data["end_page"] = str(int(end_page))

        file_obj = io.BytesIO(file_bytes)
        files = {"file": (str(filename or "document"), file_obj, "application/octet-stream")}
        resp = self._request_with_retry("POST", url, data=data, files=files, timeout=self.timeout_s)
        resp.raise_for_status()
        result = resp.json()
        status = str(result.get("status") or "").lower()
        if status in {"success", "failed"}:
            return result
        task_id = str(result.get("task_id") or "").strip()
        if not task_id:
            return result
        return self.wait_for_parse(task_id)

    def get_parse_status(self, task_id: str) -> Dict[str, Any]:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
        self._ensure_valid_base_url()
        resp = self._request_with_retry("GET", f"{self.base_url}/parse/status/{task_id}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def wait_for_parse(self, task_id: str) -> Dict[str, Any]:
        deadline = None
        if self.poll_timeout_s > 0:
            deadline = time.monotonic() + float(self.poll_timeout_s)
        while True:
            status_payload = self.get_parse_status(task_id)
            status = str(status_payload.get("status") or "").lower()
            if status in {"success", "failed"}:
                return status_payload
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"MinerU parse timed out after {self.poll_timeout_s}s (task_id={task_id})")
            time.sleep(self.poll_interval_s)

    def get_manifest(self, task_id: str) -> Dict[str, Any]:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
        self._ensure_valid_base_url()
        resp = self._request_with_retry("GET", f"{self.base_url}/task/{task_id}/manifest", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def download_task_file(self, task_id: str, rel_path: str, dst: Path) -> None:
        if not self.base_url:
            raise ValueError("MinerU base_url is empty (set MINERU_SERVER_URL).")
        self._ensure_valid_base_url()
        rel_path = str(rel_path).lstrip("/")
        resp = self._request_with_retry(
            "GET",
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
