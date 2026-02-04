import time

import pytest
import requests

from encapsulation.remote_services.mineru_service_client import MinerUServiceClient


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1024 * 1024):
        yield b""


class _FakeSession:
    def __init__(self, script):
        # script: list of either Exception or _FakeResponse
        self._script = list(script)
        self.calls = []

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self._script:
            return _FakeResponse(200, {})
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _mineru_openapi_payload():
    return {
        "openapi": "3.1.0",
        "info": {"title": "MinerU API Server"},
        "paths": {"/parse": {}, "/health": {}},
    }


def test_request_with_retry_retries_connection_error(monkeypatch):
    # Speed up retries by skipping actual sleeping.
    monkeypatch.setattr(time, "sleep", lambda _: None)

    client = MinerUServiceClient(
        base_url="http://127.0.0.1:8899",
        timeout_s=1,
        poll_interval_s=1,
        poll_timeout_s=1,
        http_max_retries=2,
        http_retry_backoff_s=0.01,
        http_retry_max_backoff_s=0.01,
    )
    client._validated_base_url = True

    client.session = _FakeSession(
        [
            requests.ConnectionError("boom1"),
            requests.ConnectionError("boom2"),
            _FakeResponse(200, {"status": "success"}),
        ]
    )

    payload = client.get_parse_status("t1")
    assert payload["status"] == "success"
    assert len(client.session.calls) == 3


def test_request_with_retry_exhausts_and_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    client = MinerUServiceClient(
        base_url="http://127.0.0.1:8899",
        timeout_s=1,
        poll_interval_s=1,
        poll_timeout_s=1,
        http_max_retries=1,
        http_retry_backoff_s=0.01,
        http_retry_max_backoff_s=0.01,
    )
    client._validated_base_url = True

    client.session = _FakeSession(
        [
            requests.ConnectionError("boom1"),
            requests.ConnectionError("boom2"),
        ]
    )

    with pytest.raises(requests.ConnectionError):
        client.get_parse_status("t1")
    assert len(client.session.calls) == 2


def test_candidate_base_url_prefers_raw_then_ipv4_then_ipv6(monkeypatch):
    # If raw base_url validates, we should keep it (avoid confusing "switch to ::1" logs).
    client = MinerUServiceClient(base_url="http://localhost:8899", timeout_s=1)
    client.session = _FakeSession([_FakeResponse(200, _mineru_openapi_payload())])

    client._ensure_valid_base_url()
    assert client.base_url == "http://localhost:8899"
