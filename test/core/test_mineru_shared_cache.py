from types import SimpleNamespace

import pytest

from core.file_management.parser.mineru import MinerUParser
from core.file_management.parser.mineru_shared_cache import (
    MinerUSharedCacheKey,
    build_parser_fingerprint,
    sha256_hex,
)


class _DummyMinerUClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse_bytes(self, **kwargs):  # noqa: ANN001
        self.calls.append("parse_bytes")
        # Minimal MinerU response expected by parser.
        return {
            "status": "success",
            "task_id": "t1",
            "markdown_rel_path": "md",
            "content_list_rel_path": None,
            "asset_manifest_rel_path": None,
            "images_metadata": [],
        }

    def download_task_file(self, task_id, rel_path, dst):  # noqa: ANN001
        self.calls.append(f"download:{rel_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == "md":
            dst.write_text("# ok\n", encoding="utf-8")
            return
        dst.write_text("{}", encoding="utf-8")


def _parser_config(tmp_path, *, shared_root, shared_mode: str):  # noqa: ANN001
    return SimpleNamespace(
        server_url="http://mineru.local",
        timeout_s=1,
        poll_interval_s=1,
        poll_timeout_s=1,
        http_max_retries=0,
        http_retry_backoff_s=0.0,
        http_retry_max_backoff_s=0.0,
        output_dir=str(tmp_path),
        reuse_cache=False,
        shared_cache_enabled=True,
        shared_cache_dir=str(shared_root),
        shared_cache_mode=shared_mode,
        backend="vlm-transformers",
        parse_method="auto",
        lang="ch",
        formula_enable=True,
        table_enable=True,
        start_page=0,
        end_page=None,
        output_format="mm_md",
    )


@pytest.mark.asyncio
async def test_mineru_parser_reuses_shared_cache_when_bytes_match(tmp_path):
    shared_root = tmp_path / "shared"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = b"same file bytes"
    bytes_sha = sha256_hex(file_bytes)
    fp = build_parser_fingerprint(
        params={
            "backend": "vlm-transformers",
            "parse_method": "auto",
            "lang": "ch",
            "formula_enable": True,
            "table_enable": True,
            "start_page": 0,
            "end_page": None,
            "output_format": "mm_md",
        }
    )
    key = MinerUSharedCacheKey(bytes_sha256=bytes_sha, parser_fingerprint=fp)
    shared_dir = shared_root / key.rel_dir()
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / ".complete").write_text("ok\n", encoding="utf-8")
    (shared_dir / "document.md").write_text("# reused\n", encoding="utf-8")

    cfg = _parser_config(out_dir, shared_root=shared_root, shared_mode="copy")
    parser = MinerUParser(cfg)
    dummy = _DummyMinerUClient()
    parser.client = dummy  # type: ignore[assignment]

    parsed = await parser.parse_file(file_bytes, "x.pdf", source_file_id="f1")
    assert parsed and isinstance(parsed, list)
    meta = parsed[0].get("metadata") or {}
    assert meta.get("mineru_shared_cache_reused") is True
    assert "reused" in (parsed[0].get("text") or "")
    assert "parse_bytes" not in dummy.calls


@pytest.mark.asyncio
async def test_mineru_parser_publishes_to_shared_cache_after_remote_parse(tmp_path):
    shared_root = tmp_path / "shared"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = b"new bytes"
    bytes_sha = sha256_hex(file_bytes)
    fp = build_parser_fingerprint(
        params={
            "backend": "vlm-transformers",
            "parse_method": "auto",
            "lang": "ch",
            "formula_enable": True,
            "table_enable": True,
            "start_page": 0,
            "end_page": None,
            "output_format": "mm_md",
        }
    )
    key = MinerUSharedCacheKey(bytes_sha256=bytes_sha, parser_fingerprint=fp)
    shared_dir = shared_root / key.rel_dir()

    cfg = _parser_config(out_dir, shared_root=shared_root, shared_mode="copy")
    parser = MinerUParser(cfg)
    dummy = _DummyMinerUClient()
    parser.client = dummy  # type: ignore[assignment]

    parsed = await parser.parse_file(file_bytes, "x.pdf", source_file_id="f2")
    assert parsed and isinstance(parsed, list)
    assert "parse_bytes" in dummy.calls

    assert (shared_dir / ".complete").exists()
    md_files = list(shared_dir.glob("*.md"))
    assert md_files
    assert md_files[0].read_text(encoding="utf-8").strip()

