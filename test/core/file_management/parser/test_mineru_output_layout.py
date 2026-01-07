import pytest

from config.core.file_management.parser.mineru import MinerUParserConfig


@pytest.mark.asyncio
async def test_mineru_parser_uses_source_file_id_for_output_dir(tmp_path) -> None:
    cfg = MinerUParserConfig(
        server_url="http://127.0.0.1:8899",
        output_dir=str(tmp_path),
    )
    parser = cfg.build()

    def fake_parse_bytes(**_: object):
        return {
            "status": "success",
            "task_id": "task-1",
            "markdown_rel_path": "doc.md",
            "content_list_rel_path": "doc_content_list.json",
            "asset_manifest_rel_path": "asset_manifest.json",
            "images_metadata": [
                {
                    "task_rel_path": "images/demo.jpg",
                    "relative_path": "images/demo.jpg",
                    "filename": "demo.jpg",
                }
            ],
        }

    def fake_download_task_file(task_id: str, rel_path: str, dst) -> None:
        assert task_id == "task-1"
        assert rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f"{rel_path}".encode("utf-8"))

    parser.client.parse_bytes = fake_parse_bytes  # type: ignore[method-assign]
    parser.client.download_task_file = fake_download_task_file  # type: ignore[method-assign]

    results = await parser.parse_file(b"%PDF-FAKE", "same.pdf", source_file_id="file-123")
    assert len(results) == 1
    meta = results[0]["metadata"]
    assert meta["output_dir"] == str(tmp_path / "file-123")

    # Same filename, different file_id must not collide.
    results2 = await parser.parse_file(b"%PDF-FAKE2", "same.pdf", source_file_id="file-456")
    meta2 = results2[0]["metadata"]
    assert meta2["output_dir"] == str(tmp_path / "file-456")


@pytest.mark.asyncio
async def test_mineru_parser_sanitizes_output_dir_name(tmp_path) -> None:
    cfg = MinerUParserConfig(
        server_url="http://127.0.0.1:8899",
        output_dir=str(tmp_path),
    )
    parser = cfg.build()

    def fake_parse_bytes(**_: object):
        return {"status": "success", "task_id": "task-2", "markdown_rel_path": "doc.md"}

    def fake_download_task_file(task_id: str, rel_path: str, dst) -> None:
        assert task_id == "task-2"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("ok", encoding="utf-8")

    parser.client.parse_bytes = fake_parse_bytes  # type: ignore[method-assign]
    parser.client.download_task_file = fake_download_task_file  # type: ignore[method-assign]

    results = await parser.parse_file(b"%PDF-FAKE", "same.pdf", source_file_id="../evil")
    meta = results[0]["metadata"]
    # Must stay under tmp_path; no traversal.
    assert meta["output_dir"].startswith(str(tmp_path))
