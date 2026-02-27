import json


def test_read_pages_uses_content_list_when_chunk_pages_missing(monkeypatch, tmp_path) -> None:
    # Ensure io:// paths resolve under the temp LocalDB root.
    monkeypatch.setenv("IO_STORE_BACKEND", "localdb")
    monkeypatch.setenv("IO_STORE_BASE_PATH", str(tmp_path))

    file_id = "11111111-1111-1111-1111-111111111111"
    parsed_dir = tmp_path / "parsed_files" / "mineru" / file_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    content_list = [
        {"page_idx": 0, "type": "text", "text": "Intro", "text_level": 1},
        {"page_idx": 1, "type": "text", "text": "Details", "text_level": 1},
    ]
    (parsed_dir / "dummy_content_list.json").write_text(json.dumps(content_list), encoding="utf-8")

    from core.deepsearch.tools.base import ToolRunRequest
    from core.deepsearch.tools.explore.read_structured import ReadPagesTool

    tool = ReadPagesTool()
    req = ToolRunRequest(
        question="read pages",
        plan_step="p0",
        context_evidences=[],
        adapter=None,  # simulate missing cypher/page metadata in graph
        access_scope=None,
        extra={"file_id": file_id, "page_start": 0, "page_end": 1, "goal": "test"},
    )

    result = tool.run(req)  # type: ignore[misc]
    # Tool is async; run it via event loop.
    import asyncio

    out = asyncio.run(result)
    assert len(out.evidences) == 2
    joined = "\n".join([e.content for e in out.evidences])
    assert "Intro" in joined
    assert "Details" in joined
    assert out.diagnostics.get("query_mode") == "pageindex_content_list_direct"
