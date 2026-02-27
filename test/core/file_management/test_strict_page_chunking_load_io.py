def test_load_page_markdown_supports_io_manager(tmp_path) -> None:
    """Strict page chunking must be able to load MinerU content_list from io:// via IOManager.

    This is required for MinIO mode, where io:// has no local filesystem mapping.
    """

    import json

    from config.encapsulation.database.file_db.local_config import LocalDBConfig
    from config.encapsulation.io.io_manager_config import IOManagerConfig
    from core.file_management.pageindex.strict_page_chunking import load_page_markdown

    io_manager = IOManagerConfig(file_db_config=LocalDBConfig(base_path=str(tmp_path)), default_namespace="io").build()

    file_id = "00000000-0000-0000-0000-000000000002"
    md_path = f"io://parsed_files/mineru/{file_id}/dummy.md"
    output_dir = f"io://parsed_files/mineru/{file_id}"
    io_manager.put_text_path(md_path, text="# Title\n\nBody")
    io_manager.put_text_path(
        f"{output_dir}/dummy_content_list.json",
        text=json.dumps(
            [
                {"type": "text", "text": "Title", "page_idx": 0, "text_level": 1},
                {"type": "text", "text": "Body", "page_idx": 0, "text_level": 0},
            ]
        ),
    )

    pages, diag = load_page_markdown(md_path=md_path, output_dir=output_dir, io_manager=io_manager)
    assert diag.get("reason") == "ok"
    assert pages is not None
    assert 0 in pages
    assert "Title" in pages[0]

