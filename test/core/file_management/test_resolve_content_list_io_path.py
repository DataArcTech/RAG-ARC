def test_build_section_tree_loads_content_list_from_io_manager(tmp_path):
    """PageIndex tree building must support io:// parsed artifact layouts via IOManager.

    In LocalDB mode this reads from the filesystem; in MinIO mode it must not depend on
    any local path mapping. This test exercises the IOManager path, not Path.glob().
    """

    import json

    from config.encapsulation.io.io_manager_config import IOManagerConfig
    from config.encapsulation.database.file_db.local_config import LocalDBConfig
    from core.file_management.pageindex.tree_builder import build_section_tree

    io_manager = IOManagerConfig(file_db_config=LocalDBConfig(base_path=str(tmp_path)), default_namespace="io").build()

    file_id = "00000000-0000-0000-0000-000000000001"
    md_path = f"io://parsed_files/mineru/{file_id}/dummy.md"
    output_dir = f"io://parsed_files/mineru/{file_id}"
    io_manager.put_text_path(md_path, text="# Title\n\nBody")
    io_manager.put_text_path(
        f"io://parsed_files/mineru/{file_id}/dummy_content_list.json",
        text=json.dumps([{"type": "text", "text": "Title", "page_idx": 0, "text_level": 1}]),
    )

    tree = build_section_tree(
        file_id=file_id,
        markdown="# Title\n\nBody",
        filename="dummy.pdf",
        md_path=md_path,
        output_dir=output_dir,
        io_manager=io_manager,
    )
    assert tree.content_list
    assert tree.max_page_idx == 0
    # Expect page ranges to be populated when content_list is present.
    assert tree.nodes and tree.nodes[0].page_start is not None
