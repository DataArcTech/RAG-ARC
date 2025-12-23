import asyncio
from pathlib import Path

from config.core.file_management.parser.native import NativeParserConfig


def test_native_parser_md_supports_repo_relative_filename(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NATIVE_PARSER_OUTPUT_DIR", str(tmp_path))
    parser = NativeParserConfig().build()

    results = asyncio.run(parser.parse_file(b"# Title\n\nHello.\n", "RAG-ARC/docs/sample.md"))
    assert results
    meta = results[0].get("metadata") or {}
    output_file = meta.get("output_file")
    assert isinstance(output_file, str) and output_file
    assert Path(output_file).exists()

