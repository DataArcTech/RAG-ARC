import pytest

from config.core.file_management.parser_combinator_config import ParserCombinatorConfig
from config.core.file_management.parser.mineru import MinerUParserConfig
from config.core.file_management.parser.native import NativeParserConfig
from core.file_management.parser.base import AbstractParser


class MinerUParser(AbstractParser):
    def __init__(self):  # pragma: no cover
        super().__init__(config=None)  # type: ignore[arg-type]

    def get_supported_extensions(self):
        return [".pdf"]

    async def parse_file(self, file_data: bytes, filename: str, **kwargs):
        raise RuntimeError("mineru not running")


class NativeParser(AbstractParser):
    def __init__(self):  # pragma: no cover
        super().__init__(config=None)  # type: ignore[arg-type]

    def get_supported_extensions(self):
        return [".pdf"]

    async def parse_file(self, file_data: bytes, filename: str, **kwargs):
        return [{"text": "ok", "metadata": {}}]


@pytest.mark.asyncio
async def test_mineru_failure_falls_back_to_native_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "mineru")
    monkeypatch.setenv("MINERU_FALLBACK_TO_NATIVE_ON_FAILURE", "true")

    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
        mineru_parser=MinerUParserConfig(server_url="http://127.0.0.1:8899"),
    )
    parser = cfg.build()

    # Avoid real MinerU/native parsing; inject deterministic stubs.
    parser.ocr_parser = MinerUParser()
    parser.native_parser = NativeParser()
    parser._build_extension_mapping()

    results = await parser.parse_file(b"%PDF-FAKE", "demo.pdf")
    assert results and isinstance(results, list)
    meta = results[0]["metadata"]
    assert meta["parser_label"] == "native"
    assert meta["parser_fallback"]["from_label"] == "mineru"
    assert meta["parser_fallback"]["to_label"] == "native"
    assert "mineru not running" in meta["parser_fallback"]["reason"]


@pytest.mark.asyncio
async def test_mineru_failure_does_not_fallback_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "mineru")
    monkeypatch.setenv("MINERU_FALLBACK_TO_NATIVE_ON_FAILURE", "false")

    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
        mineru_parser=MinerUParserConfig(server_url="http://127.0.0.1:8899"),
    )
    parser = cfg.build()

    parser.ocr_parser = MinerUParser()
    parser.native_parser = NativeParser()
    parser._build_extension_mapping()

    with pytest.raises(RuntimeError, match="mineru not running"):
        await parser.parse_file(b"%PDF-FAKE", "demo.pdf")

