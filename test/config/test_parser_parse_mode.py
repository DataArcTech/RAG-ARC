import pytest

from config.core.file_management.parser_mode import resolve_parser_parse_mode
from config.core.file_management.parser_combinator_config import ParserCombinatorConfig
from config.core.file_management.parser.native import NativeParserConfig
from config.core.file_management.parser.dots_ocr import DotsOCRParserConfig
from config.encapsulation.llm.parse.dots_ocr import DotsOCRConfig
from config.core.file_management.parser.mineru import MinerUParserConfig


def test_resolve_parser_parse_mode_default_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARSER_PARSE_MODE", raising=False)
    assert resolve_parser_parse_mode() == "native"


def test_resolve_parser_parse_mode_invalid_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "weird")
    assert resolve_parser_parse_mode() == "native"


def test_parser_combinator_env_native_disables_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "native")
    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
        mineru_parser=MinerUParserConfig(server_url="http://127.0.0.1:8899"),
    )
    resolved = cfg._resolve_for_build()
    assert resolved.ocr_parser is None


def test_parser_combinator_env_mineru_selects_mineru(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "mineru")
    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
        parse_mode="native",
        mineru_parser=MinerUParserConfig(server_url="http://127.0.0.1:8899"),
    )
    resolved = cfg._resolve_for_build()
    assert resolved.ocr_parser is not None
    assert resolved.ocr_parser.type == "mineru_parser"


def test_parser_combinator_env_dotsocr_selects_dotsocr(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "dotsocr")
    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
        dotsocr_parser=DotsOCRParserConfig(dots_ocr=DotsOCRConfig()),
    )
    resolved = cfg._resolve_for_build()
    assert resolved.ocr_parser is not None
    assert resolved.ocr_parser.type == "dots_ocr_parser"


def test_parser_combinator_env_dotsocr_requires_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "dotsocr")
    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
    )
    with pytest.raises(ValueError, match="PARSER_PARSE_MODE=dotsocr"):
        cfg._resolve_for_build()


def test_parser_combinator_env_mineru_requires_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PARSER_PARSE_MODE", "mineru")
    cfg = ParserCombinatorConfig(
        base_output_dir=str(tmp_path),
        native_parser=NativeParserConfig(),
    )
    with pytest.raises(ValueError, match="PARSER_PARSE_MODE=mineru"):
        cfg._resolve_for_build()

