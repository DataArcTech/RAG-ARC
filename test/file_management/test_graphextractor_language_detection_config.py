from core.file_management.extractor.graphextractor import detect_language_from_text


def test_detect_language_from_text_respects_threshold() -> None:
    mixed = "Hello 世界"
    assert detect_language_from_text(mixed, chinese_ratio_threshold=0.1, default_language="zh") == "zh"
    assert detect_language_from_text(mixed, chinese_ratio_threshold=0.5, default_language="zh") == "en"

