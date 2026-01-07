from core.utils.http_headers import build_attachment_content_disposition


def test_build_attachment_content_disposition_is_latin1_safe_for_unicode_names() -> None:
    header = build_attachment_content_disposition(
        "AXA安盛「盛利II-至尊」保费回赠及预缴利率 截止至12月31日（英文版）.pdf"
    )
    header.encode("latin-1")
    assert header.startswith("attachment;")
    assert 'filename*=' in header

