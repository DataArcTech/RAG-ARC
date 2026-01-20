from core.file_management.pdf_glyph_name_decoder import decode_pypdf_glyph_names


def test_decode_pypdf_glyph_names_decodes_lining_figures() -> None:
    raw = "費用及收費/seven.lf。價值/six.lf不足以繳付/three.lf/zero.lf日內通知我們。"
    decoded = decode_pypdf_glyph_names(raw)

    assert decoded.changed is True
    assert decoded.text == "費用及收費7。價值6不足以繳付30日內通知我們。"


def test_decode_pypdf_glyph_names_leaves_other_names_untouched() -> None:
    raw = "詳情請瀏覽/www.zurich.com.hk 及參閱附錄。"
    decoded = decode_pypdf_glyph_names(raw)

    assert decoded.changed is False
    assert decoded.text == raw

