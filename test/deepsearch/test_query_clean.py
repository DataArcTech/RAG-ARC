from core.deepsearch.utils.query_clean import clean_query


def test_clean_query_strips_file_lists_and_truncates():
    text = "\n".join(
        [
            "基于已上传的港险产品小册子，请对比这些储蓄/年金产品的共同点与差异：供款年期、保证回报/非保证回报。",
            "",
            "涉及文件：",
            "- docs-proj/To-AI 港险产品小册子/A.pdf",
            "- docs-proj/To-AI 港险产品小册子/B.pdf",
        ]
    )
    cleaned = clean_query(text, max_chars=240)
    assert "涉及文件" not in cleaned
    assert ".pdf" not in cleaned
    assert "供款年期" in cleaned

