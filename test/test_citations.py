from core.utils.citations import compact_sup_citations, extract_sup_keys


def test_extract_sup_keys_strict_digits() -> None:
    text = "a<sup>2</sup> <sup>001</sup> <sup>bad</sup> <sup>3</sup>b"
    assert extract_sup_keys(text) == [2, 1, 3]
    assert extract_sup_keys(text, max_key=2) == [2, 1]


def test_compact_sup_citations_reorders_adjacent_groups() -> None:
    # Model may emit citations out of order; we normalize them for UI stability.
    text = "x<sup>2</sup> <sup>1</sup>y"
    rewritten, used_keys = compact_sup_citations(text, max_key=2)
    assert used_keys == [1, 2]
    assert rewritten == "x<sup>1</sup> <sup>2</sup>y"


def test_compact_sup_citations_dense_remap_and_dedupe() -> None:
    text = "x<sup>3</sup> <sup>1</sup> <sup>3</sup>y"
    rewritten, used_keys = compact_sup_citations(text, max_key=10)
    assert used_keys == [1, 3]
    # After compaction, old 1->1, old 3->2; adjacent group is sorted and de-duped.
    assert rewritten == "x<sup>1</sup> <sup>2</sup>y"


def test_compact_sup_citations_noop_when_no_citations() -> None:
    rewritten, used_keys = compact_sup_citations("no citations here", max_key=10)
    assert rewritten == "no citations here"
    assert used_keys == []
