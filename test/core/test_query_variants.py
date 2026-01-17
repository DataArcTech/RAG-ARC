from core.utils.query_variants import generate_query_variants


def test_query_variants_always_include_original():
    assert generate_query_variants("  hello  ")[0] == "hello"


def test_query_variants_generate_hans_hant_pair_when_applicable():
    # Simplified -> Traditional should differ for these tokens.
    variants = generate_query_variants("计划特点")
    assert variants[0] == "计划特点"
    assert "計劃特點" in variants
    # No duplicates
    assert len(variants) == len(set(variants))
