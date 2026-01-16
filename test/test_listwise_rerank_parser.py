from encapsulation.llm.rerank.listwise import _normalize_ranked_indices, _parse_ranked_indices


def test_listwise_parser_fills_to_top_k_when_llm_returns_single_id():
    # LLM outputs a valid JSON array but too short; we must fill deterministically.
    out = _parse_ranked_indices("[1]", num_chunks=10, top_k=5)
    assert out == [0, 1, 2, 3, 4]


def test_listwise_parser_dedupes_and_fills():
    out = _parse_ranked_indices("[2,2,2]", num_chunks=5, top_k=4)
    assert out == [1, 0, 2, 3]


def test_listwise_parser_accepts_zero_based_indices_defensively():
    out = _parse_ranked_indices("[0, 3]", num_chunks=6, top_k=4)
    # Per prompt we treat positive ints as 1-based; 0 is accepted defensively as 0-based.
    assert out == [0, 2, 1, 3]


def test_normalize_indices_handles_non_list_inputs():
    out = _normalize_ranked_indices({"bad": "shape"}, num_chunks=5, top_k=3)
    assert out == [0, 1, 2]
