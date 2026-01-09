from core.retrieval.graph_retrieveal.llm_selection import parse_ranked_choice_indices


def test_parse_ranked_choice_indices_from_json_list_one_based():
    raw = "[1, 3, 3, 999]"
    assert parse_ranked_choice_indices(raw, candidate_count=5, k=3, one_based=True) == [0, 2]


def test_parse_ranked_choice_indices_from_free_text():
    raw = "I pick 2, 5, 1 because ..."
    assert parse_ranked_choice_indices(raw, candidate_count=5, k=2, one_based=True) == [1, 4]


def test_parse_ranked_choice_indices_from_object_payload():
    raw = '{\"indices\": [\"3\", \"1\"]}'
    assert parse_ranked_choice_indices(raw, candidate_count=3, k=5, one_based=True) == [2, 0]

