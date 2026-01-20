from core.prompts.deepsearch.plan import GRAPH_PLANNER_SYSTEM_PROMPT_EN, GRAPH_PLANNER_USER_PROMPT_EN


def test_graph_planner_prompt_mentions_error_taxonomy_and_evidence_rules() -> None:
    prompt = GRAPH_PLANNER_SYSTEM_PROMPT_EN
    assert "schema_error" in prompt
    assert "policy_reject" in prompt
    assert "provider_error" in prompt
    assert "timeout" in prompt
    assert "derived" in prompt.lower()
    assert "evidence" in prompt.lower()


def test_graph_planner_user_prompt_mentions_empty_hit_stop_loss() -> None:
    prompt = GRAPH_PLANNER_USER_PROMPT_EN
    assert "empty_hit" in prompt
