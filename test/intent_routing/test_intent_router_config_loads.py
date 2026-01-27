from config.core.intent_routing.intent_router import load_intent_router_config


def test_intent_router_config_includes_answer_dissatisfied() -> None:
    cfg = load_intent_router_config("config/core/intent_routing/intent_router.toml")
    by_name = {r.name: r for r in cfg.intents}
    assert "ANSWER_DISSATISFIED" in by_name
    r = by_name["ANSWER_DISSATISFIED"]
    assert r.action == "no_retrieval"
    assert r.requires_history is True

