from core.prompts import rag_inference_prompt_loader as loader
from core.prompts import runtime_context


def test_prepend_today_line_idempotent(monkeypatch):
    monkeypatch.setattr(runtime_context, "current_local_date_str", lambda: "2026-02-11")
    base = "You are a helpful assistant."
    once = runtime_context.prepend_today_line(base)
    twice = runtime_context.prepend_today_line(once)
    assert once == twice
    assert once.splitlines()[0] == "今天是 2026-02-11。"


def test_rag_chat_system_prompt_includes_today(monkeypatch):
    monkeypatch.setattr(loader, "current_local_date_str", lambda: "2026-02-11")
    monkeypatch.setattr(runtime_context, "current_local_date_str", lambda: "2026-02-11")
    loader._cached_prompt = None
    loader._cached_key = None
    prompt = loader.get_rag_chat_system_prompt(profile="rag_inference", user_type=0)
    assert prompt.splitlines()[0] == "今天是 2026-02-11。"
