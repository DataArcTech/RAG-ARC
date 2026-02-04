from core.utils.json_extract import safe_json_loads


def test_safe_json_loads_repairs_invalid_json_escapes() -> None:
    # Some LLMs emit Python-style escapes (e.g. \') inside JSON strings.
    raw = """```json
{
  "reasoning": "It can\\'t be answered without evidence.",
  "tool_calls": [],
  "plan": []
}
```"""
    parsed = safe_json_loads(raw, expected="dict")
    assert isinstance(parsed, dict)
    assert parsed["reasoning"] == "It can't be answered without evidence."

