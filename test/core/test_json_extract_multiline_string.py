from core.utils.json_extract import safe_json_loads


def test_safe_json_loads_repairs_multiline_string_values() -> None:
    raw = """```json
{
  "is_consistent": false,
  "confidence": 0.9,
  "issues": [
    {
      "issue_type": "misquote",
      "location": "summary",
      "description": "Line 1
Line 2"
    }
  ]
}
```"""
    parsed = safe_json_loads(raw, expected="dict")
    assert isinstance(parsed, dict)
    assert parsed["issues"][0]["description"] == "Line 1\nLine 2"

