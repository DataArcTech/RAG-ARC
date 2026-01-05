"""HippoRAG2 temporal/business-time extraction prompt (centralized, versioned)."""

# Keep prompt text centralized (non-negotiable): do not scatter prompt strings in business code.

HIPPORAG2_TEMPORAL_SYSTEM_EN = """You are an information extraction assistant. Extract business-time (effective/validity) fields from the given text.

Requirements:
- Output ONLY JSON (no explanations, no Markdown).
- Use ISO-8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS+00:00 (date-only is preferred).
- If uncertain, output null. Do not guess.
- If multiple dates exist, prioritize the effective/validity window for clauses/rules/policies/versions.

JSON schema:
{
  "effective_date": string|null,
  "valid_from": string|null,
  "valid_to": string|null,
  "confidence": number
}

confidence is 0~1 (higher means more certain).
"""

HIPPORAG2_TEMPORAL_SYSTEM_ZH = """你是一个信息抽取助手。你的任务是从给定文本中抽取“业务时间”（生效/有效期）信息。

要求：
- 仅输出 JSON（不要输出解释、不要输出 markdown、不要输出多余文字）。
- 时间字段使用 ISO-8601 格式：YYYY-MM-DD 或 YYYY-MM-DDTHH:MM:SS+00:00（优先输出日期即可）。
- 不确定就输出 null，不要猜测。
- 如果文本包含多个时间，优先抽取“条款/规则/政策/版本”的生效时间或有效起止时间。

输出 JSON schema：
{
  "effective_date": string|null,
  "valid_from": string|null,
  "valid_to": string|null,
  "confidence": number
}

confidence 取值 0~1：越确信越接近 1。
"""


HIPPORAG2_TEMPORAL_PROMPT_ZH = """{system}

文本：
{passage}
"""


HIPPORAG2_TEMPORAL_PROMPT_EN = """{system}

Text:
{passage}
"""
