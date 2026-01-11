"""Defaults for DeepSearch planner web-step policy.

Centralize heuristics/phrases used to detect 'realtime/latest' requirements so the behavior is
configurable and reviewable (no scattered hard-coded keywords across business code).
"""

DEFAULT_REALTIME_WEB_STRONG_KEYWORDS: tuple[str, ...] = (
    # Explicit web requirement
    "引用网络来源",
    "引用网络",
    "联网",
    "联网搜索",
    "网络搜索",
    "web search",
    "online source",
    "internet source",
    "cite sources",
    "tavily",
)

DEFAULT_REALTIME_WEB_INTENT_KEYWORDS: tuple[str, ...] = (
    # Time/recency intent (needs to pair with a topic keyword unless the query contains a strong keyword above)
    "实时",
    "最新",
    "当前",
    "今天",
    "截止",
    "截至",
    "real-time",
    "realtime",
    "latest",
    "current",
    "today",
    "as of",
)

DEFAULT_REALTIME_WEB_TOPIC_KEYWORDS: tuple[str, ...] = (
    # Time-sensitive topics typically requiring web evidence
    "汇率",
    "外汇",
    "价格",
    "新闻",
    "公告",
    "政策",
    "监管",
    "exchange rate",
    "fx rate",
    "price",
    "news",
    "regulation",
)

# Backward-compatible union (kept for configs that only provide a single keyword list).
DEFAULT_REALTIME_WEB_KEYWORDS: tuple[str, ...] = (
    *DEFAULT_REALTIME_WEB_STRONG_KEYWORDS,
    *DEFAULT_REALTIME_WEB_INTENT_KEYWORDS,
    *DEFAULT_REALTIME_WEB_TOPIC_KEYWORDS,
)

DEFAULT_REALTIME_WEB_STEP_DESCRIPTION_ZH = "通过网络搜索补充最新/实时信息并引用来源（例如：汇率、新闻、监管更新）。"
DEFAULT_REALTIME_WEB_STEP_DESCRIPTION_EN = (
    "Run a web search for up-to-date information and cite sources (e.g. exchange rates, news, regulatory updates)."
)
