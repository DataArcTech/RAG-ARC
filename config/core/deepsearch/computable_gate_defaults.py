"""Defaults for classifying "computable" questions in DeepSearch.

Keep these tokens centralized/configurable so production deployments can tune
the router without scattering heuristics across business code.
"""

from typing import Final


# Generic (domain-agnostic) cues that the question expects a computable result.
DEFAULT_COMPUTABLE_KEYWORDS: Final[list[str]] = [
    # English
    "how many",
    "how much",
    "count",
    "total",
    "sum",
    "average",
    "ratio",
    "rate",
    "percent",
    "percentage",
    "amount",
    "price",
    "effective date",
    "valid from",
    "valid to",
    "start date",
    "end date",
    "expiry",
    # Chinese (finance/insurance frequently used)
    "多少",
    "金额",
    "总额",
    "合计",
    "汇总",
    "平均",
    "比例",
    "费率",
    "利率",
    "保费",
    "免赔",
    "赔付",
    "生效",
    "有效期",
    "起保",
    "截至",
    "到期",
    "期限",
]


# Operator/constraint cues.
DEFAULT_COMPUTABLE_OPERATORS: Final[list[str]] = [
    ">=",
    "<=",
    ">",
    "<",
    "=",
    "至少",
    "至多",
    "大于",
    "小于",
    "不超过",
    "不少于",
    "between",
    "before",
    "after",
]


# Policy defaults --------------------------------------------------------

DEFAULT_COMPUTABLE_POLICY: Final[dict[str, int | bool]] = {
    # Strong cues: keyword/operator/date. Require at least N strong cues.
    "min_strong_hits": 1,
    # Weak cues: number/time/date. Require at least N weak cues.
    "min_weak_hits": 1,
    # When false, allow a strong-only classification (not recommended for finance/insurance).
    "require_weak_cue": True,
}
