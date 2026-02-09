from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def _parse_dt(value: str) -> Optional[datetime]:
    s = str(value or "").strip()
    if not s:
        return None
    # Accept ISO-like strings; keep it permissive and return None on failure.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass(frozen=True)
class TimeWindow:
    valid_from: Optional[datetime]
    valid_to: Optional[datetime]

    @classmethod
    def from_strings(cls, *, valid_from: str = "", valid_to: str = "") -> "TimeWindow":
        return cls(valid_from=_parse_dt(valid_from), valid_to=_parse_dt(valid_to))

    def overlaps(self, other: "TimeWindow") -> Optional[bool]:
        """Return True/False when decidable, else None when insufficient data."""
        if self.valid_from is None and self.valid_to is None:
            return None
        if other.valid_from is None and other.valid_to is None:
            return None

        a_start = self.valid_from
        a_end = self.valid_to
        b_start = other.valid_from
        b_end = other.valid_to

        # Treat None endpoints as open intervals.
        if a_start is None or b_end is None:
            left_ok = True
        else:
            left_ok = a_start <= b_end

        if b_start is None or a_end is None:
            right_ok = True
        else:
            right_ok = b_start <= a_end

        return bool(left_ok and right_ok)

