from typing import List


def summarize_tail_percent_rows(rows: List[List[str]]) -> str:
    """Best-effort table summary for LLMs when tables contain percent-style tail columns."""
    if not rows:
        return ""

    flat = " ".join(" ".join(r) for r in rows if r).lower()
    # Only activate when the table resembles a basic/total x 5-year/10-year layout.
    if "basic" not in flat or "total" not in flat or "5-year" not in flat:
        return ""

    def _is_percent(token: str) -> bool:
        return "%" in str(token or "")

    lines: List[str] = []
    plus5_condition: str | None = None
    for row in rows:
        for cell in row or []:
            token = str(cell or "").strip()
            if token and "+5%" in token:
                plus5_condition = token
                break
        if plus5_condition:
            break

    for row in rows:
        if not row or len(row) < 5:
            continue
        tail = row[-4:]
        if sum(1 for cell in tail if _is_percent(cell)) < 2:
            continue
        label = str(row[0] or "").strip()
        if not label:
            continue
        basic_5, basic_10, total_5, total_10 = (str(x or "").strip() for x in tail)
        lines.append(
            f"- {label}: basic={basic_5} (5-year), {basic_10} (10-year); total(with +5%)={total_5} (5-year), {total_10} (10-year)"
        )

    if plus5_condition:
        lines.append(f"- +5% condition: {plus5_condition}")

    return "\n".join(lines).strip()


__all__ = ["summarize_tail_percent_rows"]

