#!/usr/bin/env python3
"""
Render a compact Markdown table from a `rag_inference_eval.py` trace.json.

The default `report.md` is verbose by design (per-variant + per-question). This script produces:
- `table.md`: metrics table + per-question answers grouped for quick scanning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _short(text: str, n: int) -> str:
    t = str(text or "").strip().replace("\n", " ")
    if len(t) <= n:
        return t
    return t[: max(0, n - 1)] + "…"


def _top_files_str(rows: List[Dict[str, Any]], limit: int = 3) -> str:
    parts: List[str] = []
    for r in (rows or [])[: max(0, int(limit))]:
        fid = str(r.get("source_file_id") or "").strip()
        cnt = r.get("count")
        if fid:
            parts.append(f"{fid}:{cnt}")
    return ", ".join(parts)


def _distinct_files(r: Dict[str, Any]) -> int:
    fd = r.get("file_distribution") or []
    return len([x for x in fd if str((x or {}).get("source_file_id") or "").strip()])


def _max_rerank_score(r: Dict[str, Any]) -> str:
    scores: List[float] = []
    for ch in r.get("chunks") or []:
        if not isinstance(ch, dict):
            continue
        s = ch.get("rerank_score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
    if not scores:
        return ""
    return str(max(scores))


def render_table(trace: Dict[str, Any]) -> str:
    variants = list((trace.get("variants") or {}).keys())
    meta = trace.get("meta") or {}

    # Assume all variants share the same question list length.
    q_count = 0
    for v in variants:
        runs = (trace["variants"][v] or {}).get("runs") or []
        q_count = max(q_count, len(runs))

    lines: List[str] = []
    lines.append("# RAG Inference Eval (table)")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    # Metrics table (one row per variant per question).
    cols = [
        "qid",
        "variant",
        "rewrite_ms",
        "retrieve_ms",
        "rerank_ms",
        "total_ms",
        "chunks",
        "distinct_files",
        "max_rerank_score",
        "top_files",
    ]
    lines.append("## Metrics")
    lines.append("")
    lines.append("|" + "|".join(cols) + "|")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for q_idx in range(q_count):
        qid = f"Q{q_idx+1}"
        for v in variants:
            runs = (trace["variants"][v] or {}).get("runs") or []
            if q_idx >= len(runs):
                continue
            r = runs[q_idx] or {}
            timings = r.get("timings_ms") or {}
            rewrite = timings.get("rewrite")
            retrieve = timings.get("retrieve")
            rerank = timings.get("rerank")
            total = ""
            if isinstance(rewrite, int) and isinstance(retrieve, int) and isinstance(rerank, int):
                total = str(rewrite + retrieve + rerank)
            row = {
                "qid": qid,
                "variant": v,
                "rewrite_ms": rewrite if isinstance(rewrite, int) else "",
                "retrieve_ms": retrieve if isinstance(retrieve, int) else "",
                "rerank_ms": rerank if isinstance(rerank, int) else "",
                "total_ms": total,
                "chunks": len(r.get("chunks") or []),
                "distinct_files": _distinct_files(r),
                "max_rerank_score": _max_rerank_score(r),
                "top_files": _top_files_str(r.get("file_distribution") or [], limit=3),
            }
            lines.append("|" + "|".join(str(row.get(c, "")) for c in cols) + "|")
    lines.append("")

    # Answers grouped by question for scanability.
    lines.append("## Answers")
    lines.append("")
    for q_idx in range(q_count):
        qid = f"Q{q_idx+1}"
        # Use the first variant's query as the canonical label.
        query = ""
        for v in variants:
            runs = (trace["variants"][v] or {}).get("runs") or []
            if q_idx < len(runs):
                query = str((runs[q_idx] or {}).get("query") or "")
                break
        lines.append(f"### {qid}: {_short(query, 160)}")
        lines.append("")
        for v in variants:
            runs = (trace["variants"][v] or {}).get("runs") or []
            if q_idx >= len(runs):
                continue
            r = runs[q_idx] or {}
            lines.append(f"#### {v}")
            lines.append("")
            fd = r.get("file_distribution") or []
            if fd:
                lines.append(f"- top_files: {_top_files_str(fd, limit=5)}")
            lines.append("")
            lines.append("```")
            lines.append(str(r.get("answer") or "").rstrip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Render table.md from a rag_inference_eval trace.json.")
    ap.add_argument("--trace", required=True, help="Path to trace.json")
    ap.add_argument("--out", default="", help="Output path (default: alongside trace.json as table.md)")
    args = ap.parse_args()

    trace_path = Path(str(args.trace))
    trace = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
    content = render_table(trace)

    out = Path(str(args.out)) if str(args.out).strip() else trace_path.parent / "table.md"
    out.write_text(content, encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
