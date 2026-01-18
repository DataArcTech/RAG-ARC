"""
RAG inference evaluation harness (end-to-end: rewrite -> retrieve -> rerank -> LLM).

This script is meant for reproducible, configurable experiments. It does NOT hardcode
any domain/product keywords; question sets are provided externally.

Outputs:
- JSON trace (machine-readable)
- Markdown report (human-readable)
"""
import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from config.application.rag_inference_config import RAGInferenceConfig
from framework.register import Register


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_dotenv(path: str) -> None:
    """
    Minimal .env loader for experiment scripts.

    We intentionally do not implement full shell parsing; this is enough for common KEY=VALUE lines.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _parse_questions(text: str) -> List[str]:
    """
    Best-effort question parser.

    Supports:
    - One question per line (non-empty).
    - Markdown enumerations like "1. ...." (extract the RHS).
    """
    enumerated: List[str] = []
    fallback: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip headings / code fences.
        if line.startswith("#") or line.startswith("```"):
            continue

        m = re.match(r"^(\d+)[\.\)]\s+(.*)$", line)
        if m:
            q = (m.group(2) or "").strip()
            if q:
                # Filter out enumerated non-questions like section titles ("1. xxx") by a minimal heuristic:
                # - allow short lines if they look like a question (ends with '?' or '？')
                # - otherwise require a small minimum length
                looks_like_question = q.endswith("?") or q.endswith("？")
                if looks_like_question or len(q) >= 8:
                    enumerated.append(q)
            continue

        # Fallback: treat non-empty lines as questions only when there is no explicit enumeration.
        if len(line) <= 300:
            fallback.append(line)

    out = enumerated if enumerated else fallback
    # De-duplicate while preserving order.
    uniq: List[str] = []
    for q in out:
        q = str(q or "").strip()
        if q and q not in uniq:
            uniq.append(q)
    return uniq


def _file_distribution(chunks: List[Any], *, limit: int = 10) -> List[Dict[str, Any]]:
    from collections import Counter

    def _coerce_file_id(meta: Any) -> Optional[str]:
        if not isinstance(meta, dict):
            return None
        for key in ("source_file_id", "sourceFileId", "file_id", "fileId", "document_id", "documentId"):
            token = str(meta.get(key) or "").strip()
            if token:
                return token
        return None

    ctr: Counter[str] = Counter()
    name_by_id: Dict[str, str] = {}
    for ch in chunks:
        meta = getattr(ch, "metadata", None) or {}
        fid = _coerce_file_id(meta)
        if not fid:
            continue
        ctr[fid] += 1
        if fid not in name_by_id:
            name = str(meta.get("filename") or "").strip()
            if name:
                name_by_id[fid] = name

    top: List[Dict[str, Any]] = []
    for fid, count in ctr.most_common(max(int(limit), 0)):
        top.append({"source_file_id": fid, "count": int(count), "filename": name_by_id.get(fid)})
    return top


def _chunk_brief(chunk: Any, *, preview_chars: int = 140) -> Dict[str, Any]:
    meta = getattr(chunk, "metadata", None) or {}
    content = getattr(chunk, "content", "")
    if isinstance(content, dict):
        content = content.get("text") or content.get("content") or ""
    preview = str(content or "").strip().replace("\n", " ")
    if len(preview) > preview_chars:
        preview = preview[:preview_chars] + "..."
    return {
        "id": getattr(chunk, "id", None),
        "filename": meta.get("filename"),
        "source_file_id": meta.get("source_file_id") or meta.get("sourceFileId") or meta.get("file_id") or meta.get("fileId"),
        "score": meta.get("score"),
        "rerank_score": meta.get("rerank_score"),
        "_hipporag_ppr_score": meta.get("_hipporag_ppr_score"),
        "_hipporag_dense_score": meta.get("_hipporag_dense_score"),
        "_hipporag_selection_strategy": meta.get("_hipporag_selection_strategy"),
        "preview": preview,
    }


def _load_config_dict(path: Path) -> Dict[str, Any]:
    raw = json.loads(_read_text(path))
    # Reuse the same env substitution logic as the main app.
    reg = Register()
    return reg._substitute_env_vars(raw)  # type: ignore[attr-defined]


def _build_rag_inference(cfg_dict: Dict[str, Any]) -> Any:
    cfg = RAGInferenceConfig(**cfg_dict)
    return cfg.build()


def _variant_overrides(variant: str) -> Dict[str, Any]:
    """
    Return retrieval-level overrides for the pruned_hipporag_neo4j_retrieval retriever.
    """
    if variant == "graph_base":
        return {
            "damping_factor": 0.5,
            "ppr_directed_mode": "auto",
            "ppr_push_threshold_mode": "residual",
            "entity_reset_weight_aggregation": "overwrite",
            "dense_file_closure_enabled": False,
        }
    if variant == "graph_optimized":
        # Mirror the tuned defaults (should match repo defaults / json config).
        return {
            "damping_factor": 0.3,
            "ppr_directed_mode": "off",
            "ppr_push_threshold_mode": "residual_over_degree",
            "entity_reset_weight_aggregation": "sum_avg",
            "dense_file_closure_enabled": True,
        }
    raise ValueError(f"Unknown graph variant: {variant}")


def _select_retrievers(cfg_dict: Dict[str, Any], *, mode: str) -> None:
    """
    Mutate cfg_dict in-place: keep only selected retrieval backends.
    """
    retrieval = cfg_dict.get("retrieval_config") or {}
    retrievers = list(retrieval.get("retrievers") or [])
    if not retrievers:
        raise ValueError("retrieval_config.retrievers is empty")

    def _rtype(r: Any) -> str:
        if isinstance(r, dict):
            return str(r.get("type") or "")
        return ""

    if mode == "all":
        kept = retrievers
    elif mode == "dense_bm25":
        kept = [r for r in retrievers if _rtype(r) in {"dense", "tantivy_bm25"}]
    elif mode == "graph":
        kept = [r for r in retrievers if _rtype(r) in {"pruned_hipporag_neo4j_retrieval"}]
    else:
        raise ValueError(f"Unknown retriever mode: {mode}")

    retrieval["retrievers"] = kept
    cfg_dict["retrieval_config"] = retrieval


def _apply_graph_overrides(cfg_dict: Dict[str, Any], *, graph_variant: str) -> None:
    retrieval = cfg_dict.get("retrieval_config") or {}
    retrievers = list(retrieval.get("retrievers") or [])

    applied = False
    overrides = _variant_overrides(graph_variant)
    for r in retrievers:
        if not isinstance(r, dict):
            continue
        if r.get("type") != "pruned_hipporag_neo4j_retrieval":
            continue
        for k, v in overrides.items():
            r[k] = v
        applied = True
    if not applied:
        raise ValueError("No pruned_hipporag_neo4j_retrieval retriever found to apply overrides")


@dataclass
class RunItem:
    query: str
    rewritten_query: Optional[str] = None
    timings_ms: Dict[str, int] = field(default_factory=dict)
    answer: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    file_distribution: List[Dict[str, Any]] = field(default_factory=list)


def _run_one(rag: Any, *, query: str, owner_id: uuid.UUID) -> RunItem:
    events: List[Dict[str, Any]] = []

    def _cb(payload: Dict[str, Any]) -> None:
        # Keep raw progress events for later aggregation.
        if isinstance(payload, dict):
            events.append(dict(payload))

    answer, chunks, _subgraph_data, _subgraph_info = rag.chat(query=query, owner_id=owner_id, progress_callback=_cb)

    timings: Dict[str, int] = {}
    rewritten: Optional[str] = None
    for ev in events:
        stage = str(ev.get("stage") or "")
        if stage and ev.get("status") == "end":
            dur = ev.get("duration_ms")
            if isinstance(dur, int):
                timings[stage] = dur
        if stage == "rewrite" and ev.get("status") == "end":
            rewritten = str(ev.get("rewritten_query") or "").strip() or None

    item = RunItem(
        query=query,
        rewritten_query=rewritten,
        timings_ms=timings,
        answer=str(answer or "").strip(),
        chunks=[_chunk_brief(c) for c in (chunks or [])],
        file_distribution=_file_distribution(chunks or [], limit=12),
    )
    return item


def _write_markdown(
    *,
    path: Path,
    title: str,
    meta: Dict[str, Any],
    runs_by_variant: Dict[str, List[RunItem]],
) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    for variant, runs in runs_by_variant.items():
        lines.append(f"## Variant: {variant}")
        lines.append("")
        for i, item in enumerate(runs, start=1):
            lines.append(f"### Q{i}")
            lines.append("")
            lines.append(f"- query: {item.query}")
            if item.rewritten_query is not None:
                lines.append(f"- rewritten_query: {item.rewritten_query}")
            if item.timings_ms:
                lines.append(f"- timings_ms: {json.dumps(item.timings_ms, ensure_ascii=False)}")
            if item.file_distribution:
                lines.append("- file_distribution(top):")
                for row in item.file_distribution[:10]:
                    fid = row.get("source_file_id")
                    cnt = row.get("count")
                    fname = row.get("filename")
                    lines.append(f"  - {fid} x{cnt} {fname or ''}".rstrip())
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append("```")
            lines.append(item.answer)
            lines.append("```")
            lines.append("")
            lines.append("Top chunks (brief):")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item.chunks[:8], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dotenv", default=".env", help="Optional dotenv file to load before resolving config placeholders.")
    ap.add_argument("--owner-id", required=True, help="Owner UUID")
    ap.add_argument(
        "--config",
        default="config/json_configs/rag_inference.json",
        help="Path to rag_inference JSON config (env placeholders are supported).",
    )
    ap.add_argument("--questions-path", default=None, help="Path to a text/markdown file containing questions.")
    ap.add_argument("--question", action="append", default=[], help="Single question (repeatable).")
    ap.add_argument(
        "--modes",
        default="all,dense_bm25,graph",
        help="Retriever modes to run (comma-separated): all | dense_bm25 | graph",
    )
    ap.add_argument(
        "--graph-variants",
        default="graph_base,graph_optimized",
        help="Graph variants to run (comma-separated): graph_base | graph_optimized",
    )
    ap.add_argument(
        "--out-dir",
        default=f"local/runtime/rag_inference_eval_{_now_tag()}",
        help="Output directory (JSON + Markdown).",
    )
    ap.add_argument("--title", default="RAG Inference Eval", help="Report title.")
    args = ap.parse_args()

    _load_dotenv(str(args.dotenv))

    owner_id = uuid.UUID(str(args.owner_id))
    cfg_path = Path(str(args.config))
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    questions: List[str] = []
    if args.questions_path:
        questions.extend(_parse_questions(_read_text(Path(args.questions_path))))
    if args.question:
        questions.extend([str(q).strip() for q in (args.question or []) if str(q).strip()])
    # De-duplicate while preserving order.
    uniq: List[str] = []
    for q in questions:
        if q not in uniq:
            uniq.append(q)
    questions = uniq
    if not questions:
        raise SystemExit("No questions provided. Use --questions-path or --question.")

    base_cfg = _load_config_dict(cfg_path)

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    gvars = [v.strip() for v in str(args.graph_variants).split(",") if v.strip()]

    runs_by_variant: Dict[str, List[RunItem]] = {}
    raw_trace: Dict[str, Any] = {"meta": {}, "variants": {}}

    for mode in modes:
        mode_has_graph = mode in {"all", "graph"}
        gv_iter = gvars if mode_has_graph else ["nog"]
        for gv in gv_iter:
            # Compose a readable variant name.
            variant_name = f"{mode}+{gv}" if mode_has_graph else f"{mode}"
            cfg = json.loads(json.dumps(base_cfg))  # deep copy via json (config dict is JSON-friendly)
            _select_retrievers(cfg, mode=mode)
            # Only apply graph overrides when graph retriever is present.
            has_graph = any(
                isinstance(r, dict) and r.get("type") == "pruned_hipporag_neo4j_retrieval"
                for r in (cfg.get("retrieval_config", {}) or {}).get("retrievers", [])  # type: ignore[union-attr]
            )
            if has_graph:
                _apply_graph_overrides(cfg, graph_variant=gv)

            rag = _build_rag_inference(cfg)

            items: List[RunItem] = []
            for q in questions:
                items.append(_run_one(rag, query=q, owner_id=owner_id))
            runs_by_variant[variant_name] = items

            raw_trace["variants"][variant_name] = {
                "mode": mode,
                "graph_variant": gv,
                "questions": [it.query for it in items],
                "runs": [
                    {
                        "query": it.query,
                        "rewritten_query": it.rewritten_query,
                        "timings_ms": it.timings_ms,
                        "answer": it.answer,
                        "file_distribution": it.file_distribution,
                        "chunks": it.chunks,
                    }
                    for it in items
                ],
            }

    raw_trace["meta"] = {
        "owner_id": str(owner_id),
        "config_path": str(cfg_path),
        "modes": modes,
        "graph_variants": gvars,
        "questions_count": len(questions),
    }

    (out_dir / "trace.json").write_text(json.dumps(raw_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        path=out_dir / "report.md",
        title=str(args.title),
        meta=raw_trace["meta"],
        runs_by_variant=runs_by_variant,
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
