"""
Evaluate intent-aware rewrite + anchored evidence filtering on drift examples.

This script is designed to make "data speak" without relying on full answer generation:
- Uses the query rewriter to produce: intent + rewritten_query + anchors (single LLM call).
- Runs retrieval (MultiPath) and prints file distributions before/after anchor-based filtering.

Notes:
- No regex-based patching in this script.
- If you want to avoid heavy graph dependencies, use --mode dense_bm25.
"""
import argparse
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framework.register import Register


def _load_dotenv(path: str) -> None:
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _substitute_env(cfg: dict[str, Any]) -> dict[str, Any]:
    reg = Register()
    return reg._substitute_env_vars(cfg)  # type: ignore[attr-defined]


def _resolve_owner_id() -> uuid.UUID:
    raw = os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID") or os.getenv("SHARE_OWNER_ID")
    if raw:
        try:
            return uuid.UUID(str(raw))
        except ValueError:
            pass
    return uuid.uuid4()


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def _chunk_filename(meta: Any) -> str:
    if not isinstance(meta, dict):
        return "unknown"
    for key in ("relative_path", "path", "filepath", "file_path"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    token = str(meta.get("filename") or "").strip()
    return token or "unknown"


def _top_filenames(chunks: list[Any], *, limit: int = 8) -> list[tuple[str, int]]:
    from collections import Counter

    ctr: Counter[str] = Counter()
    for ch in chunks or []:
        meta = getattr(ch, "metadata", None) or {}
        ctr[_chunk_filename(meta)] += 1
    return ctr.most_common(max(int(limit), 0))


def _preview(chunks: list[Any], *, limit: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in (chunks or [])[: max(int(limit), 0)]:
        meta = getattr(ch, "metadata", None) or {}
        out.append(
            {
                "filename": _chunk_filename(meta),
                "score": meta.get("score"),
                "preview": _coerce_text(getattr(ch, "content", ""))[:120].replace("\n", " "),
            }
        )
    return out


@dataclass(frozen=True, slots=True)
class EvalTurn:
    name: str
    history_text: str
    query: str


DEFAULT_TURNS: list[EvalTurn] = [
    EvalTurn(
        name="case1_followup_beneficiary_change",
        history_text=(
            "user: 財富Upgrade人生盛利产品，是什么币种的保单？\n"
            "assistant: （上一轮回答略）\n"
            "user: 財富Upgrade人生盛利适合什么样的客户投保？\n"
            "assistant: （上一轮回答略）\n"
        ),
        query="更换被保人有什么要求",
    ),
    EvalTurn(
        name="case1_topic_switch_star_diamond",
        history_text=(
            "user: 財富Upgrade人生盛利产品，是什么币种的保单？\n"
            "assistant: （上一轮回答略）\n"
            "user: 財富Upgrade人生盛利适合什么样的客户投保？\n"
            "assistant: （上一轮回答略）\n"
            "user: 更换被保人有什么要求\n"
            "assistant: （上一轮回答略）\n"
        ),
        query="星钻储蓄险有什么特点",
    ),
    EvalTurn(
        name="case2_clarification_turn",
        history_text=(
            "user: 保費繳付期选择几年最划算\n"
            "assistant: （回答略）\n"
            "user: 还有8年、12年，怎么没有测算\n"
            "assistant: （回答略）\n"
        ),
        query="可以，合理，那你应该在刚才回答的时候补充说明这一点",
    ),
    EvalTurn(
        name="case2_correction_turn",
        history_text=(
            "user: 保費繳付期选择几年最划算\n"
            "assistant: （回答略）\n"
            "user: 还有8年、12年，怎么没有测算\n"
            "assistant: （回答略）\n"
            "user: 可以，合理，那你应该在刚才回答的时候补充说明这一点\n"
            "assistant: （回答略）\n"
        ),
        query="你回答错了，我在问安达人寿的产品，你回答成了苏黎世保险",
    ),
    EvalTurn(
        name="case2_followup_discount_after_correction",
        history_text=(
            "user: 保費繳付期选择几年最划算\n"
            "assistant: （回答略）\n"
            "user: 还有8年、12年，怎么没有测算\n"
            "assistant: （回答略）\n"
            "user: 可以，合理，那你应该在刚才回答的时候补充说明这一点\n"
            "assistant: （回答略）\n"
            "user: 你回答错了，我在问安达人寿的产品，你回答成了苏黎世保险\n"
            "assistant: 抱歉，我理解错了。我们回到安达人寿的产品。\n"
        ),
        query="如果选择按年缴费， 是否有保单优惠",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dotenv", default=".env", help="Path to .env (default: .env)")
    ap.add_argument("--config", default="config/json_configs/rag_inference.json", help="RAG config JSON path")
    ap.add_argument("--k", type=int, default=30, help="Top-k retrieval candidates (default: 30)")
    ap.add_argument(
        "--mode",
        choices=("all", "dense_bm25", "graph"),
        default="dense_bm25",
        help="Which retrievers to include (default: dense_bm25)",
    )
    args = ap.parse_args()

    _load_dotenv(args.dotenv)
    # Enable the new behavior for evaluation.
    os.environ["RAG_INTENT_ROUTING_ENABLED"] = "1"
    os.environ["RAG_EVIDENCE_CONSISTENCY_ENABLED"] = "1"

    cfg = _substitute_env(_read_json(Path(args.config)))
    owner_id = _resolve_owner_id()

    # Build query rewriter and retriever from config (avoid full app wiring for experiments).
    from config.core.query_rewrite_config import LLMQueryRewriterConfig
    from config.core.retrieval.multipath_config import MultiPathRetrieverConfig

    rewriter = LLMQueryRewriterConfig(**(cfg.get("query_rewrite_config") or {})).build()

    retrieval_cfg = cfg.get("retrieval_config") or {}
    retrievers = list(retrieval_cfg.get("retrievers") or [])
    if args.mode == "dense_bm25":
        retrievers = [r for r in retrievers if isinstance(r, dict) and r.get("type") in {"dense", "tantivy_bm25"}]
    elif args.mode == "graph":
        retrievers = [r for r in retrievers if isinstance(r, dict) and r.get("type") == "pruned_hipporag_neo4j_retrieval"]
    retrieval_cfg["retrievers"] = retrievers
    retriever = MultiPathRetrieverConfig(**retrieval_cfg).build()

    from core.utils.evidence_consistency import filter_chunks_by_anchors
    from config.rag_intent_routing import rag_evidence_min_keep

    print(f"owner_id={owner_id} mode={args.mode} k={args.k}")

    for turn in DEFAULT_TURNS:
        payload = rewriter.rewrite_query_with_intent(turn.query, history_text=turn.history_text)
        intent = payload.get("intent")
        anchors = payload.get("anchors") or []
        rewritten_query = payload.get("rewritten_query") or turn.query

        chunks = retriever.invoke(str(rewritten_query), owner_id=str(owner_id), k=int(args.k)) or []
        filtered, info = filter_chunks_by_anchors(
            chunks=chunks,
            anchors=list(anchors) if isinstance(anchors, list) else [],
            min_keep=rag_evidence_min_keep(),
        )

        print(f"\n=== {turn.name} ===")
        print("query=", turn.query)
        print("intent=", intent)
        print("anchors=", anchors)
        print("rewritten_query=", rewritten_query)
        print("retrieved_chunks=", len(chunks))
        print("retrieved_top_files=", _top_filenames(chunks))
        print("filtered_chunks=", len(filtered))
        print(
            "filter_passed=",
            getattr(info, "passed", None),
            "matched_by_filename=",
            getattr(info, "matched_by_filename", None),
            "matched_by_content=",
            getattr(info, "matched_by_content", None),
        )
        print("filtered_top_files=", _top_filenames(filtered))
        print("filtered_preview=", json.dumps(_preview(filtered), ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
