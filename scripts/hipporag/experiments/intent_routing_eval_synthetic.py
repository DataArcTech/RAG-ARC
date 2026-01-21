"""
Synthetic regression evaluation for intent-aware rewrite + evidence consistency filtering.

Focus:
- "Clear question" cases: when the user explicitly includes the company/product name,
  the new filtering should NOT harm recall (it should either keep everything or keep
  only the same on-topic file set).
- "Drift" cases: underspecified follow-ups should stay anchored to the user's subject.

This script is intentionally lightweight and prints distributions ("data speaks")
without relying on full answer generation.
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


@dataclass(frozen=True, slots=True)
class EvalTurn:
    name: str
    history_text: str
    query: str
    expectation: str


DEFAULT_TURNS: list[EvalTurn] = [
    # Clear question: explicit product name should make retrieval stable.
    EvalTurn(
        name="clear_andar_product_pay_terms",
        history_text="",
        query="安達傳承守創儲蓄保障計劃 III 的保費繳付期有哪些？",
        expectation="clear_question_should_not_hurt_recall",
    ),
    EvalTurn(
        name="clear_axa_shengli_currency",
        history_text="",
        query="安盛 盛利Ⅱ儲蓄保險（至尊）是什么币种的保单？",
        expectation="clear_question_should_not_hurt_recall",
    ),
    # Drift regression: underspecified follow-up should stay on the previously discussed subject.
    EvalTurn(
        name="drift_followup_beneficiary_change",
        history_text=(
            "user: 財富Upgrade人生盛利产品，是什么币种的保单？\n"
            "assistant: （上一轮回答略）\n"
            "user: 財富Upgrade人生盛利适合什么样的客户投保？\n"
            "assistant: （上一轮回答略）\n"
        ),
        query="更换被保人有什么要求？",
        expectation="should_anchor_to_fortune_upgrade_shengli",
    ),
    # Topic switch: new subject should not inherit prior anchors.
    EvalTurn(
        name="topic_switch_star_diamond_explicit",
        history_text=(
            "user: 財富Upgrade人生盛利产品，是什么币种的保单？\n"
            "assistant: （上一轮回答略）\n"
            "user: 更换被保人有什么要求？\n"
            "assistant: （上一轮回答略）\n"
        ),
        query="我们换个话题：星钻储蓄险有什么特点？",
        expectation="topic_switch_should_not_inherit_old_anchors",
    ),
    # Clarification: should route to conversation-only (no retrieval needed).
    EvalTurn(
        name="clarification_should_skip_retrieval",
        history_text=(
            "user: 保費繳付期选择几年最划算\n"
            "assistant: （回答略）\n"
            "user: 还有8年、12年，怎么没有测算\n"
            "assistant: （回答略）\n"
        ),
        query="可以，合理，那你应该在刚才回答的时候补充说明这一点",
        expectation="clarification_skip_retrieval",
    ),
    # Correction: anchors should reflect only the intended subject, not the wrong one.
    EvalTurn(
        name="correction_both_subjects_only_keep_intended",
        history_text=(
            "user: 我在问安达人寿的产品\n"
            "assistant: （回答略）\n"
        ),
        query="你回答错了，我在问安达人寿的产品，你回答成了苏黎世保险。请回到安达人寿。",
        expectation="correction_should_keep_andar_only",
    ),
    # Follow-up after correction: should stay on corrected subject even if query omits company.
    EvalTurn(
        name="followup_after_correction_should_stay_andar",
        history_text=(
            "user: 你回答错了，我在问安达人寿的产品，你回答成了苏黎世保险。请回到安达人寿。\n"
            "assistant: 抱歉，我理解错了。我们回到安达人寿的产品。\n"
        ),
        query="如果选择按年缴费，是否有保单优惠？",
        expectation="should_anchor_to_andar",
    ),
    # Chitchat ack: should route to conversation-only.
    EvalTurn(
        name="chitchat_ack_skip_retrieval",
        history_text="user: 好的\nassistant: （回答略）\n",
        query="谢谢！",
        expectation="chitchat_skip_retrieval",
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
    os.environ["RAG_INTENT_ROUTING_ENABLED"] = "1"
    os.environ["RAG_EVIDENCE_CONSISTENCY_ENABLED"] = "1"

    cfg = _substitute_env(_read_json(Path(args.config)))
    owner_id = _resolve_owner_id()

    from config.core.query_rewrite_config import LLMQueryRewriterConfig
    from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
    from core.utils.evidence_consistency import filter_chunks_by_anchors
    from config.rag_intent_routing import rag_evidence_min_keep

    rewriter = LLMQueryRewriterConfig(**(cfg.get("query_rewrite_config") or {})).build()

    retrieval_cfg = cfg.get("retrieval_config") or {}
    retrievers = list(retrieval_cfg.get("retrievers") or [])
    if args.mode == "dense_bm25":
        retrievers = [r for r in retrievers if isinstance(r, dict) and r.get("type") in {"dense", "tantivy_bm25"}]
    elif args.mode == "graph":
        retrievers = [r for r in retrievers if isinstance(r, dict) and r.get("type") == "pruned_hipporag_neo4j_retrieval"]
    retrieval_cfg["retrievers"] = retrievers
    retriever = MultiPathRetrieverConfig(**retrieval_cfg).build()

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
        print("expectation=", turn.expectation)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

