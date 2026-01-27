#!/usr/bin/env bash
set -euo pipefail

# Full pipeline smoke test:
# user query -> intent routing -> rewrite -> retrieval/web -> rerank -> generation.
#
# Uses:
# - local Qwen embedding (intent routing only; RAG embedding is controlled by the selected RAG config)
# - existing knowledge for owner_id=00000000-0000-0000-0000-000000000001 (no re-upload)
# - test/test_pdf.pdf for quick local inspection (PDF->text preview only)
#
# Notes:
# - This script relies on your existing runtime services/configs (Faiss/BM25/Neo4j/etc).
# - Web search requires TAVILY_API_KEY to be configured; otherwise the WEB_ONLY path will run without Tavily chunks.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export INTENT_QWEN_EMBEDDING_MODEL_NAME="${INTENT_QWEN_EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
export INTENT_EMBEDDING_CACHE_FOLDER="${INTENT_EMBEDDING_CACHE_FOLDER:-./models/Qwen}"

uv run python - <<'PY'
import asyncio
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv

from config.application.rag_inference_config import RAGInferenceConfig
from framework.register import Register


OWNER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _preview_pdf(path: str) -> None:
    doc = fitz.open(path)
    text = []
    for i in range(min(2, doc.page_count)):
        text.append(doc.load_page(i).get_text("text"))
    joined = "\n".join(text).strip()
    snippet = (joined[:1200] + "...[truncated]") if len(joined) > 1200 else joined
    print(f"[pdf] path={path} pages={doc.page_count} preview_chars={len(snippet)}")
    print(snippet.replace("\r\n", "\n"))


def _history_text(turns: list[tuple[str, str]]) -> str | None:
    if not turns:
        return None
    return "\n".join(f"{r}: {c}" for r, c in turns if r and c) or None


def _coerce_file_id(meta: Any) -> str | None:
    if not isinstance(meta, dict):
        return None
    for key in ("source_file_id", "sourceFileId", "file_id", "fileId", "document_id", "documentId", "doc_id", "docId"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    return None


def _chunk_file_stats(chunks: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    ctr: Counter[str] = Counter()
    name_by_id: dict[str, str] = {}
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
    out: list[dict[str, Any]] = []
    for fid, count in ctr.most_common(limit):
        out.append({"source_file_id": fid, "count": int(count), "filename": name_by_id.get(fid)})
    return out


async def main() -> int:
    load_dotenv()  # load .env so JSON placeholders resolve
    # Keep output readable: suppress noisy INFO logs from the pipeline during e2e runs.
    logging.getLogger().setLevel(logging.WARNING)
    for name in [
        "application.rag_inference.module",
        "framework.register",
        "encapsulation.llm.utils.openai_client",
        "encapsulation.database.vector_db.faiss",
        "encapsulation.database.graph_db.pruned_hipporag_neo4j_cache",
        "encapsulation.database.graph_db.pruned_hipporag_neo4j",
        "core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_cache",
        "core.retrieval.graph_retrieveal.pruned_hipporag_neo4j",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    pdf_path = "test/test_pdf.pdf"
    if os.path.exists(pdf_path):
        _preview_pdf(pdf_path)
    else:
        print(f"[pdf] missing: {pdf_path} (skipping preview)")

    # Build rag_inference module from the same JSON config as runtime.
    # RAG uses remote models (OpenAI-compatible) by default; intent routing is configured separately (local Qwen).
    cfg_path = os.getenv("RAG_INFERENCE_CONFIG_PATH", "config/json_configs/rag_inference.json")
    reg = Register()
    t0 = time.perf_counter()
    reg.register(config_path=cfg_path, app_name="rag_inference_e2e", config_type=RAGInferenceConfig)
    rag = reg.get_object("rag_inference_e2e")
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[init] rag_inference config={cfg_path} ms={init_ms:.1f}")

    session_id = uuid.uuid4()
    turns = [
        ("为我分析下新加坡美国学校的优劣", False),
        ("详细展开说说", False),
        ("今天天气咋样", True),
        ("那新加坡的天气呢", True),
        ("如果我想要上新加坡美国学校，需要什么准备", False),
        ("谢谢", False),
    ]

    history: list[tuple[str, str]] = []
    for i, (q, enable_web) in enumerate(turns, start=1):
        events: list[dict[str, Any]] = []

        def progress_cb(ev: dict[str, Any]) -> None:
            # Keep small; we only need intent routing and stage timing.
            events.append(dict(ev))

        t1 = time.perf_counter()
        resp, chunks, *_rest = await rag.chat_async(
            query=q,
            owner_id=OWNER_ID,
            history_text=_history_text(history),
            enable_web_search=bool(enable_web),
            session_id=session_id,
            progress_callback=progress_cb,
        )
        dt_ms = (time.perf_counter() - t1) * 1000.0

        intent_ev = next((e for e in reversed(events) if e.get("stage") == "intent_routing" and e.get("status") == "end"), None)
        web_ev = next((e for e in reversed(events) if e.get("stage") == "web_search" and e.get("status") == "start"), None)
        rewrite_ev = next((e for e in reversed(events) if e.get("stage") == "rewrite" and e.get("status") == "end"), None)
        retrieve_ev = next((e for e in reversed(events) if e.get("stage") == "retrieve" and e.get("status") == "end"), None)
        web_end_ev = next((e for e in reversed(events) if e.get("stage") == "web_search" and e.get("status") == "end"), None)
        rerank_ev = next((e for e in reversed(events) if e.get("stage") == "rerank" and e.get("status") == "end"), None)

        print(f"\n[turn#{i}] enable_web_search={enable_web} dt_ms={dt_ms:.1f}")
        print(f"Q: {q}")
        if intent_ev:
            print(
                f"intent={intent_ev.get('intent')} action={intent_ev.get('action')} score={intent_ev.get('score')} topic={intent_ev.get('topic')}"
            )
            if "duration_ms" in intent_ev:
                print(f"intent_ms={intent_ev.get('duration_ms')}")
        else:
            print("intent=<missing_progress_event>")
        if rewrite_ev:
            print(f"rewrite_ms={rewrite_ev.get('duration_ms')} rewritten_query={rewrite_ev.get('rewritten_query')!r}")
        if web_ev:
            print(f"web_search_started provider={web_ev.get('provider')} max_results={web_ev.get('max_results')}")
        if web_end_ev:
            print(f"web_search_results={web_end_ev.get('results')} web_ms={web_end_ev.get('duration_ms')}")
        if retrieve_ev:
            print(f"retrieve_ms={retrieve_ev.get('duration_ms')}")
        if rerank_ev:
            print(f"rerank_ms={rerank_ev.get('duration_ms')} chunks_out={rerank_ev.get('chunks_out')}")
        print(f"chunks={len(chunks)} top_files={_chunk_file_stats(chunks)}")
        # Don't dump full response; just show a short prefix.
        resp_preview = (resp[:240] + "...[truncated]") if isinstance(resp, str) and len(resp) > 240 else resp
        print(f"answer_preview={resp_preview!r}")

        history.append(("user", q))
        history.append(("assistant", resp if isinstance(resp, str) else str(resp)))

    return 0


raise SystemExit(asyncio.run(main()))
PY

