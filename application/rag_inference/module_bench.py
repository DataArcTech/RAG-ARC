"""Benchmark-mode (experiment) HippoRAG Q&A runner.

This module is intentionally separate from `application/rag_inference/module.py` so
benchmark experiments can avoid product behaviors (rewrite/web search/user filters/streaming).
"""
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from encapsulation.data_model.schema import Chunk
from core.prompts.rag_inference_prompts import RAG_INFERENCE_BENCH_SYSTEM_PROMPT_EN
from config.benchmark_mode import benchmark_mode_enabled


@dataclass(frozen=True, slots=True)
class BenchRAGResult:
    answer: str
    chunks: List[Chunk]


class RAGInferenceBench:
    """Non-streaming benchmark runner built from an existing RAGInference instance."""

    def __init__(
        self,
        *,
        retriever: Any,
        reranker: Any,
        llm: Any,
        graph_candidates_k: int,
        rerank_keep_k: int,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.graph_candidates_k = int(graph_candidates_k)
        self.rerank_keep_k = int(rerank_keep_k)

    @classmethod
    def from_rag_inference(cls, rag_inference: Any) -> "RAGInferenceBench":
        candidate_cfg = getattr(getattr(rag_inference, "config", None), "candidate_selection", None)
        if candidate_cfg is None:
            raise ValueError("rag_inference.config.candidate_selection is required for benchmark runner")
        graph_candidates_k = int(getattr(candidate_cfg, "graph_candidates_k"))
        rerank_keep_k = int(getattr(candidate_cfg, "rerank_keep_k"))
        return cls(
            retriever=getattr(rag_inference, "retriever"),
            reranker=getattr(rag_inference, "reranker"),
            llm=getattr(rag_inference, "llm"),
            graph_candidates_k=graph_candidates_k,
            rerank_keep_k=rerank_keep_k,
        )

    def chat(
        self,
        query: str,
        *,
        owner_id: uuid.UUID,
        sources: Optional[Sequence[Chunk]] = None,
    ) -> BenchRAGResult:
        if not benchmark_mode_enabled():
            raise RuntimeError("bench_mode=1 is required to run RAGInferenceBench")

        question = str(query or "").strip()
        if not question:
            raise ValueError("query must be a non-empty string")

        chunks: List[Chunk]
        if sources is None:
            chunks = list(
                self.retriever.invoke(
                    question,
                    owner_id=owner_id,
                    return_subgraph_info=False,
                    k=self.graph_candidates_k,
                )
                or []
            )
        else:
            chunks = [c for c in sources if isinstance(c, Chunk)]

        chunks = list(self.reranker.rerank(question, chunks, top_k=self.rerank_keep_k) or [])

        messages: List[Dict[str, str]] = [{"role": "system", "content": RAG_INFERENCE_BENCH_SYSTEM_PROMPT_EN}]
        for i, chunk in enumerate(chunks):
            metadata = getattr(chunk, "metadata", None) or {}
            chunk_text = metadata.get("prompt_text") or metadata.get("index_text") or getattr(chunk, "content", "")
            filename = str(metadata.get("filename") or "").strip() or "source"
            messages.append({"role": "user", "content": f"Source key={i+1} title={filename}\n{chunk_text}"})
        messages.append({"role": "user", "content": f"Question: {question}"})

        answer = self.llm.chat(messages)
        return BenchRAGResult(answer=str(answer or "").strip(), chunks=chunks)
