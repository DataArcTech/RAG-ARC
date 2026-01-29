import asyncio
import logging
from typing import Dict, Iterable, List, Optional

from config import pageindex as pageindex_cfg
from core.file_management.pageindex.types import SectionNode
from core.file_management.pageindex.utils import truncate_text_by_tokens
from core.prompts.pageindex import (
    SECTION_SUMMARY_SYSTEM_PROMPT,
    SECTION_SUMMARY_USER_PROMPT,
)

logger = logging.getLogger(__name__)


async def _call_llm_async(llm, messages, **kwargs) -> str:
    async_chat = getattr(llm, "achat", None)
    if callable(async_chat):
        return await async_chat(messages, **kwargs)
    chat = getattr(llm, "chat", None)
    if not callable(chat):
        raise RuntimeError("LLM connector does not expose chat/achat methods")
    return await asyncio.to_thread(chat, messages, **kwargs)


class SectionSummaryGenerator:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.model_override = pageindex_cfg.section_summary_model_name()
        self.max_tokens = pageindex_cfg.section_summary_max_tokens()
        self.top_k = pageindex_cfg.section_summary_top_k()
        self.max_concurrency = pageindex_cfg.section_summary_max_concurrency()
        self.leaf_chunk_max_chars = pageindex_cfg.section_summary_leaf_chunk_max_chars()

    def _resolve_model(self) -> Optional[str]:
        if self.model_override:
            return self.model_override
        cfg = getattr(self.llm, "config", None)
        low_cost = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
        token = str(low_cost or "").strip()
        return token or None

    def _leaf_text(self, chunks: List[dict]) -> str:
        if not chunks:
            return ""
        snippets: List[str] = []
        for chunk in chunks[: self.top_k]:
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            if self.leaf_chunk_max_chars > 0 and len(content) > self.leaf_chunk_max_chars:
                content = content[: self.leaf_chunk_max_chars].rstrip()
            snippets.append(content)
        return "\n\n".join(snippets).strip()

    def _child_summaries_text(self, children: Iterable[SectionNode]) -> str:
        lines: List[str] = []
        for child in children:
            summary = str(child.summary or "").strip()
            if not summary:
                continue
            title = str(child.title or "").strip()
            if title:
                lines.append(f"{title}: {summary}")
            else:
                lines.append(summary)
        return "\n".join(lines).strip()

    async def _summarize_node(self, node: SectionNode, *, content: str) -> Optional[str]:
        if not content:
            return None
        content = truncate_text_by_tokens(content, max_tokens=self.max_tokens)
        messages = [
            {"role": "system", "content": SECTION_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SECTION_SUMMARY_USER_PROMPT.format(
                    title=node.title,
                    path=node.path,
                    content=content,
                ),
            },
        ]
        model = self._resolve_model()
        kwargs = {"model": model} if model else {}
        try:
            response = await _call_llm_async(self.llm, messages, **kwargs)
            return str(response or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Section summary failed for %s: %s", node.section_id, exc)
            return None

    async def summarize(
        self,
        nodes: List[SectionNode],
        *,
        chunks_by_section: Dict[str, List[dict]],
    ) -> Dict[str, Optional[str]]:
        if not nodes:
            return {}
        if self.llm is None:
            logger.warning("Section summary skipped: LLM connector is missing")
            return {}

        node_map = {node.section_id: node for node in nodes}
        roots = [node for node in nodes if node.parent_id is None]

        depth_map: Dict[str, int] = {}
        stack: List[tuple[SectionNode, int]] = [(root, 1) for root in roots]
        while stack:
            node, depth = stack.pop()
            depth_map[node.section_id] = depth
            for child_id in node.children:
                child = node_map.get(child_id)
                if child is not None:
                    stack.append((child, depth + 1))

        max_depth = max(depth_map.values() or [1])
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run(node: SectionNode, content: str) -> None:
            async with semaphore:
                node.summary = await self._summarize_node(node, content=content)

        for depth in range(max_depth, 0, -1):
            tasks: List[asyncio.Task] = []
            for node in nodes:
                if depth_map.get(node.section_id) != depth:
                    continue
                children = [node_map[cid] for cid in node.children if cid in node_map]
                content = self._child_summaries_text(children)
                if not content:
                    content = self._leaf_text(chunks_by_section.get(node.section_id, []))
                if not content:
                    continue
                tasks.append(asyncio.create_task(_run(node, content)))
            if tasks:
                await asyncio.gather(*tasks)

        return {node.section_id: node.summary for node in nodes}
