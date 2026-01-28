import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional

from config import pageindex as pageindex_cfg
from core.file_management.pageindex.types import DocDescription, DocProfile, SectionNode
from core.file_management.pageindex.utils import truncate_text_by_tokens
from core.prompts.pageindex import (
    DOC_DESCRIPTION_SYSTEM_PROMPT,
    DOC_DESCRIPTION_USER_PROMPT,
    DOC_PROFILE_SYSTEM_PROMPT,
    DOC_PROFILE_USER_PROMPT,
    SECTION_SUMMARY_SYSTEM_PROMPT,
    SECTION_SUMMARY_USER_PROMPT,
)
from core.utils.json_extract import safe_json_loads

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


class DocDescriptionGenerator:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.model_override = pageindex_cfg.doc_desc_model_name()
        self.max_tokens = pageindex_cfg.doc_desc_max_tokens()

    def _resolve_model(self) -> Optional[str]:
        if self.model_override:
            return self.model_override
        cfg = getattr(self.llm, "config", None)
        low_cost = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
        token = str(low_cost or "").strip()
        return token or None

    async def build_description(self, *, file_id: str, filename: str, nodes: List[SectionNode]) -> Optional[DocDescription]:
        if self.llm is None:
            logger.warning("Doc description skipped: LLM connector is missing")
            return None

        top_level = [node for node in nodes if node.parent_id is None]
        lines: List[str] = []
        for node in top_level:
            summary = str(node.summary or "").strip()
            if not summary:
                continue
            title = str(node.title or "").strip()
            if title:
                lines.append(f"- {title}: {summary}")
            else:
                lines.append(f"- {summary}")

        if not lines:
            return None

        summaries = truncate_text_by_tokens("\n".join(lines), max_tokens=self.max_tokens)
        messages = [
            {"role": "system", "content": DOC_DESCRIPTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DOC_DESCRIPTION_USER_PROMPT.format(title=filename, summaries=summaries),
            },
        ]
        model = self._resolve_model()
        kwargs = {"model": model} if model else {}
        try:
            response = await _call_llm_async(self.llm, messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Doc description failed for %s: %s", file_id, exc)
            return None

        description = str(response or "").strip()
        if not description:
            return None
        return DocDescription(file_id=file_id, title=filename, description=description)


class DocProfileGenerator:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.model_override = pageindex_cfg.doc_profile_model_name()
        self.max_tokens = pageindex_cfg.doc_profile_max_tokens()
        self.max_list_items = pageindex_cfg.doc_profile_max_list_items()

    def _resolve_model(self) -> Optional[str]:
        if self.model_override:
            return self.model_override
        cfg = getattr(self.llm, "config", None)
        low_cost = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
        token = str(low_cost or "").strip()
        return token or None

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        token = str(value or "").strip()
        return token or None

    def _coerce_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [t.strip() for t in value.split(",") if t.strip()]
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = [str(v or "").strip() for v in value if str(v or "").strip()]
        else:
            items = [str(value).strip()] if str(value or "").strip() else []
        deduped: List[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if self.max_list_items and len(deduped) >= self.max_list_items:
                break
        return deduped

    async def build_profile(self, *, file_id: str, filename: str, nodes: List[SectionNode]) -> Optional[DocProfile]:
        if not pageindex_cfg.doc_profile_enabled():
            return None
        if self.llm is None:
            logger.warning("Doc profile skipped: LLM connector is missing")
            return None

        top_level = [node for node in nodes if node.parent_id is None]
        lines: List[str] = []
        for node in top_level:
            summary = str(node.summary or "").strip()
            if not summary:
                continue
            title = str(node.title or "").strip()
            if title:
                lines.append(f"- {title}: {summary}")
            else:
                lines.append(f"- {summary}")
        if not lines:
            return None

        summaries = truncate_text_by_tokens("\n".join(lines), max_tokens=self.max_tokens)
        messages = [
            {"role": "system", "content": DOC_PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": DOC_PROFILE_USER_PROMPT.format(title=filename, summaries=summaries)},
        ]
        model = self._resolve_model()
        kwargs = {"model": model} if model else {}
        try:
            raw = await _call_llm_async(self.llm, messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Doc profile failed for %s: %s", file_id, exc)
            return None

        payload = safe_json_loads(str(raw or ""), expected="dict")
        if not isinstance(payload, dict):
            logger.warning("Doc profile parse failed for %s (non-json)", file_id)
            return None

        return DocProfile(
            company=self._coerce_str(payload.get("company")),
            product=self._coerce_str(payload.get("product")),
            model=self._coerce_str(payload.get("model")),
            version=self._coerce_str(payload.get("version")),
            doc_type=self._coerce_str(payload.get("doc_type")),
            language=self._coerce_str(payload.get("language")),
            keywords=self._coerce_list(payload.get("keywords")),
            aliases=self._coerce_list(payload.get("aliases")),
        )
