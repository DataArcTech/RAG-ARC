import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import pageindex as pageindex_cfg
from core.file_management.pageindex.indexing import (
    PageIndexIndexers,
    resolve_base_bm25_config,
    resolve_base_faiss_config,
)
from core.file_management.pageindex.summary import SectionSummaryGenerator
from core.file_management.pageindex.tree_builder import SectionTreeBuildResult, build_section_tree
from core.file_management.pageindex.types import SectionNode
from core.file_management.pageindex.utils import normalize_for_match
from core.utils.path_guard import require_writable_dir
from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


@dataclass
class PageIndexContext:
    file_id: str
    filename: str
    markdown: str
    output_dir: Optional[str]
    md_path: Optional[str]
    tree: SectionTreeBuildResult
    nodes_by_id: Dict[str, SectionNode]
    normalized_page_texts: Dict[int, str]


def _build_tree_payload(nodes: List[SectionNode]) -> dict:
    node_map = {node.section_id: node for node in nodes}
    roots = [node for node in nodes if node.parent_id is None]

    def _serialize(node: SectionNode) -> dict:
        return {
            "section_id": node.section_id,
            "title": node.title,
            "path": node.path,
            "level": node.level,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "summary": node.summary,
            "children": [_serialize(node_map[cid]) for cid in node.children if cid in node_map],
        }

    return {"nodes": [_serialize(root) for root in roots]}


def _build_nodes_payload(nodes: List[SectionNode]) -> List[dict]:
    payload = []
    for node in nodes:
        payload.append(
            {
                "section_id": node.section_id,
                "file_id": node.file_id,
                "title": node.title,
                "path": node.path,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": list(node.children),
                "page_start": node.page_start,
                "page_end": node.page_end,
                "summary": node.summary,
                "level_source": node.level_source,
            }
        )
    return payload


class PageIndexService:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.summary_generator = SectionSummaryGenerator(llm) if llm else None

    def build_context(
        self,
        *,
        file_id: str,
        filename: str,
        markdown: str,
        md_path: Optional[str],
        output_dir: Optional[str],
    ) -> PageIndexContext:
        tree = build_section_tree(
            file_id=file_id,
            markdown=markdown,
            filename=filename,
            md_path=md_path,
            output_dir=output_dir,
        )
        normalized_page_texts = {
            page: normalize_for_match(text)
            for page, text in (tree.page_texts or {}).items()
        }
        return PageIndexContext(
            file_id=file_id,
            filename=filename,
            markdown=markdown,
            output_dir=output_dir,
            md_path=md_path,
            tree=tree,
            nodes_by_id={node.section_id: node for node in tree.nodes},
            normalized_page_texts=normalized_page_texts,
        )

    def _resolve_artifact_dir(self, context: PageIndexContext) -> Optional[Path]:
        if context.md_path:
            return Path(context.md_path).parent
        if context.output_dir:
            return Path(context.output_dir)
        return None

    def _write_artifacts(self, context: PageIndexContext) -> None:
        if not pageindex_cfg.pageindex_enabled():
            return
        artifact_dir = self._resolve_artifact_dir(context)
        if artifact_dir is None:
            return
        try:
            artifact_dir = Path(require_writable_dir(str(artifact_dir)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("PageIndex artifact dir not writable: %s", exc)
            return

        tree_path = artifact_dir / pageindex_cfg.pageindex_tree_filename()
        nodes_path = artifact_dir / pageindex_cfg.pageindex_nodes_filename()

        tree_payload = {
            "file_id": context.file_id,
            "filename": context.filename,
            "tree": _build_tree_payload(context.tree.nodes),
            "meta": {
                "level_conflict_ratio": context.tree.level_conflict_ratio,
                "uniform_level_flattened": context.tree.uniform_level_flattened,
            },
        }
        tree_path.write_text(json.dumps(tree_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        nodes_payload = _build_nodes_payload(context.tree.nodes)
        with nodes_path.open("w", encoding="utf-8") as handle:
            for row in nodes_payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _chunk_offset(self, markdown: str, chunk_text: str, *, cursor: int, snippet_chars: int) -> Optional[int]:
        if not chunk_text:
            return None
        pos = markdown.find(chunk_text, cursor)
        if pos != -1:
            return pos
        if snippet_chars <= 0:
            return None
        snippet = chunk_text[:snippet_chars]
        if snippet:
            pos = markdown.find(snippet, cursor)
            if pos != -1:
                return pos
        return None

    def _match_page(
        self,
        *,
        normalized_snippet: str,
        page_candidates: List[int],
        normalized_page_texts: Dict[int, str],
    ) -> Optional[int]:
        if not normalized_snippet:
            return None
        for page_idx in page_candidates:
            page_text = normalized_page_texts.get(page_idx, "")
            if normalized_snippet in page_text:
                return page_idx
        return None

    def enrich_chunks(
        self,
        *,
        context: PageIndexContext,
        chunks: List[dict],
    ) -> dict:
        if not pageindex_cfg.pageindex_enabled() or not chunks:
            return {"sections_matched": 0, "pages_matched": 0, "unmatched_chunks": 0}

        locator = context.tree.locator
        node_map = context.nodes_by_id
        cursor = 0
        matched_sections = 0
        matched_pages = 0
        unmatched = 0
        snippet_chars = pageindex_cfg.section_chunk_match_snippet_chars()
        page_snippet_chars = pageindex_cfg.section_page_match_snippet_chars()
        max_page_scan = pageindex_cfg.section_page_match_max_pages()

        for chunk in chunks:
            content = str(chunk.get("content") or "")
            pos = self._chunk_offset(context.markdown, content, cursor=cursor, snippet_chars=snippet_chars)
            if pos is not None:
                cursor = max(pos + len(content), cursor)
            else:
                unmatched += 1

            section_id = locator.resolve(pos if pos is not None else cursor)
            if section_id is None and context.tree.nodes:
                section_id = context.tree.nodes[0].section_id

            node = node_map.get(section_id) if section_id else None
            if node is None:
                # 未匹配到 section 时仍尝试赋予第一页，避免 page_start/page_end 恒为 null
                if context.tree.nodes and context.normalized_page_texts:
                    meta = chunk.get("metadata") or {}
                    first_node = context.tree.nodes[0]
                    meta["section_id"] = first_node.section_id
                    meta["section_path"] = first_node.path
                    meta["section_level"] = first_node.level
                    first_page = min(context.normalized_page_texts.keys())
                    meta["page_start"] = first_page
                    meta["page_end"] = first_page
                    chunk["metadata"] = meta
                continue

            meta = chunk.get("metadata") or {}
            meta["section_id"] = node.section_id
            meta["section_path"] = node.path
            meta["section_level"] = node.level
            meta["page_start"] = node.page_start
            meta["page_end"] = node.page_end
            chunk["metadata"] = meta
            matched_sections += 1

            if context.normalized_page_texts:
                page_candidates: List[int] = []
                if node.page_start is not None:
                    end = node.page_end if node.page_end is not None else node.page_start
                    page_candidates = list(range(node.page_start, end + 1))
                else:
                    page_candidates = sorted(context.normalized_page_texts.keys())
                if max_page_scan > 0 and len(page_candidates) > max_page_scan:
                    page_candidates = page_candidates[:max_page_scan]

                snippet_raw = content[:page_snippet_chars] if page_snippet_chars > 0 else content
                snippet = normalize_for_match(snippet_raw)
                matched_page = self._match_page(
                    normalized_snippet=snippet,
                    page_candidates=page_candidates,
                    normalized_page_texts=context.normalized_page_texts,
                )
                if matched_page is not None:
                    meta["page_start"] = matched_page
                    meta["page_end"] = matched_page
                    chunk["metadata"] = meta
                    matched_pages += 1

            # 若 section 有匹配但 page 仍为 None（content_list 无 page_idx 或 _match_page 未命中），
            # 且有页面文本，则用第一页作为回退，避免 SSE 返回 page_start/page_end 恒为 null
            if meta.get("page_start") is None and meta.get("page_end") is None and context.normalized_page_texts:
                first_page = min(context.normalized_page_texts.keys())
                meta["page_start"] = first_page
                meta["page_end"] = first_page
                chunk["metadata"] = meta

        return {"sections_matched": matched_sections, "pages_matched": matched_pages, "unmatched_chunks": unmatched}

    async def summarize_sections(
        self,
        *,
        context: PageIndexContext,
        chunks: List[dict],
    ) -> dict:
        if not pageindex_cfg.section_summary_enabled() or not context.tree.nodes:
            return {"summaries": 0}
        if self.summary_generator is None:
            logger.warning("Section summary generator missing; skipping summaries")
            return {"summaries": 0}

        chunks_by_section: Dict[str, List[dict]] = {}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            section_id = str(meta.get("section_id") or "").strip()
            if not section_id:
                continue
            chunks_by_section.setdefault(section_id, []).append(chunk)

        await self.summary_generator.summarize(context.tree.nodes, chunks_by_section=chunks_by_section)
        summary_count = sum(1 for node in context.tree.nodes if node.summary)
        return {"summaries": summary_count}

    def _section_chunks(self, *, context: PageIndexContext, owner_id: Optional[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for node in context.tree.nodes:
            parts = [str(node.title or "").strip(), str(node.path or "").strip(), str(node.summary or "").strip()]
            content = "\n".join([p for p in parts if p]).strip()
            if not content:
                continue
            metadata = {
                "section_id": node.section_id,
                "section_path": node.path,
                "section_level": node.level,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "source_file_id": context.file_id,
                "filename": context.filename,
                "owner_id": owner_id,
            }
            chunks.append(
                Chunk(
                    id=node.section_id,
                    content=content,
                    owner_id=owner_id,
                    metadata=metadata,
                )
            )
        return chunks

    async def build_indexes(
        self,
        *,
        context: PageIndexContext,
        owner_id: Optional[str],
        base_indexers: List[Any],
    ) -> dict:
        if not pageindex_cfg.pageindex_enabled():
            return {}
        base_faiss = resolve_base_faiss_config(base_indexers)
        base_bm25 = resolve_base_bm25_config(base_indexers)
        indexers = PageIndexIndexers(base_faiss_config=base_faiss, base_bm25_config=base_bm25)

        results: dict[str, Any] = {}
        section_chunks = self._section_chunks(context=context, owner_id=owner_id)
        if section_chunks:
            results["section_index"] = await indexers.index_sections(section_chunks)

        self._write_artifacts(context)
        return results
