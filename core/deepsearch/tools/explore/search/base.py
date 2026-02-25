"""Shared helpers for search tools."""
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from config.core.deepsearch import tool_defaults
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.data_model.schema import Chunk
from core.deepsearch.utils.evidence_ids import hashed_chunk_id
from core.deepsearch.utils.file_scope import FileScope, chunk_in_scope
from core.deepsearch.tooling.file_scope_policy import strip_file_scope_from_graph_context
from core.deepsearch.utils.ids import coerce_uuid_list
from core.deepsearch.utils.owner_visibility import OwnerVisibilityResolution, resolve_owner_visibility
from core.deepsearch.utils.query_clean import clean_query
from core.graph_adapter.base import GraphDeepSearchAdapter
from framework.register import Register

from ...base import ToolRunRequest

_CHANNEL_DEFAULTS = tuple(tool_defaults.SEARCH_DEFAULT_CHANNELS)
# Allow "graph" as a user-friendly alias of the internal "graph_chunk" channel.
_ALLOWED_CHANNELS = frozenset(set(_CHANNEL_DEFAULTS) | {"graph"})


def resolve_explicit_file_scope(extra: Mapping[str, Any] | None) -> tuple[FileScope | None, List[str], List[str]]:
    """Resolve explicit file_id/file_ids from tool args only (UUID-only)."""
    payload = dict(extra or {})
    raw_ids: List[str] = []
    seen: set[str] = set()
    for key in ("file_ids", "file_id", "source_file_ids", "source_file_id"):
        raw = payload.get(key)
        if raw is None:
            continue
        items = raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
        for item in items:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            raw_ids.append(token)
    file_ids, invalid = coerce_uuid_list(raw_ids)
    if not file_ids:
        return None, invalid, raw_ids
    scope = FileScope(file_ids=frozenset(file_ids), filename_contains=(), source="tool_args")
    return scope, invalid, raw_ids


@dataclass(frozen=True)
class _RetrieverBundle:
    dense: Any | None
    bm25: Any | None


@dataclass
class _ChannelResult:
    channel: str
    evidences: List[EvidenceChunk]
    diagnostics: Dict[str, Any]
    summary: str


class _SearchToolBase:
    def __init__(
        self,
        *,
        llm_connector=None,
        dense_retriever=None,
        bm25_retriever=None,
    ) -> None:
        self.llm_connector = llm_connector
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._retrievers_resolved = False

    @staticmethod
    def _require_adapter(adapter: GraphDeepSearchAdapter | None) -> GraphDeepSearchAdapter:
        if adapter is None:
            raise RuntimeError("graph_chunk channel requires a GraphDeepSearchAdapter instance")
        return adapter

    def _resolve_retrievers(self) -> _RetrieverBundle:
        if self._retrievers_resolved:
            return _RetrieverBundle(self._dense_retriever, self._bm25_retriever)
        self._retrievers_resolved = True

        if self._dense_retriever is not None or self._bm25_retriever is not None:
            return _RetrieverBundle(self._dense_retriever, self._bm25_retriever)

        try:
            registrator = Register()
            rag_module = getattr(registrator, "registrations", {}).get("rag_inference")
        except Exception:
            rag_module = None

        retriever = getattr(rag_module, "retriever", None) if rag_module is not None else None
        candidates: List[Any] = []
        if retriever is not None:
            built = getattr(getattr(retriever, "config", None), "built_retrievers", None)
            if isinstance(built, list) and built:
                candidates.extend(built)
            else:
                candidates.append(retriever)

        dense = None
        bm25 = None
        for candidate in candidates:
            cfg = getattr(candidate, "config", None)
            cfg_type = str(getattr(cfg, "type", "") or "")
            class_name = candidate.__class__.__name__
            if dense is None and (cfg_type == "dense" or class_name == "DenseRetriever"):
                dense = candidate
            if bm25 is None and (cfg_type == "tantivy_bm25" or class_name == "TantivyBM25Retriever"):
                bm25 = candidate
        self._dense_retriever = dense
        self._bm25_retriever = bm25
        return _RetrieverBundle(dense, bm25)

    @staticmethod
    def _resolve_query(request: ToolRunRequest) -> str:
        max_chars = int(tool_defaults.SEARCH_DEFAULT_QUERY_MAX_CHARS)
        extra = request.extra or {}
        focus = extra.get("focus_query") or extra.get("query") or request.question
        return clean_query(str(focus or ""), max_chars=max_chars) or str(focus or "").strip()

    def _resolve_query_variants(self, query: str, *, cache_scope: str | None = None) -> List[str]:
        try:
            from core.utils.query_variants import generate_query_variants

            variants = generate_query_variants(query, llm_connector=self.llm_connector, cache_scope=cache_scope)
        except Exception:  # noqa: BLE001
            variants = [str(query or "").strip()]

        cleaned: List[str] = []
        seen: set[str] = set()
        for item in variants or []:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            cleaned.append(token)
        if not cleaned:
            token = str(query or "").strip()
            if token:
                cleaned = [token]
        return cleaned

    @staticmethod
    def _resolve_owner_id(request: ToolRunRequest) -> Optional[str]:
        scope = request.access_scope
        if scope is not None and scope.scope_id:
            return str(scope.scope_id)
        graph_context = request.graph_context
        if graph_context is not None and graph_context.access_scope:
            return str(graph_context.access_scope.scope_id)
        if graph_context is not None and graph_context.owner_id:
            return str(graph_context.owner_id)
        return None

    @staticmethod
    def _resolve_owner_visibility(request: ToolRunRequest) -> OwnerVisibilityResolution:
        return resolve_owner_visibility(
            extra=(request.extra or {}),
            access_scope=request.access_scope,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
        )

    @staticmethod
    def _resolve_channels(extra: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
        raw = extra.get("channels")
        if raw is None:
            return list(_CHANNEL_DEFAULTS), []
        channels: List[str] = []
        if isinstance(raw, str):
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
        elif isinstance(raw, (list, tuple)):
            tokens = [str(t).strip() for t in raw if str(t).strip()]
        else:
            tokens = []
        unknown: List[str] = []
        for token in tokens:
            key = token.lower()
            if key in {"none", "null"}:
                continue
            if key == "graph":
                key = "graph_chunk"
            if key in _ALLOWED_CHANNELS:
                channels.append(key)
            else:
                unknown.append(token)
        return channels, unknown

    @staticmethod
    def _resolve_top_k(value: Any, default: int) -> int:
        try:
            parsed = int(value) if value is not None else int(default)
        except Exception:
            parsed = int(default)
        return max(0, parsed)

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        token = str(value).strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _chunk_id(chunk: Chunk | Mapping[str, Any], fallback_prefix: str) -> str:
        if isinstance(chunk, Chunk):
            meta = getattr(chunk, "metadata", None) or {}
            chunk_id = chunk.id or meta.get("chunk_id") or meta.get("id")
        else:
            meta = chunk.get("metadata") if isinstance(chunk, dict) else {}
            chunk_id = (
                (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
                or (chunk.get("id") if isinstance(chunk, dict) else None)
                or (meta.get("chunk_id") if isinstance(meta, dict) else None)
                or (meta.get("id") if isinstance(meta, dict) else None)
            )
        if chunk_id:
            return str(chunk_id)
        content = ""
        if isinstance(chunk, Chunk):
            content = chunk.content
        elif isinstance(chunk, dict):
            content = str(chunk.get("content") or "")
        return hashed_chunk_id(source=fallback_prefix, content=content, prefix=fallback_prefix)

    @staticmethod
    def _chunk_meta(chunk: Chunk | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(chunk, Chunk):
            return dict(getattr(chunk, "metadata", None) or {})
        if isinstance(chunk, dict) and isinstance(chunk.get("metadata"), dict):
            return dict(chunk.get("metadata") or {})
        return {}

    @staticmethod
    def _chunk_score(chunk: Chunk | Mapping[str, Any]) -> Optional[float]:
        if isinstance(chunk, Chunk):
            meta = getattr(chunk, "metadata", None) or {}
            score = meta.get("score")
        else:
            score = None
            if isinstance(chunk, dict):
                score = chunk.get("score")
                if score is None and isinstance(chunk.get("metadata"), dict):
                    score = chunk["metadata"].get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None

    @staticmethod
    def _chunk_content(chunk: Chunk | Mapping[str, Any]) -> str:
        if isinstance(chunk, Chunk):
            return str(chunk.content or "")
        if isinstance(chunk, dict):
            return str(chunk.get("content") or "")
        return ""

    @staticmethod
    def _chunk_file_name(meta: Mapping[str, Any]) -> Optional[str]:
        for key in ("source_file_name", "file_name", "filename", "source_path", "path", "source_file_id", "file_id"):
            token = meta.get(key)
            if token:
                return str(token)
        return None

    @staticmethod
    def _summary_window(text: str) -> str:
        raw = str(text or "")
        tokens = raw.split()
        if not tokens:
            return raw.strip()
        max_tokens = max(1, int(tool_defaults.SEARCH_SUMMARY_MAX_TOKENS))
        if len(tokens) <= max_tokens:
            return raw.strip()

        head_ratio = float(tool_defaults.SEARCH_SUMMARY_HEAD_RATIO)
        mid_ratio = float(tool_defaults.SEARCH_SUMMARY_MID_RATIO)
        tail_ratio = float(tool_defaults.SEARCH_SUMMARY_TAIL_RATIO)
        separator = str(tool_defaults.SEARCH_SUMMARY_SEPARATOR or " ... ")

        total = min(len(tokens), max_tokens)
        head_count = max(1, int(round(total * head_ratio)))
        mid_count = max(0, int(round(total * mid_ratio)))
        tail_count = max(0, total - head_count - mid_count)
        if head_count + mid_count + tail_count < total:
            tail_count += total - (head_count + mid_count + tail_count)

        head_tokens = tokens[:head_count]
        tail_tokens = tokens[-tail_count:] if tail_count else []

        mid_tokens: List[str] = []
        if mid_count:
            mid_start = max(head_count, int((len(tokens) - mid_count) / 2))
            mid_end = min(len(tokens) - tail_count, mid_start + mid_count)
            if mid_end > mid_start:
                mid_tokens = tokens[mid_start:mid_end]

        segments = [" ".join(head_tokens)]
        if mid_tokens:
            segments.append(" ".join(mid_tokens))
        if tail_tokens:
            segments.append(" ".join(tail_tokens))
        return separator.join([seg for seg in segments if seg.strip()])

    @staticmethod
    def _apply_file_scope(chunks: List[Chunk], file_scope) -> Tuple[List[Chunk], int]:
        if not file_scope or not file_scope.enabled:
            return chunks, 0
        filtered: List[Chunk] = []
        dropped = 0
        for chunk in chunks:
            meta = getattr(chunk, "metadata", None) or {}
            if chunk_in_scope(chunk_metadata=meta, scope=file_scope):
                filtered.append(chunk)
            else:
                dropped += 1
        return filtered, dropped

    @staticmethod
    def _resolve_section_scope(extra: Mapping[str, Any] | None) -> frozenset[str]:
        payload = dict(extra or {})
        raw = payload.get("section_ids")
        if raw is None:
            raw = payload.get("section_id")
        if raw is None:
            return frozenset()
        if isinstance(raw, (list, tuple, set, frozenset)):
            items = raw
        else:
            items = [raw]
        out: set[str] = set()
        for item in items:
            token = str(item or "").strip()
            if token:
                out.add(token)
        return frozenset(out)

    @staticmethod
    def _apply_section_scope(chunks: List[Chunk], section_ids: frozenset[str]) -> Tuple[List[Chunk], int]:
        if not section_ids:
            return chunks, 0
        kept: List[Chunk] = []
        dropped = 0
        for chunk in chunks:
            meta = getattr(chunk, "metadata", None) or {}
            sid = str(meta.get("section_id") or "").strip()
            if sid and sid in section_ids:
                kept.append(chunk)
            else:
                dropped += 1
        return kept, dropped

    @staticmethod
    def _low_cost_model_name(llm_connector) -> Optional[str]:
        cfg = getattr(llm_connector, "config", None)
        token = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
        token = str(token or "").strip()
        return token or None
