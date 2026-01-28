import logging
from typing import Any, Callable, Dict, Optional, Type

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.deepsearch.state import DeepSearchState
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.scope_provider import require_scope
from core.utils.owner_guard import get_share_owner_id, is_admin_owner, normalize_owner_id

logger = logging.getLogger(__name__)


class DeepSearchServiceContextMixin:
    def _get_pageindex_retriever(self):
        from config import pageindex as pageindex_cfg
        from core.retrieval.pageindex_retriever import PageIndexRetriever

        if not pageindex_cfg.pageindex_enabled():
            return None
        existing = getattr(self, "_pageindex_retriever", None)
        if existing is not None:
            return existing
        try:
            retriever = PageIndexRetriever()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSearch PageIndex retriever unavailable: %s", exc)
            retriever = None
        setattr(self, "_pageindex_retriever", retriever)
        return retriever

    @staticmethod
    def _split_title_desc(text: str) -> tuple[str, str]:
        raw = str(text or "").strip()
        if not raw:
            return "", ""
        lines = [line.rstrip() for line in raw.splitlines()]
        lines = [line for line in lines if line.strip()]
        if not lines:
            return "", ""
        title = lines[0].strip()
        desc = "\n".join(lines[1:]).strip()
        return title, desc

    def _attach_doc_routing_hints(
        self,
        context: GraphQueryContext,
        *,
        question: str,
        scope: GraphAccessScope,
    ) -> GraphQueryContext:
        """Attach lightweight doc-routing candidates before initial_think.

        This enables initial_think to decide the default behavior for cross-document ambiguity:
        when the query lacks company/version and candidates conflict, set report_needed=false
        (non-deepsearch clarification branch).
        """

        metadata = dict(context.metadata or {})
        if metadata.get("doc_routing_candidates") is not None:
            return context

        owner_id = normalize_owner_id(scope.scope_id) if scope and scope.scope_id else None
        if not owner_id:
            return context

        retriever = self._get_pageindex_retriever()
        if retriever is None:
            metadata["doc_routing_candidates_error"] = "pageindex_retriever_unavailable"
            try:
                return context.model_copy(update={"metadata": metadata})
            except AttributeError:
                payload = context.model_dump(exclude_none=True)
                payload["metadata"] = metadata
                return GraphQueryContext(**payload)

        # Default to checking primary + share (when configured). Admin keeps primary only here.
        owners = [owner_id]
        if not is_admin_owner(owner_id):
            share = get_share_owner_id()
            if share and share not in owners:
                owners.append(share)

        query = str(question or "").strip()
        candidates: dict[str, dict[str, Any]] = {}
        try:
            from config import pageindex as pageindex_cfg

            top_k = int(pageindex_cfg.doc_top_k() or 5)
            cand_k = int(pageindex_cfg.doc_retrieval_candidates_k() or 10)
        except Exception:
            top_k, cand_k = 5, 10

        for oid in owners:
            try:
                hits = retriever.retrieve_doc_chunks(query, owner_id=oid, k_final=top_k, k_candidates=cand_k)
            except Exception as exc:  # noqa: BLE001
                metadata.setdefault("doc_routing_candidates_errors", []).append({"owner_id": oid, "error": str(exc)})
                continue
            for hit in hits:
                meta = getattr(hit, "metadata", None) or {}
                file_id = str(meta.get("source_file_id") or meta.get("file_id") or meta.get("doc_id") or getattr(hit, "id", "") or "").strip()
                if not file_id:
                    continue
                score = meta.get("score")
                score_f = float(score) if isinstance(score, (int, float)) else 0.0
                filename = str(meta.get("filename") or meta.get("source_file_name") or "").strip() or None
                title, desc = self._split_title_desc(getattr(hit, "content", ""))
                doc_profile = {
                    "company": str(meta.get("doc_profile_company") or "").strip() or None,
                    "product": str(meta.get("doc_profile_product") or "").strip() or None,
                    "model": str(meta.get("doc_profile_model") or "").strip() or None,
                    "version": str(meta.get("doc_profile_version") or "").strip() or None,
                    "doc_type": str(meta.get("doc_profile_doc_type") or "").strip() or None,
                }
                if not any(v for v in doc_profile.values()):
                    doc_profile = None

                existing = candidates.get(file_id)
                if existing is None or score_f > float(existing.get("score") or 0.0):
                    candidates[file_id] = {
                        "file_id": file_id,
                        "owner_id": oid,
                        "score": score_f,
                        "filename": filename,
                        "title": title or None,
                        "doc_description": (desc or "").strip() or None,
                        "doc_profile": doc_profile,
                    }

        merged = sorted(candidates.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)[:top_k]
        metadata["doc_routing_candidates"] = merged
        metadata["doc_routing_candidates_source"] = "preflight_pageindex"

        try:
            return context.model_copy(update={"metadata": metadata})
        except AttributeError:
            payload = context.model_dump(exclude_none=True)
            payload["metadata"] = metadata
            return GraphQueryContext(**payload)

    def _bootstrap_graph_context(
        self,
        *,
        question: str,
        scope: GraphAccessScope,
    ) -> GraphQueryContext:
        adapter = getattr(getattr(self, "graph_loop", None), "adapter", None)
        adapter_name = "graph_adapter"
        try:
            meta = adapter.metadata() if callable(getattr(adapter, "metadata", None)) else None
        except Exception:
            meta = None
        if meta is not None:
            adapter_name = str(getattr(meta, "adapter_name", None) or adapter_name)
        elif adapter is not None:
            adapter_name = str(getattr(adapter, "name", adapter_name) or adapter_name)
        return GraphQueryContext(
            adapter_name=adapter_name,
            question=str(question or "").strip(),
            access_scope=scope,
            metadata={},
        )
    def _build_state(
        self,
        *,
        run_id: Optional[str],
        stage_listener: Optional[Callable[[Dict[str, Any], DeepSearchState], None]],
    ) -> DeepSearchState:
        state_cls: Type[DeepSearchState] = getattr(self, "state_cls", DeepSearchState)
        kwargs: Dict[str, Any] = {"config_fingerprint": self._config_fingerprint()}
        if run_id:
            kwargs["run_id"] = run_id
        if stage_listener:
            kwargs["stage_listener"] = stage_listener
        return state_cls(**kwargs)  # type: ignore[arg-type]

    def _resolve_scope(
        self,
        *,
        owner_id: Optional[str],
        access_scope: Optional[GraphAccessScope],
        graph_context: Optional[GraphQueryContext],
    ) -> GraphAccessScope:
        if graph_context is not None and getattr(graph_context, "access_scope", None):
            return graph_context.access_scope
        if access_scope:
            return access_scope
        if owner_id:
            return GraphAccessScope(scope_id=str(owner_id), scope_type="owner")
        return require_scope()

    @staticmethod
    def _attach_run_metadata(
        context: GraphQueryContext,
        *,
        run_id: str,
        metadata: Optional[Dict[str, Any]],
        artifact_dir: Optional[str] = None,
    ) -> GraphQueryContext:
        base = dict(context.metadata or {})
        base["run_id"] = run_id
        if artifact_dir:
            base["artifact_dir"] = str(artifact_dir)
        if metadata:
            request_bucket = base.setdefault("request_metadata", {})
            if isinstance(request_bucket, dict):
                request_bucket.update(metadata)
            for key in ("file_scope", "compression"):
                value = metadata.get(key)
                if value is not None:
                    base[key] = value
        try:
            return context.model_copy(update={"metadata": base})
        except AttributeError:
            payload = context.model_dump(exclude_none=True)
            payload["metadata"] = base
            return GraphQueryContext(**payload)

    @staticmethod
    def _attach_file_scope_hints(context: GraphQueryContext, *, question: str) -> GraphQueryContext:
        from core.deepsearch.utils.file_scope import extract_titles_from_question

        titles = extract_titles_from_question(question or "")
        if not titles:
            return context
        metadata = dict(context.metadata or {})
        metadata.setdefault(
            "file_scope",
            {
                "filename_contains": titles,
                "source": "question_titles",
            },
        )
        try:
            return context.model_copy(update={"metadata": metadata})
        except AttributeError:
            payload = context.model_dump(exclude_none=True)
            payload["metadata"] = metadata
            return GraphQueryContext(**payload)
