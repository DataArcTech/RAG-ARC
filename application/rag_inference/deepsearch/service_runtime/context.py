import logging
from typing import Any, Callable, Dict, Optional, Type

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.deepsearch.state import DeepSearchState
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.scope_provider import require_scope

logger = logging.getLogger(__name__)


class DeepSearchServiceContextMixin:
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
        external_allowed: Optional[bool] = None,
        artifact_dir: Optional[str] = None,
    ) -> GraphQueryContext:
        base = dict(context.metadata or {})
        base["run_id"] = run_id
        if artifact_dir:
            base["artifact_dir"] = str(artifact_dir)
        if external_allowed is not None:
            base["external_allowed"] = bool(external_allowed)
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

    def _external_allowed_flag(self) -> bool:
        channel = getattr(self, "external_channel", None)
        if channel is None:
            return False
        checker = getattr(channel, "_is_enabled", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        enabled = getattr(channel, "enabled", None)
        if enabled is None:
            # If a channel instance is explicitly injected but does not expose an enable flag,
            # assume it is enabled (test stubs and custom implementations).
            return True
        return bool(enabled)
