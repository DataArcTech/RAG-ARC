import asyncio
import uuid
from typing import Dict, Optional


class KnowledgeRuntimeStateMixin:
    """Helpers for tracking background tasks and deletion state.

    This mixin keeps Knowledge module state-handling cohesive and testable without
    bloating `application/knowledge/module.py`.
    """

    def _track_background_task(self, doc_id: str, task: asyncio.Task) -> None:
        """Register a background indexing task so it can be cancelled or awaited later."""

        self._active_index_tasks[doc_id] = task

        def _cleanup(fut: asyncio.Task, file_id: str = doc_id) -> None:
            self._active_index_tasks.pop(file_id, None)
            if fut.cancelled():
                self.logger.info("Background indexing task cancelled for file_id: %s", file_id)
            elif fut.exception():
                self.logger.error(
                    "Background indexing task failed for file_id %s: %s",
                    file_id,
                    fut.exception(),
                    exc_info=True,
                )

        task.add_done_callback(_cleanup)

    async def _cancel_indexing_task(self, doc_id: str) -> None:
        """Cancel an active background indexing task for the specified file."""

        task = self._active_index_tasks.get(doc_id)
        if not task:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            self.logger.info("Cancelled indexing task awaited for file_id: %s", doc_id)

    def _mark_file_for_deletion(self, doc_id: str, owner_id: Optional[uuid.UUID] = None) -> None:
        """Mark a file so background tasks can skip further processing."""

        self._files_marked_for_deletion.add(doc_id)
        resolved_owner = self._resolve_owner_for_file(doc_id, owner_id)
        if resolved_owner is not None:
            owner_set = self._files_marked_for_deletion_by_owner.setdefault(resolved_owner, set())
            owner_set.add(doc_id)
            self._file_owner_cache[doc_id] = resolved_owner

    def _unmark_file_for_deletion(self, doc_id: str) -> None:
        self._files_marked_for_deletion.discard(doc_id)
        owner_id = self._file_owner_cache.pop(doc_id, None)
        if owner_id is not None:
            owner_set = self._files_marked_for_deletion_by_owner.get(owner_id)
            if owner_set is not None:
                owner_set.discard(doc_id)
                if not owner_set:
                    self._files_marked_for_deletion_by_owner.pop(owner_id, None)

    def _is_file_marked_for_deletion(self, doc_id: str) -> bool:
        return doc_id in self._files_marked_for_deletion

    def _resolve_owner_for_file(
        self,
        doc_id: str,
        explicit_owner: Optional[uuid.UUID] = None,
    ) -> Optional[uuid.UUID]:
        if explicit_owner is not None:
            return explicit_owner
        cached_owner = self._file_owner_cache.get(doc_id)
        if cached_owner is not None:
            return cached_owner
        try:
            metadata = self.file_storage.get_file_metadata(doc_id)
        except Exception:
            metadata = None
        if metadata and getattr(metadata, "owner_id", None) is not None:
            owner_id = metadata.owner_id
            self._file_owner_cache[doc_id] = owner_id
            return owner_id
        return None

