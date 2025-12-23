from dataclasses import dataclass

from core.file_management.index_manager import IndexManager
from encapsulation.data_model.orm_models import FileStatus


@dataclass
class _Meta:
    status: FileStatus


class _MetaStore:
    def __init__(self) -> None:
        self.updated_to_indexed = False

    def update_file_metadata(self, file_id: str, patch: dict, **kwargs):  # noqa: ARG002
        if patch.get("status") == FileStatus.INDEXED:
            self.updated_to_indexed = True
        return True


class _FileStorage:
    def __init__(self, meta: _Meta, store: _MetaStore) -> None:
        self._meta = meta
        self.metadata_store = store

    def get_file_metadata(self, file_id: str):  # noqa: ARG002
        return self._meta


def test_index_manager_does_not_resurrect_deleted_file():
    store = _MetaStore()
    file_storage = _FileStorage(_Meta(status=FileStatus.DELETED), store)

    mgr = IndexManager.__new__(IndexManager)
    mgr.file_storage = file_storage  # type: ignore[attr-defined]

    mgr._update_file_status_to_indexed("file_1")  # noqa: SLF001
    assert store.updated_to_indexed is False

