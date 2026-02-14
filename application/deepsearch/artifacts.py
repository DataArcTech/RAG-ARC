"""Artifacts store for DeepSearch runs.

Make runs reproducible by writing plan/evidence/report/external calls
to a stable per-run folder, so we can replay and debug regressions.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.constants.io_namespaces import DEEPSEARCH_ARTIFACTS_NAMESPACE
from core.utils.json_safe import json_safe


@dataclass(frozen=True)
class DeepSearchArtifactStore:
    io_manager: Any
    namespace: str = DEEPSEARCH_ARTIFACTS_NAMESPACE
    prefix: str = ""

    @staticmethod
    def from_io_manager(
        io_manager: Any,
        *,
        namespace: str = DEEPSEARCH_ARTIFACTS_NAMESPACE,
        prefix: str = "",
    ) -> "DeepSearchArtifactStore":
        if io_manager is None:
            raise ValueError("DeepSearchArtifactStore.from_io_manager requires io_manager")
        ns = str(namespace or "").strip() or DEEPSEARCH_ARTIFACTS_NAMESPACE
        pre = str(prefix or "").strip().lstrip("/").lstrip("\\")
        return DeepSearchArtifactStore(io_manager=io_manager, namespace=ns, prefix=pre)

    def ensure_run_dir(self, run_id: str) -> str:
        token = str(run_id or "").strip()
        if not token:
            raise ValueError("run_id is required")
        return f"io://{self.namespace}/{self.prefix}/{token}/".replace("//", "/").replace("io:/", "io://")

    def _key(self, run_id: str, filename: str) -> str:
        rid = str(run_id or "").strip()
        name = str(filename or "").strip().lstrip("/").lstrip("\\")
        if not rid:
            raise ValueError("run_id is required")
        if not name:
            raise ValueError("filename is required")
        if self.prefix:
            return f"{self.prefix}/{rid}/{name}".lstrip("/")
        return f"{rid}/{name}".lstrip("/")

    def write_json(self, run_id: str, filename: str, payload: Dict[str, Any]) -> str:
        put = self.io_manager.put_json(
            namespace=self.namespace,
            key=self._key(run_id, filename),
            payload=json_safe(payload),
        )
        return str(put.ref)

    def write_text(self, run_id: str, filename: str, content: str) -> str:
        put = self.io_manager.put_text(
            namespace=self.namespace,
            key=self._key(run_id, filename),
            text=str(content or ""),
        )
        return str(put.ref)

    def run_metadata(self, run_id: str) -> Dict[str, Any]:
        folder = self.ensure_run_dir(run_id)
        return {"artifact_dir": str(folder)}
