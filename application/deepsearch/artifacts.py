"""Filesystem-backed artifacts for DeepSearch runs.

Make runs reproducible by writing plan/evidence/report/external calls
to a stable per-run folder, so we can replay and debug regressions.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils.json_safe import json_safe


@dataclass(frozen=True)
class DeepSearchArtifactStore:
    root_dir: Path

    @staticmethod
    def from_env(root_dir: str | None = None) -> "DeepSearchArtifactStore | None":
        value = root_dir or os.getenv("DEEPSEARCH_ARTIFACT_DIR")
        if not value:
            return None
        path = Path(str(value)).expanduser()
        return DeepSearchArtifactStore(root_dir=path)

    def ensure_run_dir(self, run_id: str) -> Path:
        folder = self.root_dir / str(run_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def write_json(self, run_id: str, filename: str, payload: Dict[str, Any]) -> Path:
        folder = self.ensure_run_dir(run_id)
        path = folder / filename
        path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_text(self, run_id: str, filename: str, content: str) -> Path:
        folder = self.ensure_run_dir(run_id)
        path = folder / filename
        path.write_text(str(content or ""), encoding="utf-8")
        return path

    def run_metadata(self, run_id: str) -> Dict[str, Any]:
        folder = self.ensure_run_dir(run_id)
        return {"artifact_dir": str(folder)}

