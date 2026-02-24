import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class LocalPathIO:
    """Minimal IOManager-like facade backed by the local filesystem.

    This is used by parsers to support `output_dir` as a normal filesystem directory in
    unit tests and lightweight scripts, while keeping the production `io://...` path
    behavior unchanged.
    """

    base_dir: Path

    def _resolve(self, path: str) -> Path:
        token = str(path or "").strip()
        if not token:
            raise ValueError("LocalPathIO requires a non-empty path")
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        resolved = candidate.expanduser().resolve()
        try:
            resolved.relative_to(self.base_dir.resolve())
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Refusing to write outside base_dir={self.base_dir}: {resolved}") from exc
        return resolved

    def put_text_path(self, path: str, *, text: str, content_type: Optional[str] = None) -> None:  # noqa: ARG002
        dst = self._resolve(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")

    def put_bytes_path(self, path: str, *, payload: bytes, content_type: Optional[str] = None) -> None:  # noqa: ARG002
        dst = self._resolve(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(payload)

    def put_json_path(self, path: str, *, payload: Any) -> None:
        dst = self._resolve(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def get_text_path(self, path: str) -> Optional[str]:
        src = self._resolve(path)
        if not src.exists() or not src.is_file():
            return None
        return src.read_text(encoding="utf-8")

    def list_keys_path(self, dir_path: str, *, limit: int = 2000) -> list[str]:
        root = self._resolve(dir_path)
        if not root.exists() or not root.is_dir():
            return []
        out: list[str] = []
        for p in root.rglob("*"):
            if p.is_file():
                out.append(str(p))
                if len(out) >= limit:
                    break
        return sorted(out)

