import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils.path_guard import require_writable_dir
from framework.virtual_paths import is_io_path, resolve_io_to_local_path

logger = logging.getLogger(__name__)


def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    """Hash a small JSON-serializable dict in a stable way (sorted keys, compact)."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MinerUSharedCacheKey:
    bytes_sha256: str
    parser_fingerprint: str

    def rel_dir(self) -> Path:
        return Path("by_sha256") / self.bytes_sha256 / self.parser_fingerprint


def build_parser_fingerprint(*, params: Dict[str, Any]) -> str:
    """Build a fingerprint for MinerU parse parameters.

    This fingerprint is part of the shared-cache key so that the same file bytes parsed
    with different MinerU settings do not accidentally reuse each other.
    """
    return _stable_json_hash(params)


def resolve_shared_cache_dir(*, base_dir: str | None) -> Optional[Path]:
    raw = str(base_dir or "").strip()
    if not raw:
        return None
    if is_io_path(raw):
        # Ensure the underlying local directory exists and is writable.
        require_writable_dir(raw)
        return resolve_io_to_local_path(raw)
    return Path(raw).expanduser()


def shared_cache_hit(shared_root: Path, key: MinerUSharedCacheKey) -> Optional[Path]:
    """Return cache directory when present and complete, else None."""
    cand = (shared_root / key.rel_dir()).resolve()
    complete = cand / ".complete"
    if not cand.exists() or not cand.is_dir():
        return None
    if not complete.exists():
        return None
    md_files = list(cand.glob("*.md"))
    if not md_files:
        return None
    try:
        if any(p.stat().st_size > 0 for p in md_files):
            return cand
    except Exception:
        return None
    return None


def _try_symlink_dir(src: Path, dst: Path) -> bool:
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
        return True
    except Exception:
        return False


def _safe_clear_empty_dir(path: Path) -> None:
    try:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except Exception:
        return


def materialize_shared_cache(
    *,
    shared_dir: Path,
    dest_dir: Path,
    mode: str,
) -> Dict[str, Any]:
    """Materialize shared cache artifacts under dest_dir.

    The caller owns the directory naming convention; this helper only performs filesystem operations.
    """
    mode_norm = (mode or "").strip().lower() or "symlink"
    if mode_norm not in {"symlink", "copy"}:
        mode_norm = "symlink"

    result: Dict[str, Any] = {"mode": mode_norm, "ok": False, "error": None}
    if dest_dir.exists() and dest_dir.is_symlink():
        try:
            # If already pointing at the same shared dir, treat as ok.
            target = Path(os.readlink(dest_dir)).resolve()
            if target == shared_dir.resolve():
                result["ok"] = True
                return result
        except Exception:
            pass

    if mode_norm == "symlink":
        _safe_clear_empty_dir(dest_dir)
        if not dest_dir.exists():
            if _try_symlink_dir(shared_dir, dest_dir):
                result["ok"] = True
                return result
        # fallthrough to copy
        result["mode"] = "copy"
        mode_norm = "copy"

    if mode_norm == "copy":
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(shared_dir, dest_dir, dirs_exist_ok=True)
            result["ok"] = True
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result

    return result


def publish_to_shared_cache(
    *,
    shared_root: Path,
    key: MinerUSharedCacheKey,
    src_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Best-effort publish MinerU artifacts into shared cache.

    Uses a temp dir + rename to avoid exposing partial results.
    """
    dest = (shared_root / key.rel_dir()).resolve()
    result: Dict[str, Any] = {"ok": False, "dest": str(dest)}
    try:
        if not overwrite and (dest / ".complete").exists():
            result["ok"] = True
            result["skipped"] = "already_complete"
            return result
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src_dir, tmp, dirs_exist_ok=True)
        (tmp / ".complete").write_text("ok\n", encoding="utf-8")
        try:
            if overwrite and dest.exists():
                backup = dest.with_name(dest.name + f".bak-{os.getpid()}")
                try:
                    if backup.exists():
                        shutil.rmtree(backup, ignore_errors=True)
                    os.rename(str(dest), str(backup))
                except Exception:
                    # If we can't move away the existing dir, do not risk clobbering it.
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise
                try:
                    os.rename(str(tmp), str(dest))
                    shutil.rmtree(backup, ignore_errors=True)
                    result["overwritten"] = True
                except Exception:
                    # Best-effort rollback.
                    try:
                        if dest.exists():
                            shutil.rmtree(dest, ignore_errors=True)
                        os.rename(str(backup), str(dest))
                    except Exception:
                        pass
                    raise
            else:
                os.rename(str(tmp), str(dest))
        except FileExistsError:
            shutil.rmtree(tmp, ignore_errors=True)
        result["ok"] = True
        return result
    except Exception as exc:
        logger.debug("Failed to publish MinerU artifacts to shared cache: %s", exc, exc_info=True)
        result["error"] = str(exc)
        return result
