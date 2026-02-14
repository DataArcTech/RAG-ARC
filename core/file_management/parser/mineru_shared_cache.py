import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from encapsulation.io.io_manager import IOManager
from framework.virtual_paths import IO_PATH_PREFIX, io_key, is_io_path

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

    def rel_dir(self) -> str:
        return "/".join(["by_sha256", self.bytes_sha256, self.parser_fingerprint])


def build_parser_fingerprint(*, params: Dict[str, Any]) -> str:
    """Build a fingerprint for MinerU parse parameters.

    This fingerprint is part of the shared-cache key so that the same file bytes parsed
    with different MinerU settings do not accidentally reuse each other.
    """
    return _stable_json_hash(params)


def resolve_shared_cache_dir(*, base_dir: str | None) -> Optional[str]:
    raw = str(base_dir or "").strip()
    if not raw:
        return None
    if not is_io_path(raw):
        raise ValueError("MinerU shared cache dir must be an io:// virtual path")
    return raw.rstrip("/")


def _join_virtual_dir(base: str, suffix: str) -> str:
    left = str(base or "").strip().rstrip("/")
    right = str(suffix or "").strip().lstrip("/")
    return f"{left}/{right}" if right else left


def _split_virtual_dir(base_dir: str) -> tuple[str, str]:
    token = str(base_dir or "").strip()
    if not is_io_path(token):
        raise ValueError(f"expected an io:// virtual dir, got: {base_dir!r}")
    key = io_key(token)
    parts = [p for p in key.split("/") if p]
    if not parts:
        raise ValueError(f"empty io:// dir: {base_dir!r}")
    namespace = parts[0]
    prefix = "/".join(parts[1:])
    return namespace, prefix


def shared_cache_hit(io_manager: IOManager, shared_root: str, key: MinerUSharedCacheKey) -> Optional[str]:
    """Return cache virtual dir when present and complete, else None."""

    cand = _join_virtual_dir(shared_root, key.rel_dir())
    complete = _join_virtual_dir(cand, ".complete")
    if not io_manager.exists(complete):
        return None
    try:
        keys = io_manager.list_keys_path(cand, limit=2000)
    except Exception:
        return None
    md_keys = [k for k in keys if str(k).lower().endswith(".md")]
    if not md_keys:
        return None
    for ref in md_keys[:3]:
        try:
            blob = io_manager.get_bytes(ref)
            if blob and blob.strip():
                return cand
        except Exception:
            continue
    return None


def materialize_shared_cache(
    *,
    io_manager: IOManager,
    shared_dir: str,
    dest_dir: str,
    mode: str,
) -> Dict[str, Any]:
    """Materialize shared cache artifacts under dest_dir via object copies."""

    mode_norm = (mode or "").strip().lower() or "copy"
    if mode_norm != "copy":
        mode_norm = "copy"

    result: Dict[str, Any] = {"mode": mode_norm, "ok": False, "error": None}
    shared_dir = str(shared_dir or "").strip().rstrip("/")
    dest_dir = str(dest_dir or "").strip().rstrip("/")
    if not (shared_dir.startswith(IO_PATH_PREFIX) and dest_dir.startswith(IO_PATH_PREFIX)):
        result["error"] = "shared_dir/dest_dir must be io:// virtual dirs"
        return result

    try:
        keys = io_manager.list_keys_path(shared_dir, limit=None)
        prefix = shared_dir.rstrip("/") + "/"
        for ref in keys:
            token = str(ref)
            if not token.startswith(prefix):
                continue
            suffix = token[len(prefix) :]
            if not suffix or suffix == ".complete":
                continue
            payload = io_manager.get_bytes(token)
            if payload is None:
                continue
            io_manager.put_bytes_path(_join_virtual_dir(dest_dir, suffix), payload=payload)
        io_manager.put_text_path(_join_virtual_dir(dest_dir, ".complete"), text="ok\n")
        result["ok"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def publish_to_shared_cache(
    *,
    io_manager: IOManager,
    shared_root: str,
    key: MinerUSharedCacheKey,
    src_dir: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Best-effort publish MinerU artifacts into shared cache (object store).

    Consistency rule:
    - Files are copied under the destination prefix.
    - `.complete` is written last as a commit marker.
    """

    shared_root = str(shared_root or "").strip().rstrip("/")
    src_dir = str(src_dir or "").strip().rstrip("/")
    if not (shared_root.startswith(IO_PATH_PREFIX) and src_dir.startswith(IO_PATH_PREFIX)):
        return {"ok": False, "error": "shared_root/src_dir must be io:// virtual dirs"}

    dest_dir = _join_virtual_dir(shared_root, key.rel_dir())
    result: Dict[str, Any] = {"ok": False, "dest": dest_dir}
    try:
        complete = _join_virtual_dir(dest_dir, ".complete")
        if not overwrite and io_manager.exists(complete):
            result["ok"] = True
            result["skipped"] = "already_complete"
            return result

        if overwrite:
            try:
                existing = io_manager.list_keys_path(dest_dir, limit=None)
                for ref in existing:
                    io_manager.delete(str(ref))
            except Exception:
                pass

        keys = io_manager.list_keys_path(src_dir, limit=None)
        prefix = src_dir.rstrip("/") + "/"
        for ref in keys:
            token = str(ref)
            if not token.startswith(prefix):
                continue
            suffix = token[len(prefix) :]
            if not suffix or suffix == ".complete":
                continue
            payload = io_manager.get_bytes(token)
            if payload is None:
                continue
            io_manager.put_bytes_path(_join_virtual_dir(dest_dir, suffix), payload=payload)
        io_manager.put_text_path(complete, text="ok\n")
        result["ok"] = True
        return result
    except Exception as exc:
        logger.debug("Failed to publish MinerU artifacts to shared cache: %s", exc, exc_info=True)
        result["error"] = str(exc)
        return result
