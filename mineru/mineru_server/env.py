import os
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


def _strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    return v


def parse_dotenv(text: str) -> Dict[str, str]:
    """
    Minimal .env parser:
      - KEY=VALUE
      - ignores empty lines and lines starting with '#'
      - strips surrounding quotes from VALUE
      - does not support shell expansion (by design)
    """
    out: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = _strip_quotes(value)
    return out


def find_dotenv_path(*, cwd: Optional[Path] = None, repo_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Search order:
      1) MINERU_DOTENV_PATH (explicit)
      2) {repo_dir}/.env
      3) {cwd}/.env
    """
    env_path = os.environ.get("MINERU_DOTENV_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None

    if repo_dir is not None:
        p = repo_dir.resolve() / ".env"
        if p.exists():
            return p

    cwd = (cwd or Path.cwd()).resolve()
    p = cwd / ".env"
    if p.exists():
        return p

    return None


def load_dotenv(*, override: bool = False, cwd: Optional[Path] = None, repo_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Load dotenv into os.environ. By default does NOT override existing env vars.
    Returns the loaded path or None.
    """
    path = find_dotenv_path(cwd=cwd, repo_dir=repo_dir)
    if path is None:
        return None
    try:
        data = parse_dotenv(path.read_text(encoding="utf-8", errors="ignore"))
        for k, v in data.items():
            if override or (k not in os.environ):
                os.environ[k] = v
        logger.info(f"Loaded .env: {path}")
        return path
    except Exception as exc:
        logger.warning(f"Failed to load .env {path}: {exc}")
        return None
