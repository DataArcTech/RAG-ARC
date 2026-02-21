import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is importable even when CLI runs outside repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import app_registration
from dotenv import load_dotenv
from encapsulation.data_model.orm_models import User
from core.graph_adapter.scope_provider import configure_scope_provider

logger = logging.getLogger(__name__)
DEFAULT_OWNER_FILE = Path(os.getenv("CLI_OWNER_ID_FILE", Path.home() / ".rag_arc_owner_id"))


@dataclass
class CLIContext:
    """Holds shared dependencies for CLI commands."""

    owner_id: uuid.UUID


_initialized = False


def initialize(owner_id: Optional[str] = None) -> CLIContext:
    """Load environment variables and initialize registrator if needed."""
    global _initialized
    if not _initialized:
        # Avoid python-dotenv's frame-based find_dotenv() heuristics; in some execution
        # environments (e.g. uv + stdin, embedded runners), it can fail to locate .env.
        # CLI should always honor repo-local `RAG-ARC/.env` as the single source of truth.
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
        configure_scope_provider()  # ensure GraphAccessScope picks up freshly loaded .env values
        app_registration.initialize()
        _initialized = True
        logger.info("CLI bootstrap completed")
    resolved_owner = _resolve_owner_id(owner_id)
    if os.getenv("DEVELOP_MODE", "false").lower() == "true":
        _ensure_develop_owner_user(resolved_owner)
    return CLIContext(owner_id=resolved_owner)


def _resolve_owner_id(owner_id: Optional[str]) -> uuid.UUID:
    """Get the owner_id from args/env/persisted file (in that order)."""
    if owner_id:
        if not _is_valid_uuid(owner_id):
            raise ValueError("owner_id must be a valid UUID string")
        return uuid.UUID(owner_id)

    env_owner = os.getenv("CLI_OWNER_ID") or os.getenv("RAG_ARC_OWNER_ID") or os.getenv("DEFAULT_OWNER_ID")
    if env_owner:
        if _is_valid_uuid(env_owner):
            return uuid.UUID(env_owner)
        logger.warning("Environment owner id (%s) is not a valid UUID, ignoring", env_owner)

    if os.getenv("DEVELOP_MODE", "false").lower() == "true":
        dev_owner = os.getenv("DEVELOP_OWNER_ID", "00000000-0000-0000-0000-000000000001")
        logger.info("DEVELOP_MODE enabled, using owner id %s", dev_owner)
        return uuid.UUID(dev_owner)

    owner_file = DEFAULT_OWNER_FILE
    if owner_file.exists():
        try:
            value = owner_file.read_text().strip()
            if value and _is_valid_uuid(value):
                return uuid.UUID(value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read CLI owner id file (%s): %s, regenerating", owner_file, exc)

    new_owner = uuid.uuid4()
    try:
        owner_file.parent.mkdir(parents=True, exist_ok=True)
        owner_file.write_text(str(new_owner))
        logger.info("Generated CLI owner id %s and saved to %s", new_owner, owner_file)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist CLI owner id to %s: %s", owner_file, exc)
    return new_owner


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except Exception:
        return False


def _ensure_develop_owner_user(owner_id: uuid.UUID) -> None:
    """Create a placeholder user for develop mode if it does not exist."""
    try:
        account = app_registration.registrator.get_object("account")
    except KeyError:
        logger.warning("Account module not available; cannot create develop-mode user")
        return

    try:
        if account.get_user_by_id(owner_id):
            return
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to query develop-mode user: %s", exc)

    base_username = os.getenv("DEVELOP_OWNER_USERNAME", "dev_cli_user")
    password = os.getenv("DEVELOP_OWNER_PASSWORD", "dev-cli-password")
    hashed_password = account.get_password_hash(password)

    username_candidates = [
        base_username,
        f"{base_username}_{owner_id.hex[:8]}",
        f"{base_username}_{owner_id.hex}",
    ]
    username = None
    for candidate in username_candidates:
        try:
            if not account.get_user_by_username(candidate):
                username = candidate
                break
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to query develop-mode username %s: %s", candidate, exc)

    if username is None:
        username = f"{base_username}_{uuid.uuid4().hex[:8]}"

    user = User(
        id=owner_id,
        user_name=username,
        hashed_password=hashed_password,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    try:
        account.user_storage.metadata_store.store_user(user)
        logger.info("Created develop-mode CLI user %s (%s)", username, owner_id)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to create develop-mode user: %s", exc)
