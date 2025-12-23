import logging

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def apply_postgres_schema_patches(engine: Engine) -> None:
    """
    Apply backward-compatible, additive schema patches for PostgreSQL.

    Why this exists:
    - SQLAlchemy `Base.metadata.create_all()` only creates missing tables/types.
      It does NOT add missing columns to existing tables.
    - For non-destructive upgrades (keeping data), we apply small "IF NOT EXISTS"
      ALTERs/INDEXes here.
    """

    statements: list[str] = [
        # Added in ORM (`FileMetadata.content_hash`) but may be missing in older DBs.
        "ALTER TABLE public.file_metadata ADD COLUMN IF NOT EXISTS content_hash varchar(64);",
        "CREATE INDEX IF NOT EXISTS ix_file_metadata_content_hash ON public.file_metadata (content_hash);",
        # TaskRun should not depend on user table rows existing; historical Redis task runs may reference deleted users.
        "ALTER TABLE public.task_run DROP CONSTRAINT IF EXISTS task_run_owner_id_fkey;",
    ]

    for stmt in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except SQLAlchemyError as exc:
            logger.warning("Schema patch failed: %s (%s)", stmt, exc)
