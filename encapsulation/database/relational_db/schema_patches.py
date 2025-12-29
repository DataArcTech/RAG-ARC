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
        # Additive enum value for `FileStatus.PARTIAL_INDEXED` (avoid masking "no graph" as INDEXED).
        """
        DO $$
        DECLARE enum_type text;
        DECLARE has_value boolean;
        BEGIN
          SELECT udt_name INTO enum_type
          FROM information_schema.columns
          WHERE table_schema='public' AND table_name='file_metadata' AND column_name='status';

          IF enum_type IS NULL THEN
            RETURN;
          END IF;

          SELECT EXISTS (
            SELECT 1
            FROM pg_enum e
            JOIN pg_type t ON t.oid = e.enumtypid
            WHERE t.typname = enum_type AND e.enumlabel = 'PARTIAL_INDEXED'
          ) INTO has_value;

          IF NOT has_value THEN
            EXECUTE format('ALTER TYPE %I ADD VALUE %L', enum_type, 'PARTIAL_INDEXED');
          END IF;
        END $$;
        """,
        # TaskRun should not depend on user table rows existing; historical Redis task runs may reference deleted users.
        "ALTER TABLE public.task_run DROP CONSTRAINT IF EXISTS task_run_owner_id_fkey;",
    ]

    for stmt in statements:
        try:
            # ALTER TYPE ... ADD VALUE historically required autocommit in some Postgres versions.
            use_autocommit = "alter type" in stmt.lower()
            if use_autocommit:
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.execute(text(stmt))
            else:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
        except SQLAlchemyError as exc:
            logger.warning("Schema patch failed: %s (%s)", stmt, exc)
