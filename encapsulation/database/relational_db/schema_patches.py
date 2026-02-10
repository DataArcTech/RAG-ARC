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
        # Backward compatible user schema upgrades.
        # Historical deployments created `public.user` without `type`, but newer auth/account logic requires it.
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS type integer NOT NULL DEFAULT 0;',
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS last_login_at timestamp;',
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS name varchar(255);',
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS company_name varchar(255);',
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS department_id uuid;',
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS role_id uuid;',
        # Ensure uniqueness matches ORM expectation (`UniqueConstraint("user_name", "type")`).
        'CREATE UNIQUE INDEX IF NOT EXISTS uq_user_name_type_idx ON public."user"(user_name, type);',
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
        # Add sources field to chat_message table for storing complete source information
        "ALTER TABLE public.chat_message ADD COLUMN IF NOT EXISTS sources jsonb;",
        # Create ChatMessageStatus enum type (if not exists)
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chatmessagestatus') THEN
                CREATE TYPE chatmessagestatus AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED');
            END IF;
        END $$;
        """,
        # Add task tracking fields to chat_message table
        "ALTER TABLE public.chat_message ADD COLUMN IF NOT EXISTS status chatmessagestatus;",
        "ALTER TABLE public.chat_message ADD COLUMN IF NOT EXISTS task_id varchar(255);",
        "ALTER TABLE public.chat_message ADD COLUMN IF NOT EXISTS request_id varchar(255);",
        "CREATE INDEX IF NOT EXISTS ix_chat_message_status ON public.chat_message (status);",
        "CREATE INDEX IF NOT EXISTS ix_chat_message_task_id ON public.chat_message (task_id);",
        # Add current task tracking fields to chat_session table
        "ALTER TABLE public.chat_session ADD COLUMN IF NOT EXISTS current_task_id varchar(255);",
        "ALTER TABLE public.chat_session ADD COLUMN IF NOT EXISTS current_task_status varchar(50);",
        "ALTER TABLE public.chat_session ADD COLUMN IF NOT EXISTS current_task_started_at timestamp;",
        "CREATE INDEX IF NOT EXISTS ix_chat_session_current_task_id ON public.chat_session (current_task_id);",
        # Parsed content metadata: store optional parser artifact paths for parse/index decoupling.
        "ALTER TABLE public.parsed_content_metadata ADD COLUMN IF NOT EXISTS md_path varchar(1000);",
        "ALTER TABLE public.parsed_content_metadata ADD COLUMN IF NOT EXISTS output_dir varchar(1000);",
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
