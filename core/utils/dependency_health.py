import logging
import os
from typing import Any, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _env_mode(name: str, default: str) -> str:
    value = (os.getenv(name, default) or default).strip().lower()
    if value in {"off", "0", "false", "no"}:
        return "off"
    if value in {"strict", "hard", "raise"}:
        return "strict"
    return "warn"


def check_postgres() -> Dict[str, Any]:
    try:
        from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

        db = PostgreSQLConfig().build()
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def check_redis() -> Dict[str, Any]:
    try:
        from encapsulation.database.cache_db.redis_db import RedisDB
        from config.encapsulation.database.cache_db.redis_config import RedisConfig

        client = RedisDB(RedisConfig()).client
        client.ping()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def check_neo4j() -> Dict[str, Any]:
    try:
        import neo4j

        url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        driver = neo4j.GraphDatabase.driver(url, auth=(username, password), notifications_min_severity="OFF")
        try:
            with driver.session(database=database) as session:
                session.run("RETURN 1 AS ok").consume()
        finally:
            driver.close()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def check_dependencies(
    *,
    include_postgres: bool = True,
    include_redis: bool = True,
    include_neo4j: bool = True,
    mode_env: str = "RAGARC_DEPENDENCY_CHECK_MODE",
    default_mode: str = "warn",
) -> Dict[str, Any]:
    """
    Dependency health checks (Postgres/Redis/Neo4j).

    Modes:
    - off: skip all checks
    - warn: return diagnostics and log warnings
    - strict: raise RuntimeError on any failed dependency
    """
    mode = _env_mode(mode_env, default_mode)
    if mode == "off":
        return {"mode": mode, "ok": True, "checks": {}}

    checks: Dict[str, Any] = {}
    if include_postgres:
        checks["postgres"] = check_postgres()
    if include_redis:
        checks["redis"] = check_redis()
    if include_neo4j:
        checks["neo4j"] = check_neo4j()

    failed = {name: res for name, res in checks.items() if not res.get("ok")}
    ok = not failed

    if not ok:
        error_summary = "; ".join(f"{name}={res.get('error')}" for name, res in failed.items())
        logger.warning("Dependency health check failed: %s", error_summary)
        if mode == "strict":
            raise RuntimeError(f"Dependency health check failed: {error_summary}")

    return {"mode": mode, "ok": ok, "checks": checks}


def format_dependency_failures(health: Dict[str, Any]) -> Optional[str]:
    checks = health.get("checks") or {}
    failures = [(name, res.get("error")) for name, res in checks.items() if not res.get("ok")]
    if not failures:
        return None
    return "; ".join(f"{name}: {err}" for name, err in failures)
