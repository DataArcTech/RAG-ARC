import logging
import os
from typing import Any, Dict, Optional

import httpx
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


def _mineru_health_url(base_url: str) -> str:
    token = str(base_url or "").strip()
    return token.rstrip("/") + "/health"


def _mineru_healthcheck_timeout_s() -> float:
    raw = str(os.getenv("MINERU_HEALTHCHECK_TIMEOUT_S", "") or "").strip()
    if not raw:
        return 2.0
    try:
        return max(0.1, float(raw))
    except Exception:
        return 2.0


def check_mineru() -> Dict[str, Any]:
    """
    Check MinerU health endpoint when PARSER_PARSE_MODE=mineru.

    Uses:
    - MINERU_SERVER_URL: base URL, e.g. http://127.0.0.1:8899
    - MINERU_HEALTHCHECK_TIMEOUT_S: startup healthcheck timeout seconds (default: 2)
    """
    base_url = str(os.getenv("MINERU_SERVER_URL", "") or "").strip()
    if not base_url:
        return {"ok": False, "error": "MINERU_SERVER_URL is empty (required when PARSER_PARSE_MODE=mineru)."}

    url = _mineru_health_url(base_url)
    timeout_s = _mineru_healthcheck_timeout_s()
    try:
        resp = httpx.get(url, timeout=timeout_s)
        if resp.status_code != 200:
            return {"ok": False, "error": f"GET {url} returned HTTP {resp.status_code}"}
        try:
            payload = resp.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").strip().lower()
            if status and status != "healthy":
                return {"ok": False, "error": f"GET {url} returned status={status!r}"}
        return {"ok": True, "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"GET {url} failed: {exc}"}


def check_dependencies(
    *,
    include_postgres: bool = True,
    include_redis: bool = True,
    include_neo4j: bool = True,
    include_mineru: bool | None = None,
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
    if include_mineru is None:
        include_mineru = str(os.getenv("PARSER_PARSE_MODE", "native") or "native").strip().lower() == "mineru"
    if include_mineru:
        checks["mineru"] = check_mineru()

    failed = {name: res for name, res in checks.items() if not res.get("ok")}
    ok = not failed

    if not ok:
        error_summary = "; ".join(f"{name}={res.get('error')}" for name, res in failed.items())
        logger.warning("Dependency health check failed: %s", error_summary)
        if mode == "strict":
            if "mineru" in failed:
                mineru_url = str(os.getenv("MINERU_SERVER_URL", "") or "").strip()
                health_url = mineru_url.rstrip("/") + "/health" if mineru_url else "<empty MINERU_SERVER_URL>/health"
                hint = (
                    "MinerU is required but not reachable (PARSER_PARSE_MODE=mineru). "
                    f"Start MinerU and verify `curl {health_url}` returns 200/healthy; "
                    "otherwise the service cannot start/index. "
                    "If you want to bypass MinerU, switch PARSER_PARSE_MODE to native/dotsocr."
                )
                raise RuntimeError(f"Dependency health check failed: {error_summary}. {hint}")
            raise RuntimeError(f"Dependency health check failed: {error_summary}")

    return {"mode": mode, "ok": ok, "checks": checks}


def format_dependency_failures(health: Dict[str, Any]) -> Optional[str]:
    checks = health.get("checks") or {}
    failures = [(name, res.get("error")) for name, res in checks.items() if not res.get("ok")]
    if not failures:
        return None
    return "; ".join(f"{name}: {err}" for name, err in failures)
