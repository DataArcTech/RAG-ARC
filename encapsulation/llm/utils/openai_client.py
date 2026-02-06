"""OpenAI client creation utilities"""
import openai
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_OPENAI_CLIENT_CACHE_MAXSIZE = 32


@lru_cache(maxsize=_OPENAI_CLIENT_CACHE_MAXSIZE)
def _cached_sync_client(
    *,
    api_key: str,
    base_url: Optional[str],
    organization: Optional[str],
    max_retries: int,
    timeout: float,
) -> openai.OpenAI:
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_retries=max_retries,
        timeout=timeout,
    )
    logger.info("OpenAI sync client initialized successfully (cached)")
    return client


@lru_cache(maxsize=_OPENAI_CLIENT_CACHE_MAXSIZE)
def _cached_async_client(
    *,
    api_key: str,
    base_url: Optional[str],
    organization: Optional[str],
    max_retries: int,
    timeout: float,
) -> openai.AsyncOpenAI:
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        max_retries=max_retries,
        timeout=timeout,
    )
    logger.info("OpenAI async client initialized successfully (cached)")
    return client


def clear_openai_client_cache() -> None:
    """Clear in-process OpenAI client caches (useful for tests)."""
    _cached_sync_client.cache_clear()
    _cached_async_client.cache_clear()


def create_openai_sync_client(config) -> openai.OpenAI:
    """
    Create OpenAI synchronous client with common configuration

    Args:
        config: Configuration object with OpenAI settings

    Returns:
        Configured OpenAI client instance

    Raises:
        Exception: If client creation fails
    """
    # Extract OpenAI-specific config parameters
    api_key = config.openai_api_key
    base_url = getattr(config, 'openai_base_url', None) or None
    organization = getattr(config, 'organization', None)
    max_retries = getattr(config, 'max_retries', 3)
    timeout = getattr(config, 'timeout', 60.0)

    try:
        return _cached_sync_client(
            api_key=str(api_key),
            base_url=str(base_url) if base_url else None,
            organization=str(organization) if organization else None,
            max_retries=int(max_retries),
            timeout=float(timeout),
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI sync client: {str(e)}")
        raise


def create_openai_async_client(config) -> openai.AsyncOpenAI:
    """
    Create OpenAI asynchronous client with common configuration

    Args:
        config: Configuration object with OpenAI settings

    Returns:
        Configured OpenAI async client instance

    Raises:
        Exception: If client creation fails
    """
    # Extract OpenAI-specific config parameters (same as sync client)
    api_key = config.openai_api_key
    base_url = getattr(config, 'openai_base_url', None) or None
    organization = getattr(config, 'organization', None)
    max_retries = getattr(config, 'max_retries', 3)
    timeout = getattr(config, 'timeout', 30)

    try:
        return _cached_async_client(
            api_key=str(api_key),
            base_url=str(base_url) if base_url else None,
            organization=str(organization) if organization else None,
            max_retries=int(max_retries),
            timeout=float(timeout),
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI async client: {str(e)}")
        raise
