"""OpenAI client creation utilities"""

import openai
import logging

logger = logging.getLogger(__name__)


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
    base_url = config.openai_base_url
    organization = getattr(config, 'organization', None)
    max_retries = getattr(config, 'max_retries', 3)
    timeout = getattr(config, 'timeout', 60.0)

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout
        )
        logger.info("OpenAI sync client initialized successfully")
        return client
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
    base_url = config.openai_base_url
    organization = getattr(config, 'organization', None)
    max_retries = getattr(config, 'max_retries', 3)
    timeout = getattr(config, 'timeout', 30)

    try:
        async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout
        )
        logger.info("OpenAI async client initialized successfully")
        return async_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI async client: {str(e)}")
        raise