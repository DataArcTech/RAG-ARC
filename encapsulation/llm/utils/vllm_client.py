"""VLLM client creation utilities"""

import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def create_vllm_client(config) -> Any:
    """
    Create VLLM client for server-based inference

    Args:
        config: Configuration object with VLLM settings

    Returns:
        Configured OpenAI client for VLLM server

    Raises:
        ImportError: If openai library is not available
        Exception: If client creation fails
    """
    try:
        from openai import OpenAI

        # Get base_url and api_key from config
        base_url = getattr(config, 'base_url', "http://localhost:8000/v1")
        api_key = getattr(config, 'api_key', "sk-xxx")

        client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"VLLM client initialized - connecting to {base_url}")
        return client

    except ImportError:
        logger.error("openai library required for VLLM client")
        raise ImportError("openai required for VLLM client")
    except Exception as e:
        logger.error(f"Failed to initialize VLLM client: {str(e)} Make sure you launch vllm server using vllm_launch.py")
        raise