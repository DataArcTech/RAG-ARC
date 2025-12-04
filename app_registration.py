import logging
import os
from pathlib import Path
from config.application.account_config import AccountConfig
from config.application.knowledge_config import KnowledgeConfig
from config.application.rag_inference_config import RAGInferenceConfig
from config.application.session_config import ChatSessionConfig
from config.application.chat_message_config import ChatMessageManagerConfig
from framework.register import Register

# Set up logging with environment variable support
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

registrator = Register()
BASE_DIR = Path(__file__).resolve().parent


def _normalize_config_path(path_value: str) -> str:
    """Return absolute path for config files."""
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path_value
    return str(path)


def _resolve_config_path(env_var: str, fallback: str, profile_map: dict | None = None) -> str:
    """Resolve config path using env override and optional profile defaults."""
    env_override = os.getenv(env_var)
    if env_override:
        return _normalize_config_path(env_override)

    if profile_map:
        profile = os.getenv("MODEL_PROFILE", "api").lower()
        profile_path = profile_map.get(profile)
        if profile_path:
            normalized = _normalize_config_path(profile_path)
            if os.path.exists(normalized):
                return normalized
        # Fall back to api profile if requested profile is unavailable
        api_default = profile_map.get("api")
        if api_default:
            normalized = _normalize_config_path(api_default)
            if os.path.exists(normalized):
                return normalized

    return _normalize_config_path(fallback)


def _resolve_provider(env_var: str, api_default: str, local_default: str) -> str:
    env_value = os.getenv(env_var)
    if env_value:
        return env_value.lower()
    profile = os.getenv("MODEL_PROFILE", "api").lower()
    return local_default if profile == "local" else api_default

def initialize():
    try:
        rag_config_path = _resolve_config_path(
            "RAG_INFERENCE_CONFIG_PATH",
            "config/json_configs/rag_inference.json",
            {
                "api": "config/json_configs/rag_inference.json",
                "local": "config/json_configs/rag_inference_local.json"
            }
        )
        knowledge_config_path = _resolve_config_path(
            "KNOWLEDGE_CONFIG_PATH",
            "config/json_configs/knowledge.json",
            {
                "api": "config/json_configs/knowledge.json",
                "local": "config/json_configs/knowledge_local.json"
            }
        )

        profile = os.getenv("MODEL_PROFILE", "api").lower()
        logger.info(
            "MODEL_PROFILE=%s | rag_config=%s | knowledge_config=%s",
            profile,
            rag_config_path,
            knowledge_config_path
        )
        chat_provider = _resolve_provider("CHAT_MODEL_PROVIDER", "openai", "huggingface")
        embedding_provider = _resolve_provider("EMBEDDING_MODEL_PROVIDER", "openai", "huggingface")
        ocr_provider = _resolve_provider("OCR_MODEL_PROVIDER", "openai", "vllm") if profile == "api" else "dots_ocr_parser"
        reranker_provider = "listwise(chat)" if profile != "local" else f"qwen_local({os.getenv('RERANKER_MODEL_NAME', 'Qwen/Qwen3-Reranker-0.6B')})"
        logger.info(
            "Provider summary -> chat=%s, embedding=%s, ocr=%s, reranker=%s",
            chat_provider,
            embedding_provider,
            ocr_provider,
            reranker_provider
        )

        registrator.register(config_path=rag_config_path, app_name="rag_inference", config_type=RAGInferenceConfig)
        registrator.register(config_path=knowledge_config_path, app_name="knowledge", config_type=KnowledgeConfig)
        registrator.register(
            config_path=_resolve_config_path("ACCOUNT_CONFIG_PATH", "config/json_configs/account.json"),
            app_name="account",
            config_type=AccountConfig
        )
        registrator.register(
            config_path=_resolve_config_path("SESSION_CONFIG_PATH", "config/json_configs/session.json"),
            app_name="chat_session",
            config_type=ChatSessionConfig
        )
        registrator.register(
            config_path=_resolve_config_path("CHAT_MESSAGE_CONFIG_PATH", "config/json_configs/chat_message.json"),
            app_name="chat_message",
            config_type=ChatMessageManagerConfig
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG inference: {e}")
        # Continue without RAG inference for now
