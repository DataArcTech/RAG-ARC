import logging
import os
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

def initialize():
    try:
        registrator.register(config_path="config/json_configs/rag_inference.json", app_name="rag_inference", config_type=RAGInferenceConfig)
        registrator.register(config_path="config/json_configs/knowledge.json", app_name="knowledge", config_type=KnowledgeConfig)
        registrator.register(config_path="config/json_configs/account.json", app_name="account", config_type=AccountConfig)
        registrator.register(config_path="config/json_configs/session.json", app_name="chat_session", config_type=ChatSessionConfig)
        registrator.register(config_path="config/json_configs/chat_message.json", app_name="chat_message", config_type=ChatMessageManagerConfig)
    except Exception as e:
        logger.error(f"Failed to initialize RAG inference: {e}")
        # Continue without RAG inference for now
