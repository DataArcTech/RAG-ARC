import logging
import os
from config.application.rag_inference_config import RAGInferenceConfig
from framework.register import Register

# Set up logging with environment variable support
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

registrator = Register()

def initialize():
    try:
        registrator.register(config_path="config/json_configs/rag_inference.json", app_name="rag_inference", config_type=RAGInferenceConfig)
    except Exception as e:
        logger.error(f"Failed to initialize RAG inference: {e}")
        # Continue without RAG inference for now
