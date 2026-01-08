"""
RAG Inference Prompt Loader

根据 USER_TYPE 环境变量从 YAML 配置文件中加载对应的 system prompt。
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 默认 prompt（当找不到配置时使用）
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful RAG assistant.\n"
    "You may be given a list of numbered Sources (key=1..N).\n"
    "Rules:\n"
    "1) If the user message is just a greeting / test / acknowledgement (e.g. '测试', 'test', 'hello', 'hi', '你好'),\n"
    "   answer briefly and DO NOT use any Sources and DO NOT include any <sup> tags.\n"
    "2) If Sources are provided (the list is not empty), ground your answer in Sources and add inline citations using HTML <sup> tags.\n"
    "   - Every sentence that contains factual information supported by Sources MUST end with one or more <sup>key</sup>.\n"
    "   - Cite only the minimal number of sources needed; do NOT cite all sources by default.\n"
    "   - Do NOT output a bare block/list of citations (e.g. '<sup>1</sup><sup>2</sup>...') without nearby supporting text.\n"
    "   - Do NOT cite a source you did not use.\n"
    "3) If NO Sources are provided (the list is empty), DO NOT use any <sup> tags in your answer.\n"
    "   - Say you don't know or cannot answer based on the available information.\n"
    "   - Do NOT make up citations or use <sup> tags when there are no Sources.\n"
    "4) If Sources are provided but none are relevant, say you don't know based on the provided Sources and ask a clarifying question.\n"
    "5) Do NOT use bracket citations like [1] and do NOT add a trailing 'Sources:' section.\n"
    "6) Output in Markdown. The only HTML allowed is <sup>...</sup>.\n"
)

# 缓存加载的 prompt
_cached_prompt: Optional[str] = None
_cached_user_type: Optional[int] = None


def _get_yaml_path() -> Path:
    """获取 YAML 配置文件的路径"""
    # 优先使用环境变量指定的路径
    yaml_path = os.getenv("RAG_INFERENCE_PROMPTS_YAML_PATH")
    if yaml_path:
        return Path(yaml_path)
    
    # 默认路径：项目根目录下的 config/prompts/rag_inference_prompts.yaml
    # 尝试从当前文件位置推断项目根目录
    current_file = Path(__file__)
    # core/prompts/rag_inference_prompt_loader.py -> ../../config/prompts/rag_inference_prompts.yaml
    default_path = current_file.parent.parent.parent / "config" / "prompts" / "rag_inference_prompts.yaml"
    return default_path


def _load_prompt_from_yaml(user_type: int) -> str:
    """从 YAML 文件加载指定 user_type 的 prompt"""
    yaml_path = _get_yaml_path()
    
    if not yaml_path.exists():
        logger.warning(
            "RAG inference prompts YAML file not found at %s, using default prompt",
            yaml_path
        )
        return _DEFAULT_SYSTEM_PROMPT
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data or "prompts" not in data:
            logger.warning(
                "Invalid YAML structure in %s, using default prompt",
                yaml_path
            )
            return _DEFAULT_SYSTEM_PROMPT
        
        # 查找匹配的 user_type
        for prompt_config in data["prompts"]:
            if prompt_config.get("type") == user_type:
                system_prompt = prompt_config.get("system_prompt", "").strip()
                if system_prompt:
                    logger.info(
                        "Loaded RAG inference prompt for user_type=%d from %s",
                        user_type,
                        yaml_path
                    )
                    return system_prompt
        
        # 如果找不到匹配的 type，使用默认 prompt
        logger.warning(
            "No prompt found for user_type=%d in %s, using default prompt",
            user_type,
            yaml_path
        )
        return _DEFAULT_SYSTEM_PROMPT
        
    except Exception as e:
        logger.error(
            "Failed to load prompt from %s: %s, using default prompt",
            yaml_path,
            e,
            exc_info=True
        )
        return _DEFAULT_SYSTEM_PROMPT


def get_rag_inference_system_prompt(user_type: Optional[int] = None) -> str:
    """
    获取 RAG inference 的 system prompt。
    
    Args:
        user_type: 用户类型（0 或 1）。如果为 None，则从环境变量 USER_TYPE 读取。
    
    Returns:
        System prompt 字符串
    """
    # 如果未指定 user_type，从环境变量读取
    if user_type is None:
        user_type_str = os.getenv("USER_TYPE", "0")
        try:
            user_type = int(user_type_str)
        except ValueError:
            logger.warning(
                "Invalid USER_TYPE environment variable: %s, defaulting to 0",
                user_type_str
            )
            user_type = 0
    
    # 使用缓存避免重复加载
    global _cached_prompt, _cached_user_type
    if _cached_prompt is not None and _cached_user_type == user_type:
        return _cached_prompt
    
    # 加载 prompt
    prompt = _load_prompt_from_yaml(user_type)
    
    # 更新缓存
    _cached_prompt = prompt
    _cached_user_type = user_type
    
    return prompt

