"""
RAG Inference Prompt Loader

根据 USER_TYPE 环境变量从 YAML 配置文件中加载对应的 prompt layer，并按 profile 进行拼接。
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple

from .rag_inference_prompts import (
    RAG_CHAT_BASE_SYSTEM_PROMPT_EN,
    RAG_CHAT_CITATION_ADDON_SYSTEM_PROMPT_EN,
    RAG_INFERENCE_CITATION_SYSTEM_PROMPT_EN,
)
from .runtime_context import current_local_date_str, prepend_today_line

logger = logging.getLogger(__name__)

# 默认 prompt（当找不到 YAML 配置/或配置不合法时使用）
# - rag_inference: base + citation addon（与历史行为一致）
# - cli: base only
_DEFAULT_SYSTEM_PROMPT_RAG_INFERENCE = RAG_INFERENCE_CITATION_SYSTEM_PROMPT_EN
_DEFAULT_SYSTEM_PROMPT_CLI = RAG_CHAT_BASE_SYSTEM_PROMPT_EN

# 缓存加载的 prompt
_cached_prompt: Optional[str] = None
_cached_key: Optional[Tuple[str, int, str]] = None


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


def _load_prompt_layers_from_yaml(user_type: int) -> Dict[str, str]:
    """
    从 YAML 文件加载指定 user_type 的 prompt layers。

    Backward-compatible:
    - If `system_prompt` exists: treat it as a FULL system prompt override.
    - Otherwise, treat `style_prompt` / `domain_prompt` / `domain_prompt_path` as optional layers.
    """
    yaml_path = _get_yaml_path()
    
    if not yaml_path.exists():
        logger.warning(
            "RAG inference prompts YAML file not found at %s, using default prompt",
            yaml_path
        )
        return {}
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data or "prompts" not in data:
            logger.warning(
                "Invalid YAML structure in %s, using default prompt",
                yaml_path
            )
            return {}
        
        # 查找匹配的 user_type
        for prompt_config in data["prompts"]:
            if prompt_config.get("type") == user_type:
                if not isinstance(prompt_config, dict):
                    continue
                # Legacy: full system prompt override.
                system_prompt = str(prompt_config.get("system_prompt") or "").strip()
                if system_prompt:
                    logger.info(
                        "Loaded RAG inference prompt for user_type=%d from %s",
                        user_type,
                        yaml_path
                    )
                    return {"system_prompt": system_prompt}

                style_prompt = str(prompt_config.get("style_prompt") or "").strip()
                domain_prompt = str(prompt_config.get("domain_prompt") or "").strip()
                domain_prompt_path = str(prompt_config.get("domain_prompt_path") or "").strip()
                if domain_prompt_path:
                    p = Path(domain_prompt_path)
                    if not p.is_absolute():
                        # Interpret as repo-relative path.
                        p = (yaml_path.parent.parent.parent / domain_prompt_path).resolve()
                    try:
                        if p.exists():
                            domain_prompt = p.read_text(encoding="utf-8", errors="replace").strip() or domain_prompt
                        else:
                            logger.warning("domain_prompt_path does not exist: %s", p)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to read domain_prompt_path=%s: %s", p, exc)

                out: Dict[str, str] = {}
                if style_prompt:
                    out["style_prompt"] = style_prompt
                if domain_prompt:
                    out["domain_prompt"] = domain_prompt
                return out
        
        # 如果找不到匹配的 type，使用默认 prompt
        logger.warning(
            "No prompt found for user_type=%d in %s, using default prompt",
            user_type,
            yaml_path
        )
        return {}
        
    except Exception as e:
        logger.error(
            "Failed to load prompt from %s: %s, using default prompt",
            yaml_path,
            e,
            exc_info=True
        )
        return {}


def _resolve_user_type(user_type: Optional[int]) -> int:
    if user_type is not None:
        return int(user_type)
    user_type_str = os.getenv("USER_TYPE", "0")
    try:
        return int(user_type_str)
    except ValueError:
        logger.warning("Invalid USER_TYPE environment variable: %s, defaulting to 0", user_type_str)
        return 0


def get_rag_chat_system_prompt(
    *,
    profile: Literal["cli", "rag_inference"],
    user_type: Optional[int] = None,
) -> str:
    """
    获取 RAG chat 的 system prompt（按 profile 拼接）。
    
    Args:
        profile: "cli" (debug) | "rag_inference" (user-facing).
        user_type: 用户类型（如 0/1）。如果为 None，则从环境变量 USER_TYPE 读取。
    
    Returns:
        System prompt 字符串
    """
    resolved_user_type = _resolve_user_type(user_type)
    
    # 使用缓存避免重复加载
    global _cached_prompt, _cached_key
    cache_key = (str(profile), int(resolved_user_type), current_local_date_str())
    if _cached_prompt is not None and _cached_key == cache_key:
        return _cached_prompt
    
    layers = _load_prompt_layers_from_yaml(resolved_user_type)

    # CLI: base only (debug; no extra constraints).
    if profile == "cli":
        prompt = _DEFAULT_SYSTEM_PROMPT_CLI
        # Allow legacy full override only if explicitly configured.
        legacy = str(layers.get("system_prompt") or "").strip()
        if legacy:
            prompt = legacy
    else:
        # rag_inference: base + citation addon + optional style/domain layers.
        legacy = str(layers.get("system_prompt") or "").strip()
        if legacy:
            prompt = legacy
        else:
            parts = [RAG_CHAT_BASE_SYSTEM_PROMPT_EN, RAG_CHAT_CITATION_ADDON_SYSTEM_PROMPT_EN]
            style = str(layers.get("style_prompt") or "").strip()
            domain = str(layers.get("domain_prompt") or "").strip()
            if style:
                parts.append(style)
            if domain:
                parts.append(domain)
            prompt = "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()
    
    prompt = prepend_today_line(prompt)

    # 更新缓存
    _cached_prompt = prompt
    _cached_key = cache_key
    
    return prompt


def get_rag_inference_system_prompt(user_type: Optional[int] = None) -> str:
    """Backward-compatible entrypoint for rag_inference module."""
    return get_rag_chat_system_prompt(profile="rag_inference", user_type=user_type)
