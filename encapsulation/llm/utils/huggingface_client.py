"""HuggingFace client creation utilities"""

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)


def create_sentence_transformer_client(config) -> Any:
    """
    Create SentenceTransformer client for embedding models

    Args:
        config: Configuration object with SentenceTransformer settings

    Returns:
        Configured SentenceTransformer client instance

    Raises:
        ImportError: If sentence-transformers library is not available
        Exception: If client creation fails
    """
    try:
        import sentence_transformers

        model_name = getattr(config, 'model_name', 'sentence-transformers/all-mpnet-base-v2')
        device = getattr(config, 'device', 'cpu')
        cache_folder = getattr(config, 'cache_folder', None)
        model_kwargs = getattr(config, 'model_kwargs', {})

        client = sentence_transformers.SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            device=device,
            **model_kwargs
        )

        logger.info(f"SentenceTransformer client initialized: {model_name}")
        return client

    except ImportError:
        logger.error("sentence-transformers library required for SentenceTransformer task")
        raise ImportError("sentence-transformers required for SentenceTransformer task")
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformer client: {str(e)}")
        raise


def _create_transformers_base(config, tokenizer_class, model_class, **kwargs):
    """
    Internal base function for transformers model creation

    Args:
        config: Configuration object
        tokenizer_class: Tokenizer class (AutoTokenizer, AutoProcessor, etc.)
        model_class: Model class (AutoModelForCausalLM, etc.)
        **kwargs: Additional arguments for tokenizer_kwargs and model_kwargs

    Returns:
        Tuple of (model, tokenizer/processor)
    """
    try:
        model_path = getattr(config, 'model_path', getattr(config, 'model_name', 'Qwen/qwen_reranker_0.6B'))
        device = getattr(config, 'device', 'cpu')
        cache_folder = getattr(config, 'cache_folder', None)
        model_kwargs = getattr(config, 'model_kwargs', {})

        # Merge with provided model_kwargs
        final_model_kwargs = {**model_kwargs, **kwargs.get('model_kwargs', {})}
        tokenizer_kwargs = kwargs.get('tokenizer_kwargs', {})

        # Create tokenizer/processor
        tokenizer = tokenizer_class.from_pretrained(
            model_path,
            cache_dir=cache_folder,
            trust_remote_code=True,
            **tokenizer_kwargs
        )

        # Create model
        model = model_class.from_pretrained(
            model_path,
            cache_dir=cache_folder,
            trust_remote_code=True,
            **final_model_kwargs
        )
        model.to(device)

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to create transformers model: {str(e)}")
        raise


def create_transformers_client(config) -> Tuple[Any, Any]:
    """
    Create Transformers client (model + tokenizer) for any Transformers-based model

    Args:
        config: Configuration object with Transformers settings

    Returns:
        Tuple of (model, tokenizer) instances

    Raises:
        ImportError: If transformers library is not available
        Exception: If client creation fails
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info(f"Transformers client initialized: {getattr(config, 'model_name', 'unknown')}")
        return _create_transformers_base(
            config,
            AutoTokenizer,
            AutoModelForCausalLM,
            tokenizer_kwargs={'padding_side': 'left'},
            model_kwargs={'torch_dtype': torch.float16}
        )

    except ImportError:
        logger.error("transformers library required for Transformers task")
        raise ImportError("transformers required for Transformers task")


def create_vision_language_model(config) -> Tuple[Any, Any, Any]:
    """
    Create Vision-Language model for local inference (e.g., DotsOCR, Qwen-VL, etc.)

    Args:
        config: Configuration object with HuggingFace settings

    Returns:
        Tuple of (model, processor, process_vision_info) instances

    Raises:
        ImportError: If required libraries are not available
        Exception: If model creation fails
    """
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from qwen_vl_utils import process_vision_info
        import torch

        model, processor = _create_transformers_base(
            config,
            AutoProcessor,
            AutoModelForCausalLM,
            tokenizer_kwargs={'use_fast': True},
            model_kwargs={
                'attn_implementation': "flash_attention_2",
                'torch_dtype': torch.bfloat16,
                'device_map': getattr(config, 'device', 'auto')
            }
        )

        logger.info(f"HuggingFace Vision-Language model loaded successfully from {getattr(config, 'model_path', 'unknown')}")
        return model, processor, process_vision_info

    except ImportError as e:
        logger.error(f"Required libraries not available for Vision-Language model: {str(e)}")
        raise ImportError(f"transformers, torch, qwen_vl_utils required for Vision-Language model: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load HuggingFace Vision-Language model: {str(e)}")
        raise


def setup_rerank_tokens(tokenizer, config, model_type="qwen"):
    """
    Setup rerank model tokens and templates based on model type

    Args:
        tokenizer: HuggingFace tokenizer instance
        config: Configuration object with model settings
        model_type: Type of rerank model ("qwen", "bge", "cohere", etc.)

    Returns:
        Dict with token IDs and tokenized templates

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == "qwen":
        # Qwen-specific binary classification tokens
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")

        # Qwen conversation format defaults
        default_prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        default_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    elif model_type.lower() == "bge":
        # BGE reranker tokens (example - adjust based on actual BGE model)
        token_false_id = tokenizer.convert_tokens_to_ids("irrelevant")
        token_true_id = tokenizer.convert_tokens_to_ids("relevant")

        # BGE format defaults
        default_prefix = "[CLS] Query: "
        default_suffix = " Document: "

    elif model_type.lower() == "cohere":
        # Cohere reranker tokens (example - adjust based on actual Cohere model)
        token_false_id = tokenizer.convert_tokens_to_ids("0")
        token_true_id = tokenizer.convert_tokens_to_ids("1")

        # Cohere format defaults
        default_prefix = "Query: "
        default_suffix = " Document: "

    else:
        raise ValueError(f"Unsupported rerank model type: {model_type}")

    # Generic template processing - allow config override
    prefix = getattr(config, 'prefix', default_prefix)
    suffix = getattr(config, 'suffix', default_suffix)

    # Tokenize prefix and suffix
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    return {
        'token_false_id': token_false_id,
        'token_true_id': token_true_id,
        'prefix': prefix,
        'suffix': suffix,
        'prefix_tokens': prefix_tokens,
        'suffix_tokens': suffix_tokens,
        'model_type': model_type
    }