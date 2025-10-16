import logging
from typing import Any
from PIL import Image
import base64
from io import BytesIO

from .base import ParseLLMBase
from ..utils.vllm_client import create_vllm_client
from ..utils.openai_client import create_openai_sync_client
from framework.singleton_decorator import singleton

logger = logging.getLogger(__name__)


@singleton
class VLMOcrLLM(ParseLLMBase):
    """
    VLM OCR LLM service wrapper for vision-language inference.

    This class provides a thin wrapper around VLM (or compatible) model inference,
    supporting both OpenAI API and VLLM server deployment.
    Handles only model loading and raw inference - no parsing business logic.

    Key features:
    - Dual inference modes: OpenAI API and VLLM (server-based)
    - Pure service interface: (image, prompt) → raw response
    - Model configuration and client management
    - Error handling for inference failures

    Configuration:
        loading_method: "openai" or "vllm"
        For OpenAI: openai_api_key, openai_base_url, model_name
        For VLLM: base_url, api_key, model_name, temperature, etc.
    """

    def __init__(self, config):
        """Initialize VLM OCR LLM with loading method support"""
        super().__init__(config)

        self.loading_method = getattr(self.config, 'loading_method', 'openai')

        # Initialize based on loading method
        if self.loading_method == 'openai':
            logger.info("Initializing OpenAI client for VLM Vision inference")
            self.client = create_openai_sync_client(self.config)
        elif self.loading_method == 'vllm':
            logger.info("Initializing VLLM client for server-based inference")
            self.client = create_vllm_client(self.config)
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    def infer(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Perform vision-language inference on an image with a prompt.

        Args:
            image: PIL Image to process
            prompt: Text prompt for the model
            **kwargs: Additional inference parameters

        Returns:
            Raw model response as string

        Raises:
            Exception: If inference fails
        """
        if self.loading_method == 'openai':
            return self._inference_with_openai(image, prompt, **kwargs)
        elif self.loading_method == 'vllm':
            return self._inference_with_vllm(image, prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dict containing model information
        """
        info = {
            "loading_method": self.loading_method,
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }

        if self.loading_method == 'openai':
            info.update({
                "base_url": getattr(self.config, 'openai_base_url', 'unknown'),
                "model_name": getattr(self.config, 'model_name', 'unknown'),
                "provider": "openai"
            })
        elif self.loading_method == 'vllm':
            info.update({
                "base_url": getattr(self.config, 'base_url', 'unknown'),
                "model_name": getattr(self.config, 'model_name', 'unknown'),
                "provider": "vllm"
            })

        return info

    # ==================== PRIVATE INFERENCE METHODS ====================

    def _inference_with_openai(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """Internal OpenAI API inference method"""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": self._pil_image_to_base64(image)},
                },
                {"type": "text", "text": prompt}
            ],
        }]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=getattr(self.config, 'model_name', 'gpt-4o'),
                max_tokens=getattr(self.config, 'max_tokens', 4096),
                temperature=getattr(self.config, 'temperature', 0.0),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API request error: {e}")
            raise

    def _inference_with_vllm(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """Internal VLLM inference method"""
        import requests

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": self._pil_image_to_base64(image)},
                },
                {"type": "text", "text": prompt}
            ],
        }]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=getattr(self.config, 'model_name', 'model'),
                max_completion_tokens=getattr(self.config, 'max_completion_tokens', 4096),
                temperature=getattr(self.config, 'temperature', 0.0),
                top_p=getattr(self.config, 'top_p', 1.0)
            )
            return response.choices[0].message.content
        except requests.exceptions.RequestException as e:
            logger.error(f"VLLM request error: {e}")
            raise

    def _pil_image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 data URL format for LLM service calls"""
        buffered = BytesIO()
        image.save(buffered, format=format)
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{base64_str}"

