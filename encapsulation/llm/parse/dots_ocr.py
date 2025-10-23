import logging
from typing import Any, TYPE_CHECKING
from PIL import Image
import base64
from io import BytesIO

from .base import ParseLLMBase
from ..utils.vllm_client import create_vllm_client
from ..utils.huggingface_client import create_vision_language_model
from framework.singleton_decorator import singleton

if TYPE_CHECKING:
    from config.encapsulation.llm.parse.dots_ocr import DotsOCRConfig

logger = logging.getLogger(__name__)


@singleton
class DotsOCRLLM(ParseLLMBase):
    """
    DotsOCR LLM service wrapper for vision-language inference.

    This class provides a thin wrapper around DotsOCR model inference,
    supporting both local HuggingFace inference and remote VLLM server deployment.
    Handles only model loading and raw inference - no parsing business logic.

    Key features:
    - Dual inference modes: HuggingFace (local) and VLLM (server-based)
    - Pure service interface: (image, prompt) → raw response
    - Model configuration and client management
    - Error handling for inference failures

    Configuration:
        loading_method: "huggingface" or "vllm"
        For HuggingFace: model_path, device
        For VLLM: base_url, api_key, model_name, temperature, etc.
    """

    def __init__(self, config: "DotsOCRConfig"):
        """Initialize DotsOCR LLM with loading method support"""
        super().__init__(config)

        self.loading_method = getattr(self.config, 'loading_method', 'huggingface')

        # Initialize based on loading method
        if self.loading_method == 'huggingface':
            logger.info("Initializing HuggingFace model for local inference")
            self.model, self.processor, self.process_vision_info = create_vision_language_model(self.config)
        elif self.loading_method == 'vllm':
            logger.info("Initializing VLLM client for server-based inference")
            self.vllm_client = create_vllm_client(self.config)
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
        if self.loading_method == 'huggingface':
            return self._inference_with_hf(image, prompt, **kwargs)
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

        if self.loading_method == 'huggingface':
            info.update({
                "model_path": getattr(self.config, 'model_path', 'unknown'),
                "device": getattr(self.config, 'device', 'auto'),
                "provider": "huggingface"
            })
        elif self.loading_method == 'vllm':
            info.update({
                "base_url": getattr(self.config, 'base_url', 'unknown'),
                "model_name": getattr(self.config, 'model_name', 'unknown'),
                "provider": "vllm"
            })

        return info

    # ==================== PRIVATE INFERENCE METHODS ====================

    def _inference_with_hf(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """Internal HuggingFace inference method"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(getattr(self.config, 'device', 'cuda'))

        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

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
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
            ],
        }]

        try:
            response = self.vllm_client.chat.completions.create(
                messages=messages,
                model=getattr(self.config, 'model_name', 'model'),
                max_completion_tokens=getattr(self.config, 'max_completion_tokens', 16384),
                temperature=getattr(self.config, 'temperature', 0.1),
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