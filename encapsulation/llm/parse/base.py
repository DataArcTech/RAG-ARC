from abc import abstractmethod
from typing import Any, Optional
from PIL import Image

from framework.module import AbstractModule


class ParseLLMBase(AbstractModule):
    """
    Abstract base class for LLM-based parsing service wrappers.

    This class defines the interface for LLM services that can perform
    vision-language inference on images with text prompts. Implementations
    should handle model loading and raw inference only - no business logic.
    """

    @abstractmethod
    def infer(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs
    ) -> str:
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
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dict containing model information (name, loading method, etc.)
        """
        pass