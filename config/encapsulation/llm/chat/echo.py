from typing import Literal

from framework.config import AbstractConfig
from encapsulation.llm.chat.echo import EchoChat


class EchoChatConfig(AbstractConfig):
    type: Literal["echo_chat"] = "echo_chat"
    prefix: str = "ECHO:"
    chunk_chars: int = 16
    delay_s: float = 0.0

    def build(self) -> EchoChat:
        return EchoChat(self)

