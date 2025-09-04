from dataclasses import dataclass
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AbstractConfig


@dataclass
class AbstractModule(ABC):
    config: "AbstractConfig"
