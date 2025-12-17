from typing import Literal

from framework.config import AbstractConfig
from core.query_rewrite.noop_rewriter import NoOpQueryRewriter


class NoOpQueryRewriterConfig(AbstractConfig):
    type: Literal["noop_query_rewriter"] = "noop_query_rewriter"

    def build(self) -> NoOpQueryRewriter:
        return NoOpQueryRewriter(self)

