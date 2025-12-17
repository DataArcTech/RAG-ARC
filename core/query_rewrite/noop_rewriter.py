from framework.module import AbstractModule


class NoOpQueryRewriter(AbstractModule):
    def rewrite_query(self, query: str) -> str:
        return str(query or "").strip()

