"""Web search clients (e.g., Tavily) shared across DeepSearch and HippoRAG Q&A."""

from .tavily_client import TavilySearchClient, TavilySearchResult

__all__ = ["TavilySearchClient", "TavilySearchResult"]

