"""Web search clients (e.g., Tavily) shared across DeepSearch and HippoRAG Q&A."""

from .aggregation import WebSearchAggregationStats, aggregate_tavily_results
from .tavily_client import TavilySearchClient, TavilySearchResult

__all__ = [
    "TavilySearchClient",
    "TavilySearchResult",
    "WebSearchAggregationStats",
    "aggregate_tavily_results",
]
