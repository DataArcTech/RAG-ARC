try:
    from tantivy import (  # type: ignore
        Filter,
        Index,
        Occur,
        Order,
        Query,
        SchemaBuilder,
        Document as TantivyDocument,
        TextAnalyzerBuilder,
        Tokenizer,
    )
except ImportError as exc:
    raise ImportError("Please install dependencies with: uv sync") from exc

