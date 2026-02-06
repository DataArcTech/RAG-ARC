"""Central defaults for embedding request shaping.

Why this lives in config/:
- Avoid scattered magic numbers in client code.
- Keep performance-sensitive defaults reviewable in one place.

These defaults apply to OpenAI-compatible embedding calls (OpenAI/OpenRouter/gateways).
"""

# Default batch size for embedding requests when batching is supported.
# Larger batches reduce per-request overhead and typically improve indexing throughput,
# but some gateways may time out on very large batches; override via EMBEDDING_REQUEST_BATCH_SIZE.
EMBEDDING_REQUEST_BATCH_SIZE_DEFAULT = 128

