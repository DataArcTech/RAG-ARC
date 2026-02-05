"""Central defaults for embedding retry behavior.

Why this lives in config/:
- Avoid scattered magic constants in client code.
- Keep behavior tunable and reviewable without adding .env knobs.

These defaults apply to OpenAI-compatible embedding calls (including OpenRouter/OpenAI gateways).
They are intended to handle transient network failures (timeouts, connection resets, 5xx).
Rate-limit (429) backoff remains handled separately by the embedding client.
"""

# How many times to retry on transient (non-429) failures.
EMBEDDING_TRANSIENT_MAX_RETRIES = 2

# Exponential backoff base (seconds). Sleep pattern:
#   min(max, base * (2**attempt)) + uniform(0, jitter)
EMBEDDING_TRANSIENT_BACKOFF_INITIAL_SECONDS = 1.0
EMBEDDING_TRANSIENT_BACKOFF_MAX_SECONDS = 20.0
EMBEDDING_TRANSIENT_BACKOFF_JITTER_SECONDS = 0.8

