"""Central defaults for chunk ingestion (post-parse storage stage).

Why this lives in config/:
- Avoid scattered, implicit behavior switches inside pipeline code.
- Keep ingestion behavior reviewable and (later) tunable from a single source of truth.
"""

# Batch chunk store validation:
# - Single-chunk `store_chunk()` can validate metadata+blob after each write (safer but slow).
# - Batch ingestion already relies on DB commit + object-store write acknowledgements.
#   Per-chunk validation becomes very expensive at scale and is typically redundant.
CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT = False

# Blob store write parallelism during batch chunk ingestion.
# This primarily helps when the blob store is remote (e.g., MinIO/S3) and per-object PUT latency dominates.
# Keep a conservative default to avoid overwhelming object storage / network.
CHUNK_BATCH_BLOB_WRITE_MAX_WORKERS_DEFAULT = 8

# Parser label used when indexing begins from an already-parsed markdown string.
# This is only a metadata label stored alongside parsed content; it does not affect chunking/indexing logic.
PARSED_MARKDOWN_PARSER_TYPE_DEFAULT = "preparsed_markdown"
