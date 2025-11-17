# 1. Use an official Python base image
# This version uses CPU-only dependencies and calls external LLM APIs
FROM python:3.11-slim

# 2. Build arguments for region-specific configuration
ARG UV_INSTALL_URL=https://astral.sh/uv/install.sh
ARG UV_INDEX_URL=https://pypi.org/simple

# 3. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=300 \
    UV_INDEX_URL=${UV_INDEX_URL}

# 4. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    tzdata \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Install uv
RUN curl -LsSf ${UV_INSTALL_URL} | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv

# 6. Create and set working directory
WORKDIR /rag_arc

# 7. Copy only dependency files first (for better caching)
COPY pyproject.toml uv.lock /rag_arc/

# 8. Install Python dependencies using uv sync with virtual environment
# Use --no-cache and configure index URL based on region
RUN uv sync --no-cache --index-url ${UV_INDEX_URL}

# 9. Copy the rest of the project files (after dependencies are installed)
COPY . /rag_arc/

# 10. Reinstall the package in editable mode using uv sync
RUN uv sync --no-cache --index-url ${UV_INDEX_URL}

# 11. Create necessary directories
RUN mkdir -p \
    /rag_arc/data/parsed_files \
    /rag_arc/data/file_store \
    /rag_arc/data/chunk_store \
    /rag_arc/data/faiss_index \
    /rag_arc/data/unified_faiss_index \
    /rag_arc/data/unified_bm25_index \
    /rag_arc/data/graph_index \
    /rag_arc/data/graph_index_neo4j \
    /rag_arc/local/files \
    /rag_arc/models

# 12. Expose application port
EXPOSE 8000

# 13. Set PATH to include virtual environment
ENV PATH="/rag_arc/.venv/bin:$PATH"

# 14. Start the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
