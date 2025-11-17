# RAG-ARC Docker Deployment Guide

> 💡 **Two-step deployment: Build once, start anytime**

## 🚀 Quick Start

### Step 1: Build Docker Images

```bash
# 1. Clone RAG-ARC project
git clone https://github.com/DataArcTech/RAG-ARC.git
cd RAG-ARC

# 2. Build all Docker images
./build.sh
```

The build script will:
- ✅ Check Docker environment
- ✅ Create .env configuration file
- ✅ Select CPU or GPU mode (auto-detect NVIDIA GPU)
- ✅ Pull base images (PostgreSQL, Redis, Neo4j)
- ✅ Build RAG-ARC application image

### Step 2: Start All Services

```bash
# Start all containers
./start.sh
```

The start script will:
- ✅ Create Docker network
- ✅ Start PostgreSQL 16 database
- ✅ Start Redis 7 cache
- ✅ Start Neo4j graph database
- ✅ Start RAG-ARC application
- ✅ Verify all services are ready

**Note**: By default, Neo4j ports are **not exposed** to the host for security. To enable Neo4j Browser access, set `EXPOSE_NEO4J=true` in your `.env` file before running `./start.sh`

## 🎯 Deployment Architecture

**Full-stack deployment** with 4 containers:
- **PostgreSQL 16**: Metadata and application data storage
- **Redis 7**: Caching layer for performance
- **Neo4j**: Graph database for knowledge graph
- **RAG-ARC App**: FastAPI application with GPU support (if available)

**Benefits**:
- 🚀 Two-step deployment (build once, start anytime)
- 🔄 Easy restart without rebuilding
- 🎮 Automatic GPU detection and configuration
- 📦 All dependencies containerized
- 🔒 Isolated network for security

## 📦 What Gets Deployed

The deployment creates 4 containers:

1. **rag-arc-postgres** - PostgreSQL 16 database
   - Stores metadata, user data, and file information
   - Data persisted in Docker volume `rag-arc-postgres-data`
   - Network: `rag-arc-network`

2. **rag-arc-redis** - Redis 7 cache
   - Caches frequently accessed data
   - Data persisted in Docker volume `rag-arc-redis-data`
   - Network: `rag-arc-network`

3. **rag-arc-neo4j** - Neo4j graph database
   - Stores knowledge graph (entities, relations, facts)
   - Data persisted in Docker volumes `rag-arc-neo4j-data` and `rag-arc-neo4j-logs`
   - APOC plugin enabled for advanced graph operations
   - Network: `rag-arc-network`

4. **rag-arc-app** - RAG-ARC application
   - FastAPI application with GPU support (if available)
   - Connected to PostgreSQL, Redis, and Neo4j via Docker network
   - Volumes mounted: `./data`, `./local`, `./models`
   - Port: 8000 (configurable during startup)

## 🔧 Common Commands

### View Logs
```bash
# View application logs
docker logs -f rag-arc-app

# View PostgreSQL logs
docker logs -f rag-arc-postgres

# View Redis logs
docker logs -f rag-arc-redis

# View Neo4j logs
docker logs -f rag-arc-neo4j
```

### Manage Containers
```bash
# Stop all containers (keeps data)
./stop.sh

# Restart all services
./start.sh

# Cleanup Docker resources (keeps local data)
./cleanup.sh
# This removes containers, volumes, and network but keeps ./data, ./local, ./models

# Complete cleanup (⚠️ This will delete all data including local directories!)
./clean-docker-data.sh
# This removes everything: containers, volumes, and local data directories
```

### Rebuild Application
```bash
# Rebuild application image (after code changes)
./build.sh

# Then restart services
./start.sh
```

## 🌐 Access URLs

- **API Service**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/
- **Neo4j Browser**: http://localhost:7474 (only if `EXPOSE_NEO4J=true` in `.env`)
  - Username: `neo4j`
  - Password: `12345678` (or value set in `NEO4J_PASSWORD`)
  - To enable: Add `EXPOSE_NEO4J=true` to `.env` and restart with `./start.sh`

## ⚙️ Environment Configuration

### Required Configuration

Edit `.env` file before building:

```bash
# LLM API Configuration (Required)
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Neo4j Password (Recommended to change)
NEO4J_PASSWORD=12345678  # Change this to a secure password

# Local Model Configuration
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5 # or Qwen/Qwen3-Embedding-0.6B
DEVICE=cuda:0  # or cpu
```

### Auto-configured by Scripts

The following are automatically configured by the deployment scripts:

```bash
# PostgreSQL Configuration
POSTGRES_HOST=rag-arc-postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123
POSTGRES_DB=rag_arc

# Redis Configuration
REDIS_HOST=rag-arc-redis
REDIS_PORT=6379

# Neo4j Configuration
NEO4J_URL=bolt://rag-arc-neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DATABASE=neo4j

# Neo4j Port Exposure (optional, for development/debugging)
EXPOSE_NEO4J=false          # Set to true to access Neo4j Browser
NEO4J_HTTP_PORT=7474        # Neo4j Browser port (if exposed)
NEO4J_BOLT_PORT=7687        # Bolt protocol port (if exposed)
```

### Enable Neo4j Browser (Optional)

If you want to access Neo4j Browser for debugging or visualization:

1. Edit `.env` file:
```bash
EXPOSE_NEO4J=true
```

2. Restart services:
```bash
./start.sh
```

3. Access Neo4j Browser at http://localhost:7474
   - Username: `neo4j`
   - Password: `12345678`

**Note**: For production environments, keep `EXPOSE_NEO4J=false` for better security.

## 📝 System Requirements

### Minimum Requirements
- **Docker**: 20.10+
- **RAM**: 8GB+ recommended
- **Disk Space**: 20GB+ (for images, models, and data)
- **OS**: Linux, macOS, or Windows with WSL2

### For GPU Mode (Optional)
- **NVIDIA GPU**: CUDA-compatible GPU
- **NVIDIA Driver**: 525.60.13+ (for CUDA 12.1)
- **NVIDIA Docker Runtime**: nvidia-docker2

### Check GPU Support
```bash
# Check NVIDIA driver
nvidia-smi

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🔍 Troubleshooting

### Build Issues

**Problem**: Build fails with timeout errors
```bash
# Solution: Increase timeout in Dockerfile
# Or use a different PyPI mirror
# Edit Dockerfile and change the mirror URL
```

**Problem**: GPU build fails
```bash
# Solution: Fall back to CPU mode
# Select option 1 (CPU mode) when running ./build.sh
```

### Startup Issues

**Problem**: Port 8000 already in use
```bash
# Solution: Use a different port
# The start.sh script will prompt you to select a port
# Or manually specify: docker run -p 8080:8000 ...
```

**Problem**: PostgreSQL not ready
```bash
# Check PostgreSQL status
docker exec rag-arc-postgres pg_isready -U postgres

# View PostgreSQL logs
docker logs rag-arc-postgres

# Restart PostgreSQL
docker restart rag-arc-postgres
```

**Problem**: Neo4j not ready
```bash
# Check Neo4j status
docker exec rag-arc-neo4j cypher-shell -u neo4j -p 12345678 "RETURN 1"

# View Neo4j logs
docker logs rag-arc-neo4j

# Restart Neo4j (may take 1-2 minutes to start)
docker restart rag-arc-neo4j
```

**Problem**: Application not starting
```bash
# Check application logs
docker logs rag-arc-app

# Check if all dependencies are ready
docker ps --filter "name=rag-arc-"

# Restart application
docker restart rag-arc-app
```

### Runtime Issues

**Problem**: Out of memory errors
```bash
# Solution: Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
# Or reduce batch size in configuration
```

**Problem**: Slow inference
```bash
# Solution 1: Use GPU mode if available
./build.sh  # Select GPU mode

# Solution 2: Reduce model size in config
# Edit config/json_configs/rag_inference.json
```

### Data Issues

**Problem**: Need to reset all data
```bash
# Use cleanup script
./cleanup.sh

# Then restart services
./start.sh
```

Or use the complete cleanup script (⚠️ This will delete all data including local directories!):
```bash
./clean-docker-data.sh
# Then restart services
./start.sh
```

## 🔄 Updating RAG-ARC

```bash
# 1. Pull latest code
git pull origin main

# 2. Rebuild application image
./build.sh

# 3. Restart services
./start.sh
```

## 📚 More Information

- [Main Documentation](README.md) - Complete project documentation
- [中文文档](README-CN.md) - Chinese documentation
- [API Documentation](http://localhost:8000/docs) - Available after deployment

## 🆘 Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section above
2. View container logs: `docker logs rag-arc-app`
3. Open an issue on [GitHub](https://github.com/DataArcTech/RAG-ARC/issues)

---

**Happy deploying!** 🎉
