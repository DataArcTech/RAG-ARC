#!/bin/bash

# RAG-ARC Start Script - Start All Services

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    print_message "$BLUE" "=========================================="
    print_message "$BLUE" "  RAG-ARC Start Script"
    print_message "$BLUE" "=========================================="
    echo ""
}

# Check if Docker is installed
check_docker() {
    print_message "$BLUE" "📦 Checking Docker environment..."
    
    if ! command -v docker &> /dev/null; then
        print_message "$RED" "❌ Docker is not installed"
        print_message "$YELLOW" "   Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    print_message "$GREEN" "✅ Docker installed: $(docker --version)"
    echo ""
}

# Check if images exist
check_images() {
    print_message "$BLUE" "🔍 Checking Docker images..."
    
    MISSING_IMAGES=()
    
    if ! docker images | grep -q "postgres.*16-alpine"; then
        MISSING_IMAGES+=("postgres:16-alpine")
    fi
    
    if ! docker images | grep -q "redis.*7-alpine"; then
        MISSING_IMAGES+=("redis:7-alpine")
    fi
    
    if ! docker images | grep -q "neo4j.*latest"; then
        MISSING_IMAGES+=("neo4j:latest")
    fi
    
    if ! docker images | grep -q "rag_arc"; then
        MISSING_IMAGES+=("rag_arc:v1 or rag_arc:v1-gpu")
    fi
    
    if [ ${#MISSING_IMAGES[@]} -gt 0 ]; then
        print_message "$RED" "❌ Missing Docker images:"
        for img in "${MISSING_IMAGES[@]}"; do
            print_message "$RED" "   - $img"
        done
        echo ""
        print_message "$YELLOW" "⚠️  Please run ./build.sh first to build images"
        exit 1
    fi
    
    print_message "$GREEN" "✅ All required images found"
    echo ""
}

# Detect which app image to use
detect_app_image() {
    if docker images | grep -q "rag_arc.*v1-gpu"; then
        APP_IMAGE="rag_arc:v1-gpu"
        MODE="gpu"
        print_message "$GREEN" "✅ Detected GPU image: $APP_IMAGE"
    elif docker images | grep -q "rag_arc.*v1"; then
        APP_IMAGE="rag_arc:v1"
        MODE="cpu"
        print_message "$GREEN" "✅ Detected CPU image: $APP_IMAGE"
    else
        print_message "$RED" "❌ No rag_arc image found"
        print_message "$YELLOW" "   Please run ./build.sh first"
        exit 1
    fi
    echo ""
}

# Stop old containers
stop_old_containers() {
    print_message "$BLUE" "🔍 Checking for old containers..."
    
    # Stop and remove old app container
    OLD_APP=$(docker ps -a -q -f name=rag-arc-app)
    if [ ! -z "$OLD_APP" ]; then
        print_message "$YELLOW" "⚠️  Found old app container, stopping and removing..."
        docker stop rag-arc-app 2>/dev/null || true
        docker rm rag-arc-app 2>/dev/null || true
        print_message "$GREEN" "✅ Old app container cleaned"
    fi
    
    # Stop and remove old postgres container
    OLD_POSTGRES=$(docker ps -a -q -f name=rag-arc-postgres)
    if [ ! -z "$OLD_POSTGRES" ]; then
        print_message "$YELLOW" "⚠️  Found old postgres container, stopping and removing..."
        docker stop rag-arc-postgres 2>/dev/null || true
        docker rm rag-arc-postgres 2>/dev/null || true
        print_message "$GREEN" "✅ Old postgres container cleaned"
    fi
    
    # Stop and remove old redis container
    OLD_REDIS=$(docker ps -a -q -f name=rag-arc-redis)
    if [ ! -z "$OLD_REDIS" ]; then
        print_message "$YELLOW" "⚠️  Found old redis container, stopping and removing..."
        docker stop rag-arc-redis 2>/dev/null || true
        docker rm rag-arc-redis 2>/dev/null || true
        print_message "$GREEN" "✅ Old redis container cleaned"
    fi
    
    # Stop and remove old neo4j container
    OLD_NEO4J=$(docker ps -a -q -f name=rag-arc-neo4j)
    if [ ! -z "$OLD_NEO4J" ]; then
        print_message "$YELLOW" "⚠️  Found old neo4j container, stopping and removing..."
        docker stop rag-arc-neo4j 2>/dev/null || true
        docker rm rag-arc-neo4j 2>/dev/null || true
        print_message "$GREEN" "✅ Old neo4j container cleaned"
    fi
    
    if [ -z "$OLD_APP" ] && [ -z "$OLD_POSTGRES" ] && [ -z "$OLD_REDIS" ] && [ -z "$OLD_NEO4J" ]; then
        print_message "$GREEN" "✅ No old containers found"
    fi
    echo ""
}

# Create Docker network
create_network() {
    print_message "$BLUE" "🌐 Creating Docker network..."
    
    if docker network inspect rag-arc-network &> /dev/null; then
        print_message "$GREEN" "✅ Network already exists"
    else
        docker network create rag-arc-network
        print_message "$GREEN" "✅ Network created"
    fi
    echo ""
}

# Start PostgreSQL container
start_postgres() {
    print_message "$BLUE" "🗄️  Starting PostgreSQL..."
    
    docker run -d \
        --name rag-arc-postgres \
        --network rag-arc-network \
        -e POSTGRES_USER=${POSTGRES_USER:-postgres} \
        -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123} \
        -e POSTGRES_DB=${POSTGRES_DB:-rag_arc} \
        -v rag-arc-postgres-data:/var/lib/postgresql/data \
        postgres:16-alpine
    
    print_message "$GREEN" "✅ PostgreSQL started"
    echo ""
}

# Start Redis container
start_redis() {
    print_message "$BLUE" "📦 Starting Redis..."
    
    docker run -d \
        --name rag-arc-redis \
        --network rag-arc-network \
        -v rag-arc-redis-data:/data \
        redis:7-alpine redis-server --appendonly yes
    
    print_message "$GREEN" "✅ Redis started"
    echo ""
}

# Start Neo4j container
start_neo4j() {
    print_message "$BLUE" "🔷 Starting Neo4j..."

    # Check if Neo4j ports should be exposed (from .env file)
    # Default: not exposed (more secure)
    EXPOSE_NEO4J=${EXPOSE_NEO4J:-false}
    NEO4J_HTTP_PORT=${NEO4J_HTTP_PORT:-7474}
    NEO4J_BOLT_PORT=${NEO4J_BOLT_PORT:-7687}

    NEO4J_PORTS=""
    if [[ "$EXPOSE_NEO4J" == "true" ]]; then
        NEO4J_PORTS="-p ${NEO4J_HTTP_PORT}:7474 -p ${NEO4J_BOLT_PORT}:7687"
        print_message "$GREEN" "✅ Neo4j ports will be exposed:"
        print_message "$GREEN" "   - Browser: http://localhost:${NEO4J_HTTP_PORT}"
        print_message "$GREEN" "   - Bolt: bolt://localhost:${NEO4J_BOLT_PORT}"
    else
        print_message "$YELLOW" "ℹ️  Neo4j ports not exposed (set EXPOSE_NEO4J=true in .env to expose)"
    fi

    docker run -d \
        --name rag-arc-neo4j \
        --network rag-arc-network \
        $NEO4J_PORTS \
        -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-12345678} \
        -e NEO4J_PLUGINS='["apoc"]' \
        -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
        -v rag-arc-neo4j-data:/data \
        -v rag-arc-neo4j-logs:/logs \
        neo4j:latest

    print_message "$GREEN" "✅ Neo4j started"
    echo ""
}

# Wait for database to be ready
wait_for_database() {
    print_message "$BLUE" "⏳ Waiting for PostgreSQL to be ready..."
    
    MAX_ATTEMPTS=30
    ATTEMPT=0
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if docker exec rag-arc-postgres pg_isready -U ${POSTGRES_USER:-postgres} > /dev/null 2>&1; then
            print_message "$GREEN" "✅ PostgreSQL is ready"
            echo ""
            return 0
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
        echo -n "."
        sleep 1
    done
    
    echo ""
    print_message "$YELLOW" "⚠️  PostgreSQL startup timeout"
    echo ""
}

# Wait for Neo4j to be ready
wait_for_neo4j() {
    print_message "$BLUE" "⏳ Waiting for Neo4j to be ready..."
    
    MAX_ATTEMPTS=60
    ATTEMPT=0
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if docker exec rag-arc-neo4j cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-12345678} "RETURN 1" > /dev/null 2>&1; then
            print_message "$GREEN" "✅ Neo4j is ready"
            echo ""
            return 0
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
        echo -n "."
        sleep 2
    done
    
    echo ""
    print_message "$YELLOW" "⚠️  Neo4j startup timeout"
    echo ""
}

# Start application container
start_app() {
    print_message "$BLUE" "🚀 Starting application..."
    echo ""
    
    # Select port
    PORT=8000
    read -p "Port to use (default 8000): " -r
    if [ ! -z "$REPLY" ]; then
        PORT=$REPLY
    fi
    
    # Build run command
    RUN_CMD="docker run -d \
        --name rag-arc-app \
        --network rag-arc-network \
        -p ${PORT}:8000 \
        -e POSTGRES_HOST=rag-arc-postgres \
        -e POSTGRES_PORT=5432 \
        -e POSTGRES_USER=${POSTGRES_USER:-postgres} \
        -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123} \
        -e POSTGRES_DB=${POSTGRES_DB:-rag_arc} \
        -e REDIS_HOST=rag-arc-redis \
        -e REDIS_PORT=6379 \
        -e NEO4J_URL=bolt://rag-arc-neo4j:7687 \
        -e NEO4J_USERNAME=neo4j \
        -e NEO4J_PASSWORD=${NEO4J_PASSWORD:-12345678} \
        -e NEO4J_DATABASE=neo4j \
        --env-file .env \
        -v $(pwd)/data:/rag_arc/data \
        -v $(pwd)/local:/rag_arc/local \
        -v $(pwd)/models:/rag_arc/models"
    
    # Add GPU support for GPU mode
    if [ "$MODE" == "gpu" ]; then
        RUN_CMD="$RUN_CMD --gpus all"
    fi
    
    RUN_CMD="$RUN_CMD $APP_IMAGE"
    
    # Run container
    eval $RUN_CMD
    
    CONTAINER_ID=$(docker ps -q -f name=rag-arc-app)
    
    print_message "$GREEN" "✅ Starting application"
    print_message "$NC" "   Container ID: $CONTAINER_ID"
    print_message "$NC" "   Access URL: http://localhost:${PORT}"
    echo ""
}

# Wait for service to start
wait_for_service() {
    print_message "$BLUE" "⏳ Waiting for service to start..."
    print_message "$YELLOW" "   Note: First startup may take 10-20 minutes to download/load models"
    print_message "$YELLOW" "   Checking every 5 seconds (max 20 minutes)..."
    echo ""
    
    MAX_ATTEMPTS=240
    ATTEMPT=0
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if curl -s http://localhost:${PORT}/ > /dev/null 2>&1; then
            print_message "$GREEN" "✅ Service is ready"
            echo ""
            return 0
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
        
        # Show progress every 12 attempts (1 minute)
        if [ $((ATTEMPT % 12)) -eq 0 ]; then
            ELAPSED=$((ATTEMPT * 5 / 60))
            echo ""
            print_message "$YELLOW" "   Still waiting... (${ELAPSED} minutes elapsed)"
            print_message "$NC" "   You can check progress with: docker logs -f rag-arc-app"
            echo ""
        else
            echo -n "."
        fi
        
        sleep 5  # Increased from 2 to 5 seconds
    done
    
    echo ""
    print_message "$YELLOW" "⚠️  Service startup timeout after 10 minutes"
    print_message "$NC" "   This is often due to model downloading/loading on first startup"
    print_message "$NC" "   Please check logs to see if model is still loading:"
    print_message "$NC" "   Run: docker logs rag-arc-app"
    print_message "$NC" ""
    print_message "$NC" "   If model is still loading, you can:"
    print_message "$NC" "   1. Wait longer and check logs periodically"
    print_message "$NC" "   2. Check if model files exist in ./models directory"
    print_message "$NC" "   3. Verify network connection for model download"
    echo ""
    exit 1;
}

# Show deployment info
show_info() {
    print_message "$BLUE" "=========================================="
    print_message "$GREEN" "🎉 All Services Started Successfully!"
    print_message "$BLUE" "=========================================="
    echo ""
    print_message "$NC" "📍 Access URLs:"
    print_message "$GREEN" "   - API Service: http://localhost:${PORT}"
    print_message "$GREEN" "   - API Docs: http://localhost:${PORT}/docs"
    print_message "$GREEN" "   - Health Check: http://localhost:${PORT}/"
    echo ""
    print_message "$NC" "📊 Running Containers:"
    docker ps --filter "name=rag-arc-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    print_message "$NC" "📝 Common Commands:"
    print_message "$NC" "   - View app logs: docker logs -f rag-arc-app"
    print_message "$NC" "   - View postgres logs: docker logs -f rag-arc-postgres"
    print_message "$NC" "   - View redis logs: docker logs -f rag-arc-redis"
    print_message "$NC" "   - View neo4j logs: docker logs -f rag-arc-neo4j"
    print_message "$NC" "   - Stop all: ./stop.sh"
    print_message "$NC" "   - Start all: ./start.sh"
    print_message "$NC" "   - Remove all: ./cleanup.sh"
    print_message "$NC" "   - Clean Docker data: ./clean-docker-data.sh"
    echo ""
    print_message "$NC" "🔧 Restart services:"
    print_message "$NC" "   ./start.sh"
    echo ""
    print_message "$BLUE" "=========================================="
}

# Main function
main() {
    print_header
    check_docker
    check_images
    detect_app_image
    stop_old_containers
    create_network
    start_postgres
    start_redis
    start_neo4j
    wait_for_database
    wait_for_neo4j
    start_app
    wait_for_service
    show_info
}

main

