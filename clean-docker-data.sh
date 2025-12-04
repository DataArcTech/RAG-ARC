#!/bin/bash

# RAG-ARC Clean Docker Data Script
# This script cleans all Docker data for Neo4j, PostgreSQL, Redis, and other services

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
    print_message "$BLUE" "  RAG-ARC Clean Docker Data Script"
    print_message "$BLUE" "=========================================="
    echo ""
}

# Confirm before cleaning
confirm_clean() {
    print_message "$RED" "⚠️  WARNING: This will delete the following RAG-ARC Docker resources:"
    print_message "$RED" "   - RAG-ARC containers (rag-arc-*)"
    print_message "$RED" "   - RAG-ARC volumes (rag-arc-*)"
    print_message "$RED" "   - RAG-ARC application images (rag_arc:v1, rag_arc:v1-gpu)"
    print_message "$RED" "   - PostgreSQL, Redis, Neo4j databases"
    print_message "$RED" "   - Local data directories (./data/)"
    echo ""
    print_message "$YELLOW" "ℹ️  Base images (postgres, redis, neo4j) will be preserved"
    print_message "$YELLOW" "This action cannot be undone!"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " response

    if [[ "$response" != "yes" ]]; then
        print_message "$YELLOW" "❌ Operation cancelled"
        exit 0
    fi
}

# Stop all containers
stop_containers() {
    print_message "$BLUE" "🛑 Stopping all containers..."
    
    docker stop rag-arc-app 2>/dev/null || true
    docker stop rag-arc-postgres 2>/dev/null || true
    docker stop rag-arc-redis 2>/dev/null || true
    docker stop rag-arc-neo4j 2>/dev/null || true
    
    print_message "$GREEN" "✅ All containers stopped"
    echo ""
}

# Remove containers
remove_containers() {
    print_message "$BLUE" "🗑️  Removing containers..."
    
    docker rm rag-arc-app 2>/dev/null || true
    docker rm rag-arc-postgres 2>/dev/null || true
    docker rm rag-arc-redis 2>/dev/null || true
    docker rm rag-arc-neo4j 2>/dev/null || true
    
    print_message "$GREEN" "✅ All containers removed"
    echo ""
}

# Remove Docker volumes
remove_volumes() {
    print_message "$BLUE" "💾 Removing Docker volumes..."
    
    docker volume rm rag-arc-postgres-data 2>/dev/null || true
    docker volume rm rag-arc-redis-data 2>/dev/null || true
    docker volume rm rag-arc-neo4j-data 2>/dev/null || true
    
    print_message "$GREEN" "✅ All volumes removed"
    echo ""
}

# Clean local data directories
clean_local_data() {
    print_message "$BLUE" "🧹 Cleaning local data directories..."
    
    # Clean PostgreSQL data
    if [ -d "./data/postgresql" ]; then
        rm -rf ./data/postgresql
        print_message "$GREEN" "✅ Cleaned ./data/postgresql"
    fi
    
    # Clean Neo4j data
    if [ -d "./data/neo4j" ]; then
        rm -rf ./data/neo4j
        print_message "$GREEN" "✅ Cleaned ./data/neo4j"
    fi
    
    # Clean Redis data
    if [ -d "./data/redis" ]; then
        rm -rf ./data/redis
        print_message "$GREEN" "✅ Cleaned ./data/redis"
    fi
    
    # Clean graph index
    if [ -d "./data/graph_index_neo4j" ]; then
        rm -rf ./data/graph_index_neo4j
        print_message "$GREEN" "✅ Cleaned ./data/graph_index_neo4j"
    fi
    
    echo ""
}

# Clean Docker build cache
clean_build_cache() {
    print_message "$BLUE" "🔨 Cleaning RAG-ARC Docker build cache..."

    # Only clean RAG-ARC related build cache
    docker builder prune --filter "label=project=rag-arc" -f 2>/dev/null || true

    print_message "$GREEN" "✅ Docker build cache cleaned"
    echo ""
}

# Remove RAG-ARC application images
remove_images() {
    print_message "$BLUE" "🖼️  Removing RAG-ARC application images..."

    # Remove both CPU and GPU versions of the RAG-ARC app image
    docker rmi rag_arc:v1 2>/dev/null || true
    docker rmi rag_arc:v1-gpu 2>/dev/null || true

    print_message "$GREEN" "✅ RAG-ARC application images removed"
    echo ""
}

# Show summary
show_summary() {
    print_message "$GREEN" "✅ RAG-ARC Docker data cleanup completed!"
    echo ""
    print_message "$BLUE" "Summary of cleaned items:"
    print_message "$BLUE" "  - Containers: rag-arc-app, rag-arc-postgres, rag-arc-redis, rag-arc-neo4j"
    print_message "$BLUE" "  - Volumes: rag-arc-postgres-data, rag-arc-redis-data, rag-arc-neo4j-data"
    print_message "$BLUE" "  - Images: rag_arc:v1, rag_arc:v1-gpu"
    print_message "$BLUE" "  - Local directories: ./data/postgresql, ./data/neo4j, ./data/redis, ./data/graph_index_neo4j"
    print_message "$BLUE" "  - RAG-ARC build cache"
    echo ""
    print_message "$YELLOW" "⚠️  Note: Base images (PostgreSQL, Redis, Neo4j) are preserved as they may be used by other projects"
    echo ""
    print_message "$YELLOW" "Next steps:"
    print_message "$YELLOW" "  1. Run './start.sh' to start fresh containers"
    print_message "$YELLOW" "  2. Or run 'uv run uvicorn main:app' for local development"
    echo ""
}

# Main function
main() {
    print_header
    confirm_clean
    stop_containers
    remove_containers
    remove_volumes
    clean_local_data
    remove_images
    clean_build_cache
    show_summary
}

main

