#!/bin/bash

# RAG-ARC Stop Script - Stop All Services

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
    print_message "$BLUE" "  RAG-ARC Stop Script"
    print_message "$BLUE" "=========================================="
    echo ""
}

# Stop all containers
stop_containers() {
    print_message "$BLUE" "🛑 Stopping all RAG-ARC containers..."
    echo ""
    
    STOPPED=0
    
    # Stop app container
    if docker ps -q -f name=rag-arc-app | grep -q .; then
        print_message "$YELLOW" "   Stopping rag-arc-app..."
        docker stop rag-arc-app
        print_message "$GREEN" "   ✅ rag-arc-app stopped"
        STOPPED=$((STOPPED + 1))
    fi
    
    # Stop postgres container
    if docker ps -q -f name=rag-arc-postgres | grep -q .; then
        print_message "$YELLOW" "   Stopping rag-arc-postgres..."
        docker stop rag-arc-postgres
        print_message "$GREEN" "   ✅ rag-arc-postgres stopped"
        STOPPED=$((STOPPED + 1))
    fi
    
    # Stop redis container
    if docker ps -q -f name=rag-arc-redis | grep -q .; then
        print_message "$YELLOW" "   Stopping rag-arc-redis..."
        docker stop rag-arc-redis
        print_message "$GREEN" "   ✅ rag-arc-redis stopped"
        STOPPED=$((STOPPED + 1))
    fi
    
    # Stop neo4j container
    if docker ps -q -f name=rag-arc-neo4j | grep -q .; then
        print_message "$YELLOW" "   Stopping rag-arc-neo4j..."
        docker stop rag-arc-neo4j
        print_message "$GREEN" "   ✅ rag-arc-neo4j stopped"
        STOPPED=$((STOPPED + 1))
    fi
    
    echo ""
    if [ $STOPPED -eq 0 ]; then
        print_message "$YELLOW" "⚠️  No running containers found"
    else
        print_message "$GREEN" "✅ Stopped $STOPPED container(s)"
    fi
    echo ""
}

# Show container status
show_status() {
    print_message "$BLUE" "📊 Container Status:"
    echo ""
    
    # Check if any containers exist
    if docker ps -a --filter "name=rag-arc-" --format "{{.Names}}" | grep -q .; then
        docker ps -a --filter "name=rag-arc-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        print_message "$YELLOW" "   No RAG-ARC containers found"
    fi
    echo ""
}

# Show summary
show_summary() {
    print_message "$BLUE" "=========================================="
    print_message "$GREEN" "🎉 All Services Stopped!"
    print_message "$BLUE" "=========================================="
    echo ""
    print_message "$NC" "📝 Next Steps:"
    print_message "$NC" "   - Start services: ./start.sh"
    print_message "$NC" "   - Remove containers: ./cleanup.sh"
    print_message "$NC" "   - View logs: docker logs rag-arc-app"
    echo ""
    print_message "$BLUE" "=========================================="
}

# Main function
main() {
    print_header
    stop_containers
    show_status
    show_summary
}

main

