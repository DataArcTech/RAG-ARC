# RAG-ARC Cleanup Script - Remove All Containers and Volumes

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
    print_message "$BLUE" "  RAG-ARC Cleanup Script"
    print_message "$BLUE" "=========================================="
    echo ""
}

# Confirm cleanup
confirm_cleanup() {
    print_message "$YELLOW" "⚠️  WARNING: This will remove all RAG-ARC containers and data!"
    print_message "$YELLOW" "   - All containers will be removed"
    print_message "$YELLOW" "   - All data volumes will be removed (PostgreSQL, Redis, Neo4j data)"
    print_message "$YELLOW" "   - This action CANNOT be undone!"
    echo ""
    
    read -p "Are you sure you want to continue? (yes/NO): " -r
    echo
    
    if [[ ! $REPLY == "yes" ]]; then
        print_message "$GREEN" "✅ Cleanup cancelled"
        exit 0
    fi
    echo ""
}

# Stop and remove containers
remove_containers() {
    print_message "$BLUE" "🗑️  Removing containers..."
    echo ""
    
    REMOVED=0
    
    # Remove app container
    if docker ps -a -q -f name=rag-arc-app | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-app..."
        docker stop rag-arc-app 2>/dev/null || true
        docker rm rag-arc-app
        print_message "$GREEN" "   ✅ rag-arc-app removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    # Remove postgres container
    if docker ps -a -q -f name=rag-arc-postgres | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-postgres..."
        docker stop rag-arc-postgres 2>/dev/null || true
        docker rm rag-arc-postgres
        print_message "$GREEN" "   ✅ rag-arc-postgres removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    # Remove redis container
    if docker ps -a -q -f name=rag-arc-redis | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-redis..."
        docker stop rag-arc-redis 2>/dev/null || true
        docker rm rag-arc-redis
        print_message "$GREEN" "   ✅ rag-arc-redis removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    # Remove neo4j container
    if docker ps -a -q -f name=rag-arc-neo4j | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-neo4j..."
        docker stop rag-arc-neo4j 2>/dev/null || true
        docker rm rag-arc-neo4j
        print_message "$GREEN" "   ✅ rag-arc-neo4j removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    echo ""
    if [ $REMOVED -eq 0 ]; then
        print_message "$YELLOW" "⚠️  No containers found"
    else
        print_message "$GREEN" "✅ Removed $REMOVED container(s)"
    fi
    echo ""
}

# Remove volumes
remove_volumes() {
    print_message "$BLUE" "🗑️  Removing data volumes..."
    echo ""
    
    REMOVED=0
    
    # Remove postgres volume
    if docker volume ls -q -f name=rag-arc-postgres-data | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-postgres-data..."
        docker volume rm rag-arc-postgres-data
        print_message "$GREEN" "   ✅ rag-arc-postgres-data removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    # Remove redis volume
    if docker volume ls -q -f name=rag-arc-redis-data | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-redis-data..."
        docker volume rm rag-arc-redis-data
        print_message "$GREEN" "   ✅ rag-arc-redis-data removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    # Remove neo4j data volume
    if docker volume ls -q -f name=rag-arc-neo4j-data | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-neo4j-data..."
        docker volume rm rag-arc-neo4j-data
        print_message "$GREEN" "   ✅ rag-arc-neo4j-data removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    # Remove neo4j logs volume
    if docker volume ls -q -f name=rag-arc-neo4j-logs | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-neo4j-logs..."
        docker volume rm rag-arc-neo4j-logs
        print_message "$GREEN" "   ✅ rag-arc-neo4j-logs removed"
        REMOVED=$((REMOVED + 1))
    fi
    
    echo ""
    if [ $REMOVED -eq 0 ]; then
        print_message "$YELLOW" "⚠️  No volumes found"
    else
        print_message "$GREEN" "✅ Removed $REMOVED volume(s)"
    fi
    echo ""
}

# Remove network
remove_network() {
    print_message "$BLUE" "🗑️  Removing network..."
    echo ""
    
    if docker network ls -q -f name=rag-arc-network | grep -q .; then
        print_message "$YELLOW" "   Removing rag-arc-network..."
        docker network rm rag-arc-network
        print_message "$GREEN" "   ✅ rag-arc-network removed"
    else
        print_message "$YELLOW" "⚠️  Network not found"
    fi
    echo ""
}

# Show summary
show_summary() {
    print_message "$BLUE" "=========================================="
    print_message "$GREEN" "🎉 Cleanup Completed!"
    print_message "$BLUE" "=========================================="
    echo ""
    print_message "$NC" "📝 Next Steps:"
    print_message "$NC" "   - Rebuild: ./build.sh"
    print_message "$NC" "   - Start: ./start.sh"
    echo ""
    print_message "$BLUE" "=========================================="
}

# Main function
main() {
    print_header
    confirm_cleanup
    remove_containers
    remove_volumes
    remove_network
    show_summary
}

main

