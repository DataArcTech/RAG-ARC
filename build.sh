#!/bin/bash

# RAG-ARC Build Script - Download and Build Docker Images

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
    print_message "$BLUE" "  RAG-ARC Build Script"
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

# Create .env file
create_env() {
    print_message "$BLUE" "⚙️  Configuring environment variables..."
    
    if [ -f ".env" ]; then
        print_message "$YELLOW" "⚠️  .env file already exists"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_message "$GREEN" "✅ Using existing .env file"
            echo ""
            return
        fi
    fi
    
    if [ ! -f ".env.example" ]; then
        print_message "$RED" "❌ .env.example file not found"
        exit 1
    fi
    
    cp .env.example .env
    print_message "$GREEN" "✅ Created .env file"
    echo ""
    
    # Prompt user to configure API Key
    print_message "$YELLOW" "⚠️  Please configure your LLM API Key"
    print_message "$NC" "   Edit .env file and set:"
    print_message "$NC" "   - OPENAI_API_KEY=sk-your-api-key"
    print_message "$NC" "   - OPENAI_BASE_URL=https://api.openai.com/v1"
    echo ""
    
    read -p "Edit .env file now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        ${EDITOR:-nano} .env
    fi
    echo ""
}

# Select hardware mode
select_mode() {
    print_message "$BLUE" "🎯 Selecting hardware mode..."
    echo ""
    
    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null; then
        print_message "$GREEN" "✅ NVIDIA GPU detected"
        print_message "$NC" "1) CPU mode"
        print_message "$NC" "2) GPU mode (Recommended) [Default]"
        echo ""
        read -p "Select (1/2, default 2): " -n 1 -r
        echo ""
        
        # Default to GPU mode
        if [[ $REPLY == "1" ]]; then
            MODE="cpu"
            print_message "$GREEN" "✅ Selected CPU mode"
        else
            MODE="gpu"
            print_message "$GREEN" "✅ Selected GPU mode (default)"
        fi
    else
        print_message "$YELLOW" "⚠️  NVIDIA GPU or driver not detected"
        print_message "$NC" "1) CPU mode [Default]"
        print_message "$NC" "2) GPU mode (requires NVIDIA GPU)"
        echo ""
        read -p "Select (1/2, default 1): " -n 1 -r
        echo ""
        
        # Default to CPU mode
        if [[ $REPLY == "2" ]]; then
            print_message "$RED" "❌ Cannot use GPU mode, falling back to CPU mode"
            MODE="cpu"
        else
            MODE="cpu"
            print_message "$GREEN" "✅ Selected CPU mode (default)"
        fi
    fi
    echo ""
}

# Pull base images
pull_base_images() {
    print_message "$BLUE" "📥 Pulling base Docker images..."
    echo ""
    
    print_message "$NC" "   Pulling PostgreSQL 16..."
    docker pull postgres:16-alpine
    print_message "$GREEN" "   ✅ PostgreSQL image ready"
    echo ""
    
    print_message "$NC" "   Pulling Redis 7..."
    docker pull redis:7-alpine
    print_message "$GREEN" "   ✅ Redis image ready"
    echo ""
    
    print_message "$NC" "   Pulling Neo4j latest..."
    docker pull neo4j:latest
    print_message "$GREEN" "   ✅ Neo4j image ready"
    echo ""
    
    print_message "$GREEN" "✅ All base images pulled successfully"
    echo ""
}

# Build application image
build_app_image() {
    print_message "$BLUE" "🔨 Building application Docker image..."
    echo ""
    
    if [ "$MODE" == "gpu" ]; then
        DOCKERFILE="Dockerfile.gpu"
        IMAGE_TAG="rag_arc:v1-gpu"
    else
        DOCKERFILE="Dockerfile"
        IMAGE_TAG="rag_arc:v1"
    fi
    
    print_message "$NC" "   Dockerfile: $DOCKERFILE"
    print_message "$NC" "   Image Tag: $IMAGE_TAG"
    echo ""
    
    docker build -f $DOCKERFILE -t $IMAGE_TAG .
    
    print_message "$GREEN" "✅ Application image built successfully"
    echo ""
}

# Show build summary
show_summary() {
    print_message "$BLUE" "=========================================="
    print_message "$GREEN" "🎉 Build Completed Successfully!"
    print_message "$BLUE" "=========================================="
    echo ""
    print_message "$NC" "📦 Built Images:"
    docker images | grep -E "rag_arc|postgres|redis|neo4j" | head -10
    echo ""
    print_message "$NC" "📝 Next Steps:"
    print_message "$GREEN" "   Run: ./start.sh"
    print_message "$NC" "   to start all services"
    echo ""
    print_message "$BLUE" "=========================================="
}

# Main function
main() {
    print_header
    check_docker
    create_env
    select_mode
    pull_base_images
    build_app_image
    show_summary
}

main

