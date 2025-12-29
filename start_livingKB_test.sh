#!/bin/bash
set -euo pipefail

# ==============================================================================
# 1. 核心配置 (按您的要求固定)
# ==============================================================================
# PM2 应用名称
PM2_APP_NAME="rag-app-livingKB_test"
# 应用端口
APP_PORT="8000"

# ==============================================================================
# 2. 核心优化：自动定位main.py所在目录
# ==============================================================================
find_main_py() {
    local current_dir=$(pwd)
    local max_depth=10
    local depth=0

    if [ -f "${current_dir}/main.py" ]; then
        echo "${current_dir}"
        return 0
    fi

    while [ $depth -lt $max_depth ]; do
        current_dir=$(dirname "${current_dir}")
        if [ -f "${current_dir}/main.py" ]; then
            echo "${current_dir}"
            return 0
        fi
        depth=$((depth + 1))
    done

    echo "❌ 未找到main.py文件（已向上查找${max_depth}层）" >&2
    exit 1
}

APP_DIR=$(find_main_py)
echo "🔍 自动定位到应用目录: ${APP_DIR}"

# ==============================================================================
# 3. 加载.env文件
# ==============================================================================
ENV_FILE=${ENV_FILE:-${APP_DIR}/.env}
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    export $(grep -v '^#' ${ENV_FILE} | grep -v '^$' | xargs)
else
    echo "⚠️  未找到.env文件（${ENV_FILE}），将使用脚本默认值或系统环境变量"
fi

# ==============================================================================
# 4. 配置区 (Docker和数据库配置，可通过.env覆盖)
# ==============================================================================
NETWORK_NAME=${NETWORK_NAME:-rag-arc-network}
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-rag-arc-postgres}
REDIS_CONTAINER_NAME=${REDIS_CONTAINER_NAME:-rag-arc-redis}
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME:-rag-arc-neo4j}

POSTGRES_HOST_PORT=${POSTGRES_PORT:-5432}
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123}
POSTGRES_DB=${POSTGRES_DB:-rag_arc}
REDIS_HOST_PORT=${REDIS_PORT:-6379}
REDIS_PASSWORD=${REDIS_PASSWORD:-""}
NEO4J_WEB_HOST_PORT=${NEO4J_WEB_PORT:-7474}
NEO4J_BOLT_HOST_PORT=${NEO4J_BOLT_PORT:-7687}
if [[ -n "${NEO4J_URL:-}" ]]; then
    NEO4J_BOLT_HOST_PORT=$(echo ${NEO4J_URL} | awk -F':' '{print $NF}')
fi
NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-12345678}

POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:16-alpine}
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:latest}

APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app.log}
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-${APP_DIR}/data/parsed_files}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-${APP_DIR}/local/files}

# ==============================================================================
# 5. 检查核心依赖命令
# ==============================================================================
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ 错误: 未找到 '$1' 命令。请先安装 $1。" >&2
        exit 1
    fi
}
check_dependency "docker"
check_dependency "pm2"
check_dependency "uv"
check_dependency "lsof"
check_dependency "pkill"

# ==============================================================================
# 6. 启动和管理 Docker 容器
# ==============================================================================
echo -e "\n=========================================="
echo "  RAG-ARC 启动脚本 (强制清理版)"
echo "=========================================="
echo ""

echo "📦 创建/检查Docker网络 [${NETWORK_NAME}]..."
docker network create ${NETWORK_NAME} 2>/dev/null || echo "  ✅ 网络已存在，无需创建"
echo ""

echo "🗄️  处理PostgreSQL容器 [${POSTGRES_CONTAINER_NAME}]..."
if docker ps -a | grep -q "${POSTGRES_CONTAINER_NAME}"; then
    docker stop ${POSTGRES_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧PostgreSQL容器"
    docker start ${POSTGRES_CONTAINER_NAME} && echo "  ✅ PostgreSQL容器重启成功（保留数据）"
else
    docker run -d \
        --name ${POSTGRES_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        -p ${POSTGRES_HOST_PORT}:5432 \
        -e POSTGRES_USER=${POSTGRES_USER} \
        -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
        -e POSTGRES_DB=${POSTGRES_DB} \
        -v rag-arc-postgres-data:/var/lib/postgresql/data \
        ${POSTGRES_IMAGE} && echo "  ✅ PostgreSQL容器已创建并启动（新数据卷）"
fi
echo ""

echo "📦 处理Redis容器 [${REDIS_CONTAINER_NAME}]..."
REDIS_CMD="redis-server --appendonly yes"
if [[ -n "${REDIS_PASSWORD}" ]]; then
    REDIS_CMD="redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}"
fi
if docker ps -a | grep -q "${REDIS_CONTAINER_NAME}"; then
    docker stop ${REDIS_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Redis容器"
    docker start ${REDIS_CONTAINER_NAME} && echo "  ✅ Redis容器重启成功（保留数据）"
else
    docker run -d \
        --name ${REDIS_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        -p ${REDIS_HOST_PORT}:6379 \
        -v rag-arc-redis-data:/data \
        ${REDIS_IMAGE} ${REDIS_CMD} && echo "  ✅ Redis容器已创建并启动（新数据卷）"
fi
echo ""

echo "🔷 处理Neo4j容器 [${NEO4J_CONTAINER_NAME}]..."
if docker ps -a | grep -q "${NEO4J_CONTAINER_NAME}"; then
    docker stop ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Neo4j容器"
    docker start ${NEO4J_CONTAINER_NAME} && echo "  ✅ Neo4j容器重启成功（保留数据）"
else
    docker run -d \
        --name ${NEO4J_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        -p ${NEO4J_WEB_HOST_PORT}:7474 \
        -p ${NEO4J_BOLT_HOST_PORT}:7687 \
        -e NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD} \
        -e NEO4J_PLUGINS='["apoc"]' \
        -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
        -v rag-arc-neo4j-data:/data \
        -v rag-arc-neo4j-logs:/logs \
        ${NEO4J_IMAGE} && echo "  ✅ Neo4j容器已创建并启动（新数据卷）"
fi
echo ""

echo "⏳ 等待中间件就绪..."
sleep 10
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker exec ${POSTGRES_CONTAINER_NAME} pg_isready -U ${POSTGRES_USER} -h localhost > /dev/null 2>&1; then
        echo "  ✅ PostgreSQL已就绪 (端口: ${POSTGRES_HOST_PORT})"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 1
done
echo ""
if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "  ⚠️  PostgreSQL启动超时，但将继续尝试启动应用（请检查容器日志）"
fi

# ==============================================================================
# 7. 使用 PM2 启动/重启应用 (包含增强版清理逻辑)
# ==============================================================================
echo "🚀 启动/重启应用 [${APP_DIR}]..."
mkdir -p ${PARSER_OUTPUT_DIR} ${LOCAL_FILE_STORAGE_PATH} $(dirname ${APP_LOG_FILE})

if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"

cd ${APP_DIR} || { echo "❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

if [ -f "pyproject.toml" ]; then
    echo "  📦 同步项目依赖..."
    uv sync --quiet && echo "  ✅ 依赖同步完成" || echo "  ⚠️  依赖同步失败，尝试继续启动..."
fi

# 7.1 【核心修改】增强版终极清理：确保 8000 端口绝对干净
echo "🧹 正在进行增强版清理，确保端口 ${APP_PORT} 空闲..."

# 步骤 1: 通过 PM2 清理
echo "  -> 步骤 1: 通过 PM2 停止并删除旧的 [${PM2_APP_NAME}] 进程..."
pm2 stop "${PM2_APP_NAME}" > /dev/null 2>&1 || true
pm2 delete "${PM2_APP_NAME}" > /dev/null 2>&1 || true

# 步骤 2: 通过 pkill 清理 (根据命令模式)
echo "  -> 步骤 2: 通过 pkill 强制杀死所有相关的 uvicorn 进程..."
pkill -f "uvicorn main:app --port ${APP_PORT}" > /dev/null 2>&1 || true

# 步骤 3: 通过 lsof 清理 (根据端口号，最彻底)
PID_TO_KILL=$(sudo lsof -t -i:"${APP_PORT}" 2>/dev/null || true)
if [ -n "${PID_TO_KILL}" ]; then
    echo "  -> 步骤 3: 发现进程 ${PID_TO_KILL} 占用端口 ${APP_PORT}，正在强制杀死..."
    sudo kill -9 "${PID_TO_KILL}" > /dev/null 2>&1 || true
    sleep 2 # 等待一下，确保进程已完全退出
fi

echo "  ✅ 清理完成。"

# 7.2 使用 PM2 启动新进程
echo "  🚀 正在通过 PM2 启动新的 [${PM2_APP_NAME}] 进程..."
pm2 start "uv run uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}" \
    --name "${PM2_APP_NAME}" \
    --log "${APP_LOG_FILE}" \
    --env "PYTHONUNBUFFERED=1" \
    --env "POSTGRES_HOST=${POSTGRES_HOST:-localhost}" \
    --env "POSTGRES_PORT=${POSTGRES_HOST_PORT}" \
    --env "POSTGRES_USER=${POSTGRES_USER}" \
    --env "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" \
    --env "POSTGRES_DB=${POSTGRES_DB}" \
    --env "REDIS_HOST=${REDIS_HOST:-localhost}" \
    --env "REDIS_PORT=${REDIS_HOST_PORT}" \
    --env "REDIS_PASSWORD=${REDIS_PASSWORD}" \
    --env "REDIS_DB=${REDIS_DB:-0}" \
    --env "NEO4J_URL=${NEO4J_URL:-bolt://localhost:${NEO4J_BOLT_HOST_PORT}}" \
    --env "NEO4J_USERNAME=${NEO4J_USERNAME}" \
    --env "NEO4J_PASSWORD=${NEO4J_PASSWORD}" \
    --env "NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j}" \
    --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
    --env "OPENAI_BASE_URL=${OPENAI_BASE_URL:-}" \
    --env "OPENAI_CHAT_MODEL=${OPENAI_CHAT_MODEL:-}" \
    --env "OPENAI_EMBEDDING_MODEL=${OPENAI_EMBEDDING_MODEL:-}" \
    --env "EMBEDDING_MODEL_PROVIDER=${EMBEDDING_MODEL_PROVIDER:-}" \
    --env "MODEL_PROFILE=${MODEL_PROFILE:-}" \
    --env "PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR}" \
    --env "LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH}" \
    --env "LOG_LEVEL=${LOG_LEVEL:-INFO}" \
    --env "JWT_SECRET_KEY=${JWT_SECRET_KEY:-}"

echo "  ✅ ${PM2_APP_NAME} 应用已通过 PM2 启动！"
echo ""

# ==============================================================================
# 8. 验证应用状态
# ==============================================================================
echo "⏳ 验证应用启动状态..."
sleep 8

if pm2 list | grep -q "${PM2_APP_NAME}" && pm2 list | grep -q "online"; then
    echo "=========================================="
    echo "  ✅ 所有服务启动/重启成功！（由 PM2 守护）"
    echo "=========================================="
    echo ""
    echo "📋 服务信息："
    echo "  - PM2 应用名: ${PM2_APP_NAME}"
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo ""
    echo "📝 PM2 管理命令："
    echo "  - 查看状态: pm2 list"
    echo "  - 查看日志: pm2 logs ${PM2_APP_NAME}"
    echo "  - 重启应用: pm2 restart ${PM2_APP_NAME}"
    echo "  - 停止应用: pm2 stop ${PM2_APP_NAME}"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  应用启动可能异常（请检查日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    echo "  1. 查看 PM2 状态: pm2 list"
    echo "  2. 查看应用日志: pm2 logs ${PM2_APP_NAME}"
    echo "  3. 检查端口占用: sudo lsof -i:${APP_PORT}"
    echo ""
fi