#!/bin/bash
set -euo pipefail

# ==============================================================================
# 1. 核心优化：自动定位main.py所在目录（无需写死路径）
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
# 2. 加载.env文件
# ==============================================================================
ENV_FILE=${ENV_FILE:-${APP_DIR}/.env}
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    export $(grep -v '^#' ${ENV_FILE} | grep -v '^$' | xargs)
else
    echo "⚠️  未找到.env文件（${ENV_FILE}），将使用脚本默认值或系统环境变量"
fi

# ==============================================================================
# 3. 配置区 (所有配置项均可通过.env文件覆盖)
# ==============================================================================
# Docker 网络和容器配置
NETWORK_NAME=${NETWORK_NAME:-rag-arc-network}
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-rag-arc-postgres}
REDIS_CONTAINER_NAME=${REDIS_CONTAINER_NAME:-rag-arc-redis}
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME:-rag-arc-neo4j}

# 数据库端口和凭据
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

# Docker 镜像
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:16-alpine}
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:latest}

# 应用配置
APP_PORT=${APP_PORT:-8000}
APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app.log}
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-${APP_DIR}/data/parsed_files}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-${APP_DIR}/local/files}

# PM2 应用名称 (建议使用独特名称以避免冲突)
PM2_APP_NAME="rag-app-livingKB"

# ==============================================================================
# 4. 检查核心依赖命令
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
check_dependency "lsof" # 用于端口清理
check_dependency "pkill" # 用于进程清理

# ==============================================================================
# 5. 核心功能：启动和管理 Docker 容器
# ==============================================================================
echo -e "\n=========================================="
echo "  RAG-ARC 启动脚本 (自动适配路径+PM2守护)"
echo "=========================================="
echo ""

# 5.1 创建Docker网络
echo "📦 创建/检查Docker网络 [${NETWORK_NAME}]..."
docker network create ${NETWORK_NAME} 2>/dev/null || echo "  ✅ 网络已存在，无需创建"
echo ""

# 5.2 处理PostgreSQL
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

# 5.3 处理Redis
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

# 5.4 处理Neo4j
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

# 5.5 等待中间件就绪
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
# 6. 使用 PM2 启动/重启应用
# ==============================================================================
echo "🚀 启动/重启应用 [${APP_DIR}]..."
mkdir -p ${PARSER_OUTPUT_DIR} ${LOCAL_FILE_STORAGE_PATH} $(dirname ${APP_LOG_FILE})

# 确保 uv 命令在 PATH 中
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"

cd ${APP_DIR} || { echo "❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

if [ -f "pyproject.toml" ]; then
    echo "  📦 同步项目依赖..."
    if uv sync --quiet; then
        echo "  ✅ 依赖同步完成"
    else
        echo "  ⚠️  依赖同步失败，尝试继续启动..."
    fi
fi

# 6.1 终极健壮版：清理残留进程和端口
echo "🧹 正在清理可能残留的旧应用进程和端口..."
# 优先通过 PM2 清理
pm2 stop "${PM2_APP_NAME}" > /dev/null 2>&1 || true
pm2 delete "${PM2_APP_NAME}" > /dev/null 2>&1 || true
# 然后通过命令行模式清理
pkill -f "uvicorn main:app --port ${APP_PORT}" > /dev/null 2>&1 || true
# 最后通过端口强制清理顽固进程
PID_TO_KILL=$(lsof -t -i:"${APP_PORT}" || true)
if [ -n "${PID_TO_KILL}" ]; then
    echo "  ⚠️  发现进程 ${PID_TO_KILL} 占用端口 ${APP_PORT}，正在强制杀死..."
    kill -9 "${PID_TO_KILL}" > /dev/null 2>&1 || true
    sleep 2
fi
echo "  ✅ 清理完成。"

# 6.2 使用 PM2 启动新进程
echo "  🚀 正在通过 PM2 启动新的 ${PM2_APP_NAME} 进程..."
POSTGRES_HOST=${POSTGRES_HOST:-localhost} \
POSTGRES_PORT=${POSTGRES_HOST_PORT} \
POSTGRES_USER=${POSTGRES_USER} \
POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
POSTGRES_DB=${POSTGRES_DB} \
REDIS_HOST=${REDIS_HOST:-localhost} \
REDIS_PORT=${REDIS_HOST_PORT} \
REDIS_PASSWORD=${REDIS_PASSWORD} \
REDIS_DB=${REDIS_DB:-0} \
NEO4J_URL=${NEO4J_URL:-bolt://localhost:${NEO4J_BOLT_HOST_PORT}} \
NEO4J_USERNAME=${NEO4J_USERNAME} \
NEO4J_PASSWORD=${NEO4J_PASSWORD} \
NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j} \
OPENAI_API_KEY=${OPENAI_API_KEY:-} \
OPENAI_BASE_URL=${OPENAI_BASE_URL:-} \
OPENAI_CHAT_MODEL=${OPENAI_CHAT_MODEL:-} \
OPENAI_EMBEDDING_MODEL=${OPENAI_EMBEDDING_MODEL:-} \
EMBEDDING_MODEL_PROVIDER=${EMBEDDING_MODEL_PROVIDER:-} \
MODEL_PROFILE=${MODEL_PROFILE:-} \
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR} \
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH} \
LOG_LEVEL=${LOG_LEVEL:-INFO} \
JWT_SECRET_KEY=${JWT_SECRET_KEY:-} \
pm2 start "uv run uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}" \
    --name "${PM2_APP_NAME}" \
    --log "${APP_LOG_FILE}"

echo "  ✅ ${PM2_APP_NAME} 应用已通过 PM2 启动！"
echo ""

# ==============================================================================
# 7. 验证应用状态
# ==============================================================================
echo "⏳ 验证应用启动状态..."
sleep 8

# 验证 PM2 应用状态
if pm2 list | grep -q "${PM2_APP_NAME}" && pm2 list | grep -q "online"; then
    echo "=========================================="
    echo "  ✅ 所有服务启动/重启成功！（由 PM2 守护）"
    echo "=========================================="
    echo ""
    echo "📋 服务信息（自动适配路径）："
    echo "  - PostgreSQL: ${POSTGRES_HOST}:${POSTGRES_HOST_PORT} (容器: ${POSTGRES_CONTAINER_NAME})"
    echo "  - Redis: ${REDIS_HOST}:${REDIS_HOST_PORT} (容器: ${REDIS_CONTAINER_NAME})"
    echo "  - Neo4j Web: http://localhost:${NEO4J_WEB_HOST_PORT}"
    echo "  - Neo4j Bolt: ${NEO4J_URL} (容器: ${NEO4J_CONTAINER_NAME})"
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo "  - 自动定位的代码目录: ${APP_DIR}"
    echo ""
    echo "📝 PM2 管理命令："
    echo "  - 查看状态: pm2 list"
    echo "  - 查看日志: pm2 logs ${PM2_APP_NAME}"
    echo "  - 重启应用: pm2 restart ${PM2_APP_NAME}"
    echo "  - 停止应用: pm2 stop ${PM2_APP_NAME}"
    echo "  - 开机自启: pm2 startup && pm2 save"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  应用启动可能异常（请检查 PM2 日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    echo "  1. 查看 PM2 状态: pm2 list"
    echo "  2. 查看应用日志: pm2 logs ${PM2_APP_NAME}"
    echo "  3. 检查容器状态: docker ps | grep rag-arc-"
    echo ""
fi