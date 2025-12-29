#!/bin/bash
set -euo pipefail

# ==============================================================================
# 线上环境专用配置 (遵循新的命名规范)
# ==============================================================================
# PM2 应用名称
PM2_APP_NAME="rag-app-chatKB_online"
# 应用标识 (用于Docker资源命名)
APP_ID="chatKB"
# 环境标识 (用于Docker资源命名)
ENV_ID="online"
# Docker 资源命名前缀
DOCKER_PREFIX="rag-arc"

# ==============================================================================
# 核心优化：自动定位main.py所在目录
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
# 第一步：加载.env文件
# ==============================================================================
ENV_FILE=${ENV_FILE:-${APP_DIR}/.env.${ENV_ID}} # 使用环境特定的env文件
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    export $(grep -v '^#' ${ENV_FILE} | grep -v '^$' | xargs)
else
    echo "⚠️  未找到.env文件（${ENV_FILE}），将使用脚本默认值"
fi

# ==============================================================================
# 第二步：配置区 (使用新的命名规范，端口与测试环境保持一致)
# ==============================================================================
# 1. 网络名称
NETWORK_NAME=${NETWORK_NAME:-"${DOCKER_PREFIX}-network-${APP_ID}-${ENV_ID}"}

# 2. PostgreSQL 配置
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-"${DOCKER_PREFIX}-postgres-${APP_ID}-${ENV_ID}"}
POSTGRES_HOST_PORT=${POSTGRES_PORT:-5433} # 端口与测试环境一致
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123}
POSTGRES_DB=${POSTGRES_DB:-"rag_arc_${APP_ID}_${ENV_ID}"} # 数据库名也遵循规范
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:16-alpine}

# 3. Redis 配置
REDIS_CONTAINER_NAME=${REDIS_CONTAINER_NAME:-"${DOCKER_PREFIX}-redis-${APP_ID}-${ENV_ID}"}
REDIS_HOST_PORT=${REDIS_PORT:-6380} # 端口与测试环境一致
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
REDIS_PASSWORD=${REDIS_PASSWORD:-""}

# 4. Neo4j 配置
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME:-"${DOCKER_PREFIX}-neo4j-${APP_ID}-${ENV_ID}"}
NEO4J_WEB_HOST_PORT=${NEO4J_WEB_PORT:-7475} # 端口与测试环境一致
NEO4J_BOLT_HOST_PORT=${NEO4J_BOLT_PORT:-7688} # 端口与测试环境一致
if [[ -n "${NEO4J_URL:-}" ]]; then
    NEO4J_BOLT_HOST_PORT=$(echo ${NEO4J_URL} | awk -F':' '{print $NF}')
fi
NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-12345678}
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:latest}

# 5. 应用配置
APP_PORT=${APP_PORT:-8001} # 端口与测试环境一致
APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app_${ENV_ID}.log}
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-${APP_DIR}/data/parsed_files_${ENV_ID}}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-${APP_DIR}/local/files_${ENV_ID}}

# ==============================================================================
# 依赖检查
# ==============================================================================
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ 错误: 未找到 '$1' 命令。请先安装。" >&2
        exit 1
    fi
}
check_dependency "docker"
check_dependency "pm2"
check_dependency "uv"
check_dependency "lsof"
check_dependency "pkill"

# ==============================================================================
# 第三步：启动Docker容器
# ==============================================================================
echo -e "\n=========================================="
echo "  RAG-ARC ${APP_ID} ${ENV_ID} 环境启动脚本 (PM2版)"
echo "=========================================="
echo ""

# 1. 创建Docker网络
echo "📦 创建/检查Docker网络 [${NETWORK_NAME}]..."
docker network create ${NETWORK_NAME} 2>/dev/null || echo "  ✅ 网络已存在，无需创建"
echo ""

# 2. 处理PostgreSQL
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
        -v "${POSTGRES_CONTAINER_NAME}-data:/var/lib/postgresql/data" \
        ${POSTGRES_IMAGE} && echo "  ✅ PostgreSQL容器已创建并启动（新数据卷）"
fi
echo ""

# 3. 处理Redis
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
        -v "${REDIS_CONTAINER_NAME}-data:/data" \
        ${REDIS_IMAGE} ${REDIS_CMD} && echo "  ✅ Redis容器已创建并启动（新数据卷）"
fi
echo ""

# 4. 处理Neo4j
echo "🔷 处理Neo4j容器 [${NEO4J_CONTAINER_NAME}]..."
if docker ps -a | grep -q "${NEO4J_CONTAINER_NAME}"; then
    CURRENT_BOLT_PORT=$(docker port ${NEO4J_CONTAINER_NAME} 7687 2>/dev/null | awk -F':' '{print $2}')
    if [ -z "${CURRENT_BOLT_PORT}" ] || [ "${CURRENT_BOLT_PORT}" != "${NEO4J_BOLT_HOST_PORT}" ]; then
        echo "  ⚠️  Neo4j端口不匹配或未映射，删除并重建..."
        docker rm -f ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 || true
        docker run -d \
            --name ${NEO4J_CONTAINER_NAME} \
            --network ${NETWORK_NAME} \
            -p ${NEO4J_WEB_HOST_PORT}:7474 \
            -p ${NEO4J_BOLT_HOST_PORT}:7687 \
            -e NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD} \
            -e NEO4J_PLUGINS='["apoc"]' \
            -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
            -v "${NEO4J_CONTAINER_NAME}-data:/data" \
            -v "${NEO4J_CONTAINER_NAME}-logs:/logs" \
            ${NEO4J_IMAGE} && echo "  ✅ Neo4j容器已重建并启动"
    else
        docker stop ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Neo4j容器"
        docker start ${NEO4J_CONTAINER_NAME} && echo "  ✅ Neo4j容器重启成功（保留数据）"
    fi
else
    docker run -d \
        --name ${NEO4J_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        -p ${NEO4J_WEB_HOST_PORT}:7474 \
        -p ${NEO4J_BOLT_HOST_PORT}:7687 \
        -e NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD} \
        -e NEO4J_PLUGINS='["apoc"]' \
        -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
        -v "${NEO4J_CONTAINER_NAME}-data:/data" \
        -v "${NEO4J_CONTAINER_NAME}-logs:/logs" \
        ${NEO4J_IMAGE} && echo "  ✅ Neo4j容器已创建并启动（新数据卷）"
fi
echo ""

# 5. 等待中间件就绪
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

# ==============================================================================
# 第四步：使用PM2启动/重启应用
# ==============================================================================
echo "🚀 使用PM2启动/重启应用 [${PM2_APP_NAME}]..."
mkdir -p ${PARSER_OUTPUT_DIR} ${LOCAL_FILE_STORAGE_PATH} $(dirname ${APP_LOG_FILE})

cd ${APP_DIR} || { echo "  ❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

if [ -f "pyproject.toml" ]; then
    echo "  📦 同步项目依赖..."
    uv sync --quiet && echo "  ✅ 依赖同步完成" || echo "  ⚠️  依赖同步失败，尝试继续启动..."
fi

# 终极健壮版清理
echo "🧹 正在清理可能残留的旧应用进程和端口..."
pm2 stop "${PM2_APP_NAME}" > /dev/null 2>&1 || true
pm2 delete "${PM2_APP_NAME}" > /dev/null 2>&1 || true
pkill -f "uvicorn main:app --port ${APP_PORT}" > /dev/null 2>&1 || true
PID_TO_KILL=$(sudo lsof -t -i:"${APP_PORT}" 2>/dev/null || true)
if [ -n "${PID_TO_KILL}" ]; then
    echo "  ⚠️  发现进程 ${PID_TO_KILL} 占用端口 ${APP_PORT}，正在强制杀死..."
    sudo kill -9 "${PID_TO_KILL}" > /dev/null 2>&1 || true
    sleep 2
fi
echo "  ✅ 清理完成。"

# 使用PM2启动应用
echo "  🚀 正在通过PM2启动新的 ${PM2_APP_NAME} 进程..."
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

echo "  ✅ 应用已通过PM2启动！"
echo ""

# ==============================================================================
# 第五步：验证应用状态
# ==============================================================================
echo "⏳ 验证应用启动状态..."
sleep 5

if pm2 list | grep -q "${PM2_APP_NAME}" && pm2 list | grep -q "online"; then
    echo "=========================================="
    echo "  ✅ ${APP_ID} ${ENV_ID} 环境所有服务启动/重启成功！"
    echo "=========================================="
    echo ""
    echo "📋 服务信息："
    echo "  - PM2应用名: ${PM2_APP_NAME}"
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo "  - Docker网络: ${NETWORK_NAME}"
    echo ""
    echo "📝 PM2管理命令："
    echo "  - 查看状态: pm2 status"
    echo "  - 查看日志: pm2 logs ${PM2_APP_NAME}"
    echo "  - 重启应用: pm2 restart ${PM2_APP_NAME}"
    echo "  - 停止应用: pm2 stop ${PM2_APP_NAME}"
    echo "  - 删除应用: pm2 delete ${PM2_APP_NAME}"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  应用启动可能异常（请检查日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    echo "  1. 查看PM2状态: pm2 list"
    echo "  2. 查看应用日志: pm2 logs ${PM2_APP_NAME}"
    echo "  3. 检查容器状态: docker ps | grep ${APP_ID}-${ENV_ID}"
    echo ""
fi