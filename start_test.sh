#!/bin/bash
set -euo pipefail

# ==============================================================================
# 第一步：加载.env文件（优先读取，无则用默认值）
# ==============================================================================
ENV_FILE=${ENV_FILE:-./.env}
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    # 读取.env文件，过滤注释和空行，导出为环境变量
    export $(grep -v '^#' ${ENV_FILE} | grep -v '^$' | xargs)
else
    echo "⚠️  未找到.env文件，使用脚本默认值"
fi

# ==============================================================================
# 第二步：配置区（优先用.env变量，无则用默认值，完全对齐你的.env）
# ==============================================================================
# 1. 网络名称
NETWORK_NAME=${NETWORK_NAME:-rag-arc-network}

# 2. PostgreSQL 配置（对齐.env：5432端口、密码postgres123）
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-rag-arc-postgres}
POSTGRES_HOST_PORT=${POSTGRES_PORT:-5432}          # 从.env的POSTGRES_PORT读取
POSTGRES_USER=${POSTGRES_USER:-postgres}           # 从.env读取
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123} # 默认值对齐你的.env
POSTGRES_DB=${POSTGRES_DB:-rag_arc}                # 从.env读取
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:16-alpine}

# 3. Redis 配置（对齐.env：6379端口、无密码）
REDIS_CONTAINER_NAME=${REDIS_CONTAINER_NAME:-rag-arc-redis}
REDIS_HOST_PORT=${REDIS_PORT:-6379}                # 从.env的REDIS_PORT读取
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
REDIS_PASSWORD=${REDIS_PASSWORD:-""}               # 从.env读取（无密码）

# 4. Neo4j 配置（对齐.env：7687端口、密码12345678）
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME:-rag-arc-neo4j}
NEO4J_WEB_HOST_PORT=${NEO4J_WEB_PORT:-7474}        # Neo4j Web默认7474
NEO4J_BOLT_HOST_PORT=${NEO4J_BOLT_PORT:-7687}      # 从.env的NEO4J_URL解析（默认7687）
# 从.env的NEO4J_URL解析Bolt端口（兼容bolt://localhost:7687格式）
if [[ -n "${NEO4J_URL:-}" ]]; then
    NEO4J_BOLT_HOST_PORT=$(echo ${NEO4J_URL} | awk -F':' '{print $NF}')
fi
NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}            # 从.env读取
NEO4J_PASSWORD=${NEO4J_PASSWORD:-12345678}         # 默认值对齐你的.env
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:latest}

# 5. 应用配置（对齐.env的路径）
APP_DIR=${APP_DIR:-/opt/dlami/nvme/rag-arc}        # 应用根目录
APP_PORT=${APP_PORT:-8000}                         # 应用端口
APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app.log}   # 日志路径（统一到log目录）
# 从.env读取文件存储路径
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-/opt/dlami/nvme/rag-arc/data/parsed_files}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-/opt/dlami/nvme/rag-arc/local/files}

# ==============================================================================
# 第三步：核心功能（兼容已有容器：仅重启，无则创建，保留数据）
# ==============================================================================
echo -e "\n=========================================="
echo "  RAG-ARC 启动脚本 (适配线上环境+保留数据)"
echo "=========================================="
echo ""

# 1. 创建Docker网络（兼容已有网络）
echo "📦 创建/检查Docker网络 [${NETWORK_NAME}]..."
docker network create ${NETWORK_NAME} 2>/dev/null || echo "  ✅ 网络已存在，无需创建"
echo ""

# 2. 处理PostgreSQL（已有则重启，无则创建，保留数据卷）
echo "🗄️  处理PostgreSQL容器 [${POSTGRES_CONTAINER_NAME}]..."
if docker ps -a | grep -q "${POSTGRES_CONTAINER_NAME}"; then
    # 已有容器：先停止→再启动（确保配置生效，保留数据）
    docker stop ${POSTGRES_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧PostgreSQL容器"
    docker start ${POSTGRES_CONTAINER_NAME} && echo "  ✅ PostgreSQL容器重启成功（保留数据）"
else
    # 无容器：创建新容器+数据卷（持久化数据）
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

# 3. 处理Redis（已有则重启，无则创建，保留数据卷）
echo "📦 处理Redis容器 [${REDIS_CONTAINER_NAME}]..."
REDIS_CMD="redis-server --appendonly yes"
# 若有Redis密码，添加认证配置
if [[ -n "${REDIS_PASSWORD}" ]]; then
    REDIS_CMD="redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}"
fi

if docker ps -a | grep -q "${REDIS_CONTAINER_NAME}"; then
    # 已有容器：重启
    docker stop ${REDIS_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Redis容器"
    docker start ${REDIS_CONTAINER_NAME} && echo "  ✅ Redis容器重启成功（保留数据）"
else
    # 无容器：创建新容器+数据卷
    docker run -d \
        --name ${REDIS_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        -p ${REDIS_HOST_PORT}:6379 \
        -v rag-arc-redis-data:/data \
        ${REDIS_IMAGE} ${REDIS_CMD} && echo "  ✅ Redis容器已创建并启动（新数据卷）"
fi
echo ""

# 4. 处理Neo4j（已有则重启，无则创建，保留数据卷）
echo "🔷 处理Neo4j容器 [${NEO4J_CONTAINER_NAME}]..."
if docker ps -a | grep -q "${NEO4J_CONTAINER_NAME}"; then
    # 已有容器：重启
    docker stop ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Neo4j容器"
    docker start ${NEO4J_CONTAINER_NAME} && echo "  ✅ Neo4j容器重启成功（保留数据）"
else
    # 无容器：创建新容器+数据卷
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

# 5. 等待中间件就绪（确保服务可用）
echo "⏳ 等待中间件就绪..."
sleep 10

# 检查PostgreSQL（带重试，确保真正就绪）
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker exec ${POSTGRES_CONTAINER_NAME} pg_isready -U ${POSTGRES_USER} > /dev/null 2>&1; then
        echo "  ✅ PostgreSQL已就绪 (端口: ${POSTGRES_HOST_PORT})"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 1
done
echo ""

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "  ⚠️  PostgreSQL启动超时，但继续启动应用（请检查容器日志）"
fi

# 6. 启动应用（重启旧进程，加载最新.env配置）
echo "🚀 启动/重启应用 [${APP_DIR}]..."
# 创建文件存储目录（避免应用启动报错）
mkdir -p ${PARSER_OUTPUT_DIR} ${LOCAL_FILE_STORAGE_PATH}

cd ${APP_DIR} || { echo "  ❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

# 停止旧应用进程（避免端口占用）
pkill -f "uvicorn main:app" 2>/dev/null && echo "  ⏹️  已停止旧应用进程" || echo "  ℹ️  无旧应用进程"
sleep 2

# 启动新进程（完整加载.env的所有配置）
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
nohup uv run uvicorn main:app --host 0.0.0.0 --port ${APP_PORT} >> ${APP_LOG_FILE} 2>&1 &

echo "  ✅ 应用已启动（后台运行，加载最新.env配置）"
echo ""

# 7. 验证应用状态
echo "⏳ 验证应用启动状态..."
sleep 8

if curl -s http://localhost:${APP_PORT}/ > /dev/null 2>&1; then
    echo "=========================================="
    echo "  ✅ 所有服务启动/重启成功！（保留数据）"
    echo "=========================================="
    echo ""
    echo "📋 服务信息（对齐你的.env）："
    echo "  - PostgreSQL: ${POSTGRES_HOST}:${POSTGRES_HOST_PORT} (容器: ${POSTGRES_CONTAINER_NAME})"
    echo "  - Redis: ${REDIS_HOST}:${REDIS_HOST_PORT} (容器: ${REDIS_CONTAINER_NAME})"
    echo "  - Neo4j Web: localhost:${NEO4J_WEB_HOST_PORT}"
    echo "  - Neo4j Bolt: ${NEO4J_URL} (容器: ${NEO4J_CONTAINER_NAME})"
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo ""
    echo "📝 关键提示："
    echo "  - 容器数据已保留（通过数据卷 rag-arc-*-data）"
    echo "  - 应用日志: tail -f ${APP_LOG_FILE}"
    echo "  - 容器日志: docker logs ${POSTGRES_CONTAINER_NAME}"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  应用启动可能异常（请检查日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    echo "  1. 查看应用日志: tail -50 ${APP_LOG_FILE}"
    echo "  2. 检查容器状态: docker ps | grep rag-arc-"
    echo "  3. 验证数据库连接: psql -h localhost -p ${POSTGRES_HOST_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}"
    echo ""
fi