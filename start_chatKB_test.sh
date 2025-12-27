#!/bin/bash
set -euo pipefail

# ==============================================================================
# 核心优化：自动定位main.py所在目录（无需写死路径）
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
# 加载.env文件（和正常脚本完全一致）
# ==============================================================================
ENV_FILE=${ENV_FILE:-${APP_DIR}/.env}
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    export $(grep -v '^#' ${ENV_FILE} | grep -v '^$' | xargs)
else
    echo "⚠️  未找到.env文件（${ENV_FILE}），使用脚本默认值"
fi

# ==============================================================================
# 第二步：配置区（仅修改端口/名称，其余和正常脚本一致）
# ==============================================================================
# 1. 网络名称（加chatKB_test后缀，隔离）
NETWORK_NAME=${NETWORK_NAME:-rag-arc-network-chatKB_test}

# 2. PostgreSQL 配置（仅改端口5433+名称+数据库名，其余一致）
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-rag-arc-postgres-chatKB_test}
POSTGRES_HOST_PORT=${POSTGRES_PORT:-5433}          # 仅改端口：5433
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123}
POSTGRES_DB=${POSTGRES_DB:-rag_arc_chatKB_test}    # 仅改数据库名：rag_arc_chatKB_test
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:16-alpine}

# 3. Redis 配置（仅改端口6380+名称，其余一致）
REDIS_CONTAINER_NAME=${REDIS_CONTAINER_NAME:-rag-arc-redis-chatKB_test}
REDIS_HOST_PORT=${REDIS_PORT:-6380}                # 仅改端口：6380
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
REDIS_PASSWORD=${REDIS_PASSWORD:-""}

# 4. Neo4j 配置（仅改端口7475/7688+名称，其余一致）
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME:-rag-arc-neo4j-chatKB_test}
NEO4J_WEB_HOST_PORT=${NEO4J_WEB_PORT:-7475}        # 仅改Web端口：7475
NEO4J_BOLT_HOST_PORT=${NEO4J_BOLT_PORT:-7688}      # 仅改Bolt端口：7688
if [[ -n "${NEO4J_URL:-}" ]]; then
    NEO4J_BOLT_HOST_PORT=$(echo ${NEO4J_URL} | awk -F':' '{print $NF}')
fi
NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-12345678}
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:latest}

# 5. 应用配置（仅改端口8001+日志名称，其余一致）
APP_PORT=${APP_PORT:-8001}                         # 仅改应用端口：8001
APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app_chatKB_test.log}   # 仅改日志名
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-${APP_DIR}/data/parsed_files_chatKB_test}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-${APP_DIR}/local/files_chatKB_test}

# ==============================================================================
# 第三步：核心功能（和正常脚本完全一致，无额外修改）
# ==============================================================================
echo -e "\n=========================================="
echo "  RAG-ARC chatKB_test启动脚本 (自动适配路径+保留数据)"
echo "=========================================="
echo ""

# 1. 创建Docker网络
echo "📦 创建/检查Docker网络 [${NETWORK_NAME}]..."
docker network create ${NETWORK_NAME} 2>/dev/null || echo "  ✅ 网络已存在，无需创建"
echo ""

# 2. 处理PostgreSQL（逻辑和正常脚本一致）
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
        -v rag-arc-postgres-data-chatKB_test:/var/lib/postgresql/data \
        ${POSTGRES_IMAGE} && echo "  ✅ PostgreSQL容器已创建并启动（新数据卷）"
fi
echo ""

# 3. 处理Redis（逻辑和正常脚本一致）
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
        -v rag-arc-redis-data-chatKB_test:/data \
        ${REDIS_IMAGE} ${REDIS_CMD} && echo "  ✅ Redis容器已创建并启动（新数据卷）"
fi
echo ""

# 4. 处理Neo4j（逻辑和正常脚本一致）
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
        -v rag-arc-neo4j-data-chatKB_test:/data \
        -v rag-arc-neo4j-logs-chatKB_test:/logs \
        ${NEO4J_IMAGE} && echo "  ✅ Neo4j容器已创建并启动（新数据卷）"
fi
echo ""

# 5. 等待中间件就绪（和正常脚本一致）
echo "⏳ 等待中间件就绪..."
sleep 10

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

# 6. 启动应用（逻辑和正常脚本一致，仅改进程匹配为8001端口）
echo "🚀 启动/重启chatKB_test应用 [${APP_DIR}]..."
mkdir -p ${PARSER_OUTPUT_DIR} ${LOCAL_FILE_STORAGE_PATH} $(dirname ${APP_LOG_FILE})

if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "  ❌ uv命令未找到，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd ${APP_DIR} || { echo "  ❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

if [ ! -f "main.py" ]; then
    echo "  ❌ 应用目录中缺少 main.py: ${APP_DIR}/main.py"
    exit 1
fi

if [ -f "pyproject.toml" ]; then
    echo "  📦 同步项目依赖..."
    if uv sync --quiet; then
        echo "  ✅ 依赖同步完成"
    else
        echo "  ⚠️  依赖同步失败，尝试继续启动..."
    fi
fi

# 仅停止chatKB_test的应用进程（避免影响第一套）
pkill -f "uvicorn main:app --port ${APP_PORT}" 2>/dev/null && echo "  ⏹️  已停止旧chatKB_test应用进程" || echo "  ℹ️  无旧chatKB_test应用进程"
sleep 2

# 启动新进程（和正常脚本一致，仅用chatKB_test的配置）
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

echo "  ✅ chatKB_test应用已启动（后台运行，加载最新.env配置）"
echo ""

# 7. 验证应用状态（和正常脚本一致，仅改端口）
echo "⏳ 验证chatKB_test应用启动状态..."
sleep 8

if curl -s http://localhost:${APP_PORT}/ > /dev/null 2>&1; then
    echo "=========================================="
    echo "  ✅ chatKB_test所有服务启动/重启成功！（保留数据）"
    echo "=========================================="
    echo ""
    echo "📋 chatKB_test服务信息（自动适配路径）："
    echo "  - PostgreSQL: ${POSTGRES_HOST}:${POSTGRES_HOST_PORT} (容器: ${POSTGRES_CONTAINER_NAME})"
    echo "  - Redis: ${REDIS_HOST}:${REDIS_HOST_PORT} (容器: ${REDIS_CONTAINER_NAME})"
    echo "  - Neo4j Web: localhost:${NEO4J_WEB_HOST_PORT}"
    echo "  - Neo4j Bolt: ${NEO4J_URL} (容器: ${NEO4J_CONTAINER_NAME})"
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo "  - 自动定位的代码目录: ${APP_DIR}"
    echo ""
    echo "📝 关键提示："
    echo "  - chatKB_test容器数据已保留（数据卷 rag-arc-*-data-chatKB_test）"
    echo "  - chatKB_test应用日志: tail -f ${APP_LOG_FILE}"
    echo "  - chatKB_test容器日志: docker logs ${POSTGRES_CONTAINER_NAME}"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  chatKB_test应用启动可能异常（请检查日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    if [ -f "${APP_LOG_FILE}" ]; then
        echo "  chatKB_test最近日志内容："
        tail -30 "${APP_LOG_FILE}" | sed 's/^/    /'
        echo ""
        echo "  1. 查看完整应用日志: tail -50 ${APP_LOG_FILE}"
    else
        echo "  ⚠️  chatKB_test日志文件不存在: ${APP_LOG_FILE}"
        echo "  可能原因：应用启动失败，请检查："
        echo "    - 应用目录是否存在: ${APP_DIR}"
        echo "    - main.py是否存在: ${APP_DIR}/main.py"
        echo "    - uv命令是否可用: $(which uv 2>/dev/null || echo '未找到')"
        echo ""
    fi
    echo "  2. 检查chatKB_test容器状态: docker ps | grep chatKB_test"
    echo "  3. 验证chatKB_test数据库连接: psql -h localhost -p ${POSTGRES_HOST_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}"
    echo "  4. 检查chatKB_test应用进程: ps aux | grep 'uvicorn main:app --port ${APP_PORT}' | grep -v grep"
    echo ""
fi