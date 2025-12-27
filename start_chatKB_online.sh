#!/bin/bash
set -euo pipefail

# ==============================================================================
# 核心优化：自动定位main.py所在目录（无需写死路径）
# ==============================================================================
# 第一步：查找main.py的绝对路径（优先当前目录，再递归上级）
find_main_py() {
    local current_dir=$(pwd)
    local max_depth=10  # 最多向上查找10层，避免无限循环
    local depth=0

    # 1. 先检查当前目录是否有main.py
    if [ -f "${current_dir}/main.py" ]; then
        echo "${current_dir}"
        return 0
    fi

    # 2. 递归向上查找main.py
    while [ $depth -lt $max_depth ]; do
        current_dir=$(dirname "${current_dir}")
        if [ -f "${current_dir}/main.py" ]; then
            echo "${current_dir}"
            return 0
        fi
        depth=$((depth + 1))
    done

    # 3. 未找到main.py，报错退出
    echo "❌ 未找到main.py文件（已向上查找${max_depth}层）" >&2
    exit 1
}

# 自动获取APP_DIR（main.py所在目录）
APP_DIR=$(find_main_py)
echo "🔍 自动定位到应用目录: ${APP_DIR}"

# ==============================================================================
# 第一步：加载.env文件（优先读取APP_DIR下的.env）
# ==============================================================================
ENV_FILE=${ENV_FILE:-${APP_DIR}/.env}
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    # 读取.env文件，过滤注释和空行，导出为环境变量
    export $(grep -v '^#' ${ENV_FILE} | grep -v '^$' | xargs)
else
    echo "⚠️  未找到.env文件（${ENV_FILE}），将使用脚本默认值"
fi

# ==============================================================================
# 第二步：配置区（完全基于自动获取的APP_DIR，无硬编码路径）
# ==============================================================================
# 1. 网络名称
NETWORK_NAME=${NETWORK_NAME:-rag-arc-network}

# 2. PostgreSQL 配置（对齐.env：5432端口、密码postgres123）
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-rag-arc-postgres}
POSTGRES_HOST_PORT=${POSTGRES_PORT:-5432}          # 从.env的POSTGRES_PORT读取
POSTGRES_USER=${POSTGRES_USER:-postgres}           # 从.env读取
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123} # 默认值对齐你的.env
POSTGRES_DB=${POSTGRES_DB:-rag_arc_chatKB}         # 【核心改动】数据库名称改为 rag_arc_chatKB
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

# 5. 应用配置（基于自动获取的APP_DIR，无硬编码）
APP_PORT=${APP_PORT:-8001}                         # 应用端口
APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app.log}   # 日志路径（放到APP_DIR下）
# 从.env读取文件存储路径（放到APP_DIR下，避免权限问题）
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-${APP_DIR}/data/parsed_files}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-${APP_DIR}/local/files}

# ==============================================================================
# 第三步：核心功能（兼容已有容器：仅重启，无则创建，保留数据）
# ==============================================================================
echo -e "\n=========================================="
echo "  RAG-ARC 启动脚本 (自动适配路径+保留数据)"
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

# 4. 处理Neo4j（已有则检查端口映射，无则创建，保留数据卷）
echo "🔷 处理Neo4j容器 [${NEO4J_CONTAINER_NAME}]..."
if docker ps -a | grep -q "${NEO4J_CONTAINER_NAME}"; then
    # 检查现有容器的Bolt端口映射是否匹配
    CURRENT_BOLT_PORT=$(docker port ${NEO4J_CONTAINER_NAME} 7687 | cut -d':' -f2 || true)
    if [ "${CURRENT_BOLT_PORT}" != "${NEO4J_BOLT_HOST_PORT}" ]; then
        echo "  ⚠️  Neo4j容器Bolt端口映射不匹配 (${CURRENT_BOLT_PORT} != ${NEO4J_BOLT_HOST_PORT})，删除并重建容器..."
        docker stop ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 || true
        docker rm ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 || true
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
            ${NEO4J_IMAGE} && echo "  ✅ Neo4j容器已重建并启动（保留数据卷）"
    else
        # 端口映射匹配，只需重启
        docker stop ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Neo4j容器"
        docker start ${NEO4J_CONTAINER_NAME} && echo "  ✅ Neo4j容器重启成功（保留数据）"
    fi
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
# 创建文件存储目录和日志目录（避免应用启动报错）
mkdir -p ${PARSER_OUTPUT_DIR} ${LOCAL_FILE_STORAGE_PATH} $(dirname ${APP_LOG_FILE})

# 确保uv命令在PATH中
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"

# 检查uv是否可用
if ! command -v uv &> /dev/null; then
    echo "  ❌ uv命令未找到，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd ${APP_DIR} || { echo "  ❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

# 检查必要文件是否存在（双重验证，防止查找逻辑出错）
if [ ! -f "main.py" ]; then
    echo "  ❌ 应用目录中缺少 main.py: ${APP_DIR}/main.py"
    exit 1
fi

# 如果存在pyproject.toml，确保依赖已安装
if [ -f "pyproject.toml" ]; then
    echo "  📦 同步项目依赖..."
    if uv sync --quiet; then
        echo "  ✅ 依赖同步完成"
    else
        echo "  ⚠️  依赖同步失败，尝试继续启动..."
    fi
fi

# 停止旧应用进程（避免端口占用）
pkill -f "uvicorn main:app" 2>/dev/null && echo "  ⏹️  已停止旧应用进程" || echo "  ℹ️  无旧应用进程"
sleep 2

# 启动新进程（完整加载.env的所有配置）
# 注意：PYTHONUNBUFFERED=1 确保日志实时输出（虽然文件handler已经是行缓冲）
# 请求日志会实时写入到 log/ 目录下的文件（由代码的 DailySizeRotatingHandler 管理）
# 此处的 APP_LOG_FILE 只包含 uvicorn 的启动信息和错误输出
PYTHONUNBUFFERED=1 \
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
    echo "📋 服务信息（自动适配路径）："
    echo "  - PostgreSQL: ${POSTGRES_HOST}:${POSTGRES_HOST_PORT} (容器: ${POSTGRES_CONTAINER_NAME})"
    echo "  - Redis: ${REDIS_HOST}:${REDIS_HOST_PORT} (容器: ${REDIS_CONTAINER_NAME})"
    echo "  - Neo4j Web: localhost:${NEO4J_WEB_HOST_PORT}"
    echo "  - Neo4j Bolt: ${NEO4J_URL} (容器: ${NEO4J_CONTAINER_NAME})"
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo "  - 自动定位的代码目录: ${APP_DIR}"
    echo ""
    echo "📝 关键提示："
    echo "  - 容器数据已保留（通过数据卷 rag-arc-*-data）"
    echo "  - 应用启动日志: tail -f ${APP_LOG_FILE} (仅包含 uvicorn 启动信息)"
    echo "  - 请求日志（实时）: tail -f ${APP_DIR}/log/log-\$(date +%Y-%m-%d)/*.log"
    echo "  - 容器日志: docker logs ${POSTGRES_CONTAINER_NAME}"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  应用启动可能异常（请检查日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    if [ -f "${APP_LOG_FILE}" ]; then
        echo "  最近日志内容："
        tail -30 "${APP_LOG_FILE}" | sed 's/^/    /'
        echo ""
        echo "  1. 查看完整应用日志: tail -50 ${APP_LOG_FILE}"
    else
        echo "  ⚠️  日志文件不存在: ${APP_LOG_FILE}"
        echo "  可能原因：应用启动失败，请检查："
        echo "    - 应用目录是否存在: ${APP_DIR}"
        echo "    - main.py是否存在: ${APP_DIR}/main.py"
        echo "    - uv命令是否可用: $(which uv 2>/dev/null || echo '未找到')"
        echo ""
    fi
    echo "  2. 检查容器状态: docker ps | grep rag-arc-"
    echo "  3. 验证数据库连接: psql -h localhost -p ${POSTGRES_HOST_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}"
    echo "  4. 检查应用进程: ps aux | grep uvicorn | grep -v grep"
    echo ""
fi