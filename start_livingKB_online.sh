#!/bin/bash
set -euo pipefail

# ==============================================================================
# 线上环境专用配置 (遵循新的命名规范)
# ==============================================================================
# PM2 应用名称
PM2_APP_NAME="rag-app-livingKB_online"
# 应用标识 (用于Docker资源命名)
APP_ID="livingKB"
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
ENV_FILE=${ENV_FILE:-${APP_DIR}/.env} # 线上环境直接使用 .env 文件
if [ -f "${ENV_FILE}" ]; then
    echo "🔧 加载环境配置文件: ${ENV_FILE}"
    # 使用 set -a 自动导出所有变量，避免注释和空行问题
    set -a
    source ${ENV_FILE}
    set +a
else
    echo "⚠️  未找到.env文件（${ENV_FILE}），将使用脚本默认值或系统环境变量"
fi

# ==============================================================================
# 第二步：配置区 (使用新的命名规范，端口与测试环境保持一致)
# ==============================================================================
# 1. 网络名称
NETWORK_NAME=${NETWORK_NAME:-"${DOCKER_PREFIX}-network-${APP_ID}-${ENV_ID}"}

# 2. PostgreSQL 配置
POSTGRES_CONTAINER_NAME=${POSTGRES_CONTAINER_NAME:-"${DOCKER_PREFIX}-postgres-${APP_ID}-${ENV_ID}"}
POSTGRES_HOST_PORT=${POSTGRES_PORT:-5432} # 使用5432端口
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres123}
POSTGRES_DB=${POSTGRES_DB:-"rag_arc_${APP_ID}_${ENV_ID}"} # 数据库名也遵循规范
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:16-alpine}

# 3. Redis 配置
REDIS_CONTAINER_NAME=${REDIS_CONTAINER_NAME:-"${DOCKER_PREFIX}-redis-${APP_ID}-${ENV_ID}"}
REDIS_HOST_PORT=${REDIS_PORT:-6381} # 端口与测试环境一致
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
REDIS_PASSWORD=${REDIS_PASSWORD:-""}

# 4. Neo4j 配置
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME:-"${DOCKER_PREFIX}-neo4j-${APP_ID}-${ENV_ID}"}
NEO4J_WEB_HOST_PORT=${NEO4J_WEB_PORT:-7476} # 端口与测试环境一致
NEO4J_BOLT_HOST_PORT=${NEO4J_BOLT_PORT:-7689} # 端口与测试环境一致
if [[ -n "${NEO4J_URL:-}" ]]; then
    NEO4J_BOLT_HOST_PORT=$(echo ${NEO4J_URL} | awk -F':' '{print $NF}')
fi
NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-12345678}
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:latest}

# 5. 应用配置
APP_PORT=${APP_PORT:-8000} # 端口与测试环境一致
APP_LOG_FILE=${APP_LOG_FILE:-${APP_DIR}/log/app_${ENV_ID}.log}
PARSER_OUTPUT_DIR=${PARSER_OUTPUT_DIR:-${APP_DIR}/data/parsed_files_${ENV_ID}}
LOCAL_FILE_STORAGE_PATH=${LOCAL_FILE_STORAGE_PATH:-${APP_DIR}/local/files_${ENV_ID}}

# 6. MinerU 服务配置（客户端配置，用于连接远程 MinerU 服务）
# 注意：此环境为客户端，MinerU 服务在远程服务器运行，不在此处启动
MINERU_ENABLED=${MINERU_ENABLED:-false}
MINERU_HOST=${MINERU_HOST:-0.0.0.0}
MINERU_PORT=${MINERU_PORT:-8899}
MINERU_OUTPUT_DIR=${MINERU_OUTPUT_DIR:-${APP_DIR}/data/mineru_outputs}
MINERU_TEMP_DIR=${MINERU_TEMP_DIR:-/tmp/mineru_temp}
MINERU_LOG_FILE=${MINERU_LOG_FILE:-${APP_DIR}/log/mineru.log}

# 7. MinerU SSH 隧道：由用户自行使用 tmux 等维护，本脚本不自动建立隧道。

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
    docker update --restart always ${POSTGRES_CONTAINER_NAME} > /dev/null 2>&1 && echo "  🔄 已设置自动重启策略"
    docker start ${POSTGRES_CONTAINER_NAME} && echo "  ✅ PostgreSQL容器重启成功（保留数据，已启用自动重启）"
else
    docker run -d \
        --name ${POSTGRES_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        --restart always \
        -p 127.0.0.1:${POSTGRES_HOST_PORT}:5432 \
        -e POSTGRES_USER=${POSTGRES_USER} \
        -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
        -e POSTGRES_DB=${POSTGRES_DB} \
        -v "${POSTGRES_CONTAINER_NAME}-data:/var/lib/postgresql/data" \
        ${POSTGRES_IMAGE} && echo "  ✅ PostgreSQL容器已创建并启动（新数据卷，已启用自动重启）"
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
    docker update --restart always ${REDIS_CONTAINER_NAME} > /dev/null 2>&1 && echo "  🔄 已设置自动重启策略"
    docker start ${REDIS_CONTAINER_NAME} && echo "  ✅ Redis容器重启成功（保留数据，已启用自动重启）"
else
    docker run -d \
        --name ${REDIS_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        --restart always \
        -p ${REDIS_HOST_PORT}:6379 \
        -v "${REDIS_CONTAINER_NAME}-data:/data" \
        ${REDIS_IMAGE} ${REDIS_CMD} && echo "  ✅ Redis容器已创建并启动（新数据卷，已启用自动重启）"
fi
echo ""

# 4. 处理Neo4j（内存与是否重建由 .env：NEO4J_HEAP_* / NEO4J_RECREATE_IF_RESTARTING）
echo "🔷 处理Neo4j容器 [${NEO4J_CONTAINER_NAME}]..."
NEO4J_NEED_CREATE=false
if docker ps -a | grep -q "${NEO4J_CONTAINER_NAME}"; then
    NEO4J_STATUS=$(docker inspect -f '{{.State.Status}}' ${NEO4J_CONTAINER_NAME} 2>/dev/null || true)
    RECREATE="${NEO4J_RECREATE_IF_RESTARTING:-true}"
    if [[ "${RECREATE}" == "true" || "${RECREATE}" == "1" || "${RECREATE}" == "yes" ]] && [[ "${NEO4J_STATUS}" == "restarting" ]]; then
        docker rm -f ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  🔄 Neo4j 处于 Restarting，已删除并将在下方重新创建"
        NEO4J_NEED_CREATE=true
    else
        docker stop ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  ⏹️  已停止旧Neo4j容器"
        docker update --restart always ${NEO4J_CONTAINER_NAME} > /dev/null 2>&1 && echo "  🔄 已设置自动重启策略"
        docker start ${NEO4J_CONTAINER_NAME} && echo "  ✅ Neo4j容器重启成功（保留数据，已启用自动重启）"
    fi
else
    NEO4J_NEED_CREATE=true
fi
if [[ "${NEO4J_NEED_CREATE}" == "true" ]]; then
    NEO4J_HEAP_INITIAL=${NEO4J_HEAP_INITIAL:-256m}
    NEO4J_HEAP_MAX=${NEO4J_HEAP_MAX:-512m}
    NEO4J_PAGECACHE_SIZE=${NEO4J_PAGECACHE_SIZE:-128m}
    docker run -d \
        --name ${NEO4J_CONTAINER_NAME} \
        --network ${NETWORK_NAME} \
        --restart always \
        -p 127.0.0.1:${NEO4J_WEB_HOST_PORT}:7474 \
        -p 127.0.0.1:${NEO4J_BOLT_HOST_PORT}:7687 \
        -e NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD} \
        -e NEO4J_PLUGINS='["apoc"]' \
        -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
        -e NEO4J_server_memory_heap_initial_size="${NEO4J_HEAP_INITIAL}" \
        -e NEO4J_server_memory_heap_max_size="${NEO4J_HEAP_MAX}" \
        -e NEO4J_server_memory_pagecache_size="${NEO4J_PAGECACHE_SIZE}" \
        -v "${NEO4J_CONTAINER_NAME}-data:/data" \
        -v "${NEO4J_CONTAINER_NAME}-logs:/logs" \
        ${NEO4J_IMAGE} && echo "  ✅ Neo4j容器已创建并启动（新数据卷，已启用自动重启）"
fi
echo ""

# 5. 等待中间件就绪（PG/Redis/Neo4j，由 .env 的 WAIT_MIDDLEWARE_SLEEP / WAIT_MAX_ATTEMPTS 配置）
WAIT_SLEEP=${WAIT_MIDDLEWARE_SLEEP:-15}
MAX_ATTEMPTS=${WAIT_MAX_ATTEMPTS:-45}
echo "⏳ 等待中间件就绪 (PG/Redis/Neo4j)，间隔 ${WAIT_SLEEP}s，最多 ${MAX_ATTEMPTS} 次..."
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    PG_OK=false; REDIS_OK=false; NEO4J_OK=false
    docker exec ${POSTGRES_CONTAINER_NAME} pg_isready -U ${POSTGRES_USER} -h localhost > /dev/null 2>&1 && PG_OK=true
    docker exec ${REDIS_CONTAINER_NAME} redis-cli ping > /dev/null 2>&1 && REDIS_OK=true
    docker exec ${NEO4J_CONTAINER_NAME} cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" "RETURN 1" > /dev/null 2>&1 && NEO4J_OK=true
    if [[ "${PG_OK}" == "true" && "${REDIS_OK}" == "true" && "${NEO4J_OK}" == "true" ]]; then
        echo "  ✅ PostgreSQL、Redis、Neo4j 均已就绪"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep "${WAIT_SLEEP}"
done
echo ""
if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "  ⚠️  中间件就绪超时 (PG=${PG_OK} Redis=${REDIS_OK} Neo4j=${NEO4J_OK})，将继续尝试启动应用（请检查容器日志）"
fi

# 5.1 同步 PostgreSQL 密码（确保 .env 中的密码与数据库一致，每次重启都会执行）
echo "🔐 同步 PostgreSQL 密码至数据库..."
PG_PASS_ESC="${POSTGRES_PASSWORD//\'/\'\'}"  # 转义单引号
if docker exec ${POSTGRES_CONTAINER_NAME} psql -U ${POSTGRES_USER} -h localhost -tAc "ALTER USER ${POSTGRES_USER} PASSWORD '${PG_PASS_ESC}';" 2>/dev/null; then
    echo "  ✅ PostgreSQL 密码已同步（来自 .env）"
else
    echo "  ⚠️  密码同步失败，请检查容器状态或手动执行 ALTER USER"
fi
echo ""

# ==============================================================================
# 第六步：使用PM2启动/重启应用
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

# 4.1 【核心修改】增强版终极清理：确保端口绝对干净
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

# 4.2 使用 PM2 启动新进程（带环境变量更新）
echo "  🚀 正在通过 PM2 启动新的 [${PM2_APP_NAME}] 进程..."
pm2 start "uv run uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}" \
    --name "${PM2_APP_NAME}" \
    --log "${APP_LOG_FILE}" \
    --update-env \
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
# 6.3 启动/重启队列 Workers (参考 start.sh 的做法，直接调用脚本)
# ==============================================================================
echo "🔄 启动/重启队列 Workers..."
cd ${APP_DIR} || { echo "❌ 应用目录不存在: ${APP_DIR}"; exit 1; }

# 先停止旧的队列 workers (如果存在 stop 脚本)
if [ -f "scripts/mq_tools/stop_mq_workers_local.sh" ]; then
    echo "  🛑 停止旧的队列 workers..."
    bash "scripts/mq_tools/stop_mq_workers_local.sh" >/dev/null 2>&1 || true
fi

# 使用 start_mq_workers_local.sh 启动新的 workers (强制启动模式，因为我们已经先停止了)
if [ -f "scripts/mq_tools/start_mq_workers_local.sh" ]; then
    echo "  🚀 启动新的队列 workers..."
    MQ_WORKER_FORCE_START=1 bash "scripts/mq_tools/start_mq_workers_local.sh" || true
    
    # 自动配置队列 Workers 的开机自启（通过 systemd service）
    SERVICE_NAME="rag-arc-mq-workers-livingKB_online.service"
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
    
    echo "  🔧 配置队列 Workers 开机自启..."
    cat > /tmp/${SERVICE_NAME} <<EOF
[Unit]
Description=RAG-ARC MQ Workers (Celery Workers for livingKB_online)
After=network.target docker.service
Requires=docker.service

[Service]
Type=forking
User=root
WorkingDirectory=${APP_DIR}
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin"
EnvironmentFile=-${APP_DIR}/.env
ExecStart=/bin/bash scripts/mq_tools/start_mq_workers_local.sh
ExecStop=/bin/bash scripts/mq_tools/stop_mq_workers_local.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    sudo cp /tmp/${SERVICE_NAME} ${SERVICE_FILE} && \
    sudo systemctl daemon-reload && \
    sudo systemctl enable ${SERVICE_NAME} >/dev/null 2>&1 && \
    echo "  ✅ 队列 Workers 开机自启已配置（systemd service: ${SERVICE_NAME}）" || \
    echo "  ⚠️  队列 Workers 开机自启配置失败（需要 sudo 权限）"
    rm -f /tmp/${SERVICE_NAME}
else
    echo "  ⚠️  未找到 start_mq_workers_local.sh 脚本，跳过队列 workers 启动"
fi
echo ""

# ==============================================================================
# 第七步：验证应用状态
# ==============================================================================
echo "⏳ 验证应用启动状态..."
sleep 8

# 验证 PM2 应用状态
if pm2 list | grep -q "${PM2_APP_NAME}" && pm2 list | grep -q "online"; then
    echo "=========================================="
    echo "  ✅ ${APP_ID} ${ENV_ID} 所有服务启动/重启成功！（由 PM2 守护）"
    echo "=========================================="
    echo ""
    echo "📋 服务信息（自动适配路径）："
    echo "  - PostgreSQL: ${POSTGRES_HOST:-localhost}:${POSTGRES_HOST_PORT} (容器: ${POSTGRES_CONTAINER_NAME})"
    echo "  - Redis: ${REDIS_HOST:-localhost}:${REDIS_HOST_PORT} (容器: ${REDIS_CONTAINER_NAME})"
    echo "  - Neo4j Web: http://localhost:${NEO4J_WEB_HOST_PORT}"
    echo "  - Neo4j Bolt: ${NEO4J_URL:-bolt://localhost:${NEO4J_BOLT_HOST_PORT}} (容器: ${NEO4J_CONTAINER_NAME})"
    if [[ "${MINERU_ENABLED}" == "true" ]] || [[ "${PARSER_PARSE_MODE:-}" == "mineru" ]]; then
        MINERU_SERVER_URL=${MINERU_SERVER_URL:-"http://${MINERU_HOST}:${MINERU_PORT}"}
        echo "  - MinerU 服务（远程/隧道由 tmux 等自行维护）: ${MINERU_SERVER_URL}"
    fi
    echo "  - 应用API: http://localhost:${APP_PORT}"
    echo "  - 自动定位的代码目录: ${APP_DIR}"
    echo ""
    echo "📝 PM2 管理命令："
    echo "  - 查看状态: pm2 list"
    echo "  - 查看日志: pm2 logs ${PM2_APP_NAME}"
    echo "  - 重启应用: pm2 restart ${PM2_APP_NAME} --update-env"
    echo "  - 停止应用: pm2 stop ${PM2_APP_NAME}"
    echo "  - 开机自启: pm2 startup && pm2 save"
    echo ""
    echo "📝 队列 Workers 管理命令："
    echo "  - 停止队列: bash scripts/mq_tools/stop_mq_workers_local.sh"
    echo "  - 启动队列: MQ_WORKER_FORCE_START=1 bash scripts/mq_tools/start_mq_workers_local.sh"
    echo "  - 查看队列日志: tail -f log/mq_workers/*.log"
    echo "  - 系统服务管理: sudo systemctl {start|stop|status|restart} rag-arc-mq-workers-livingKB_online.service"
    echo "  - 开机自启: ✅ 已自动配置（通过 systemd service）"
    echo ""
else
    echo "=========================================="
    echo "  ⚠️  ${APP_ID} ${ENV_ID} 应用启动可能异常（请检查 PM2 日志）"
    echo "=========================================="
    echo ""
    echo "📝 快速排查："
    echo "  1. 查看 PM2 状态: pm2 list"
    echo "  2. 查看应用日志: pm2 logs ${PM2_APP_NAME}"
    echo "  3. 检查容器状态: docker ps | grep ${APP_ID}-${ENV_ID}"
    echo ""
fi