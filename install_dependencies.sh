#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  安装 RAG-ARC 启动脚本所需依赖"
echo "=========================================="
echo ""

# 检查并安装依赖的函数
check_and_install() {
    local cmd=$1
    local package=${2:-$1}
    
    if command -v "$cmd" &> /dev/null; then
        echo "✅ $cmd 已安装"
        return 0
    else
        echo "📦 正在安装 $package..."
        apt install -y "$package" || {
            echo "❌ 安装 $package 失败"
            return 1
        }
        echo "✅ $package 安装成功"
    fi
}

# 1. 检查 Docker
echo "1. 检查 Docker..."
if command -v docker &> /dev/null; then
    echo "✅ docker 已安装"
else
    echo "📦 正在安装 Docker..."
    curl -fsSL https://get.docker.com | sh || {
        echo "⚠️  官方脚本安装失败，尝试使用 apt 安装..."
        apt update
        apt install -y docker.io
    }
    systemctl start docker
    systemctl enable docker
    echo "✅ Docker 安装成功"
fi
echo ""

# 2. 检查 PM2
echo "2. 检查 PM2..."
if command -v pm2 &> /dev/null; then
    echo "✅ pm2 已安装"
else
    echo "📦 正在安装 PM2..."
    # 先安装 Node.js 和 npm（如果未安装）
    if ! command -v node &> /dev/null; then
        echo "  安装 Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
        apt install -y nodejs
    fi
    npm install -g pm2
    echo "✅ PM2 安装成功"
fi
echo ""

# 3. 检查 uv
echo "3. 检查 uv..."
if command -v uv &> /dev/null; then
    echo "✅ uv 已安装"
else
    echo "📦 正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ uv 安装成功"
    echo "⚠️  请运行以下命令将 uv 添加到 PATH:"
    echo "   source \$HOME/.local/bin/env"
fi
echo ""

# 4. 检查 lsof
echo "4. 检查 lsof..."
check_and_install "lsof" "lsof"
echo ""

# 5. 检查 pkill（通常系统自带）
echo "5. 检查 pkill..."
if command -v pkill &> /dev/null; then
    echo "✅ pkill 已安装"
else
    echo "📦 正在安装 procps（包含 pkill）..."
    apt install -y procps
    echo "✅ procps 安装成功"
fi
echo ""

# 6. 检查 autossh
echo "6. 检查 autossh..."
check_and_install "autossh" "autossh"
echo ""

# 7. 检查 sshpass（如果需要密码认证）
echo "7. 检查 sshpass..."
if command -v sshpass &> /dev/null; then
    echo "✅ sshpass 已安装"
else
    echo "📦 正在安装 sshpass..."
    apt install -y sshpass
    echo "✅ sshpass 安装成功"
fi
echo ""

# 8. 检查 curl（通常已安装，但确保存在）
echo "8. 检查 curl..."
check_and_install "curl" "curl"
echo ""

echo "=========================================="
echo "  ✅ 所有依赖检查完成！"
echo "=========================================="
echo ""
echo "📋 已安装的工具："
echo "  - docker: $(docker --version 2>/dev/null || echo '未安装')"
echo "  - pm2: $(pm2 --version 2>/dev/null || echo '未安装')"
echo "  - uv: $(uv --version 2>/dev/null || echo '未安装')"
echo "  - lsof: $(lsof -v 2>&1 | head -n1 || echo '已安装')"
echo "  - pkill: $(pkill --version 2>&1 | head -n1 || echo '已安装')"
echo "  - autossh: $(autossh -V 2>&1 | head -n1 || echo '已安装')"
echo "  - sshpass: $(sshpass -V 2>&1 | head -n1 || echo '已安装')"
echo ""
echo "🚀 现在可以运行启动脚本了："
echo "   bash start_livingKB_online.sh"
echo ""
