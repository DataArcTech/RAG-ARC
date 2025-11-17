# RAG-ARC Docker 部署指南

> 💡 **两步部署：一次构建，随时启动**

## 🚀 快速开始

### 步骤1：构建Docker镜像

```bash
# 1. 克隆RAG-ARC项目
git clone https://github.com/DataArcTech/RAG-ARC.git
cd RAG-ARC

# 2. 构建所有Docker镜像
./build.sh
```

构建脚本将会：
- ✅ 检查Docker环境
- ✅ 创建.env配置文件
- ✅ 选择CPU或GPU模式（自动检测NVIDIA GPU）
- ✅ 拉取基础镜像（PostgreSQL、Redis、Neo4j）
- ✅ 构建RAG-ARC应用镜像

### 步骤2：启动所有服务

```bash
# 启动所有容器
./start.sh
```

启动脚本将会：
- ✅ 创建Docker网络
- ✅ 启动PostgreSQL 16数据库
- ✅ 启动Redis 7缓存
- ✅ 启动Neo4j图数据库
- ✅ 启动RAG-ARC应用
- ✅ 验证所有服务就绪

**注意**：默认情况下，Neo4j端口**不会暴露**到主机以提高安全性。如需启用Neo4j Browser访问，请在`.env`文件中设置`EXPOSE_NEO4J=true`后再运行`./start.sh`

## 🎯 部署架构

**全栈部署**包含4个容器：
- **PostgreSQL 16**：元数据和应用数据存储
- **Redis 7**：性能缓存层
- **Neo4j**：知识图谱数据库
- **RAG-ARC应用**：支持GPU的FastAPI应用（如果可用）

**优势**：
- 🚀 两步部署（一次构建，随时启动）
- 🔄 无需重新构建即可轻松重启
- 🎮 自动GPU检测和配置
- 📦 所有依赖容器化
- 🔒 隔离网络保证安全

## 📦 部署内容

部署创建4个容器：

1. **rag-arc-postgres** - PostgreSQL 16数据库
   - 存储元数据、用户数据和文件信息
   - 数据持久化到Docker卷 `rag-arc-postgres-data`
   - 网络：`rag-arc-network`

2. **rag-arc-redis** - Redis 7缓存
   - 缓存频繁访问的数据
   - 数据持久化到Docker卷 `rag-arc-redis-data`
   - 网络：`rag-arc-network`

3. **rag-arc-neo4j** - Neo4j图数据库
   - 存储知识图谱（实体、关系、事实）
   - 数据持久化到Docker卷 `rag-arc-neo4j-data` 和 `rag-arc-neo4j-logs`
   - 启用APOC插件用于高级图操作
   - 网络：`rag-arc-network`

4. **rag-arc-app** - RAG-ARC应用
   - 支持GPU的FastAPI应用（如果可用）
   - 通过Docker网络连接到PostgreSQL、Redis和Neo4j
   - 挂载卷：`./data`、`./local`、`./models`
   - 端口：8000（启动时可配置）

## 🔧 常用命令

### 查看日志
```bash
# 查看应用日志
docker logs -f rag-arc-app

# 查看PostgreSQL日志
docker logs -f rag-arc-postgres

# 查看Redis日志
docker logs -f rag-arc-redis

# 查看Neo4j日志
docker logs -f rag-arc-neo4j
```

### 管理容器
```bash
# 停止所有容器（保留数据）
./stop.sh

# 重启所有服务
./start.sh

# 清理Docker资源（保留本地数据）
./cleanup.sh
# 这会删除容器、卷和网络，但保留 ./data、./local、./models

# 完全清理（⚠️ 这将删除所有数据，包括本地目录！）
./clean-docker-data.sh
# 这会删除所有内容：容器、卷和本地数据目录
```

### 重新构建应用
```bash
# 重新构建应用镜像（代码更改后）
./build.sh

# 然后重启服务
./start.sh
```

## 🌐 访问地址

- **API服务**：http://localhost:8000
- **API文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/
- **Neo4j浏览器**：http://localhost:7474（仅当`.env`中设置`EXPOSE_NEO4J=true`时）
  - 用户名：`neo4j`
  - 密码：`12345678`（或`NEO4J_PASSWORD`设置的值）
  - 启用方法：在`.env`中添加`EXPOSE_NEO4J=true`并使用`./start.sh`重启

## ⚙️ 环境配置

### 必需配置

构建前编辑`.env`文件：

```bash
# LLM API配置（必需）
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Neo4j密码（建议修改）
NEO4J_PASSWORD=12345678  # 修改为安全的密码

# 本地运行模型配置
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5 # 或Qwen/Qwen3-Embedding-0.6B
DEVICE=cuda:0  # 或 cpu
```

### 脚本自动配置

以下配置由部署脚本自动配置：

```bash
# PostgreSQL配置
POSTGRES_HOST=rag-arc-postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123
POSTGRES_DB=rag_arc

# Redis配置
REDIS_HOST=rag-arc-redis
REDIS_PORT=6379

# Neo4j配置
NEO4J_URL=bolt://rag-arc-neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DATABASE=neo4j

# Neo4j端口暴露（可选，用于开发/调试）
EXPOSE_NEO4J=false          # 设置为true以访问Neo4j Browser
NEO4J_HTTP_PORT=7474        # Neo4j Browser端口（如果暴露）
NEO4J_BOLT_PORT=7687        # Bolt协议端口（如果暴露）
```

### 启用Neo4j Browser（可选）

如果需要访问Neo4j Browser进行调试或可视化：

1. 编辑`.env`文件：
```bash
EXPOSE_NEO4J=true
```

2. 重启服务：
```bash
./start.sh
```

3. 访问Neo4j Browser：http://localhost:7474
   - 用户名：`neo4j`
   - 密码：`12345678`

**注意**：生产环境建议保持`EXPOSE_NEO4J=false`以提高安全性。

## 📝 系统要求

### 最低要求
- **Docker**：20.10+
- **内存**：建议8GB+
- **磁盘空间**：20GB+（用于镜像、模型和数据）
- **操作系统**：Linux、macOS或带WSL2的Windows

### GPU模式（可选）
- **NVIDIA GPU**：支持CUDA的GPU
- **NVIDIA驱动**：525.60.13+（用于CUDA 12.1）
- **NVIDIA Docker运行时**：nvidia-docker2

### 检查GPU支持
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查NVIDIA Docker运行时
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🔍 故障排除

### 构建问题

**问题**：构建失败，超时错误
```bash
# 解决方案：增加Dockerfile中的超时时间
# 或使用不同的PyPI镜像
# 编辑Dockerfile并更改镜像URL
```

**问题**：GPU构建失败
```bash
# 解决方案：回退到CPU模式
# 运行./build.sh时选择选项1（CPU模式）
```

### 启动问题

**问题**：端口8000已被占用
```bash
# 解决方案：使用不同的端口
# start.sh脚本会提示您选择端口
# 或手动指定：docker run -p 8080:8000 ...
```

**问题**：PostgreSQL未就绪
```bash
# 检查PostgreSQL状态
docker exec rag-arc-postgres pg_isready -U postgres

# 查看PostgreSQL日志
docker logs rag-arc-postgres

# 重启PostgreSQL
docker restart rag-arc-postgres
```

**问题**：Neo4j未就绪
```bash
# 检查Neo4j状态
docker exec rag-arc-neo4j cypher-shell -u neo4j -p 12345678 "RETURN 1"

# 查看Neo4j日志
docker logs rag-arc-neo4j

# 重启Neo4j（可能需要1-2分钟启动）
docker restart rag-arc-neo4j
```

**问题**：应用未启动
```bash
# 检查应用日志
docker logs rag-arc-app

# 检查所有依赖是否就绪
docker ps --filter "name=rag-arc-"

# 重启应用
docker restart rag-arc-app
```

### 运行时问题

**问题**：内存不足错误
```bash
# 解决方案：增加Docker内存限制
# Docker Desktop：设置 > 资源 > 内存
# 或在配置中减少批处理大小
```

**问题**：推理速度慢
```bash
# 解决方案1：如果可用，使用GPU模式
./build.sh  # 选择GPU模式

# 解决方案2：在配置中减小模型大小
# 编辑 config/json_configs/rag_inference.json
```

### 数据问题

**问题**：需要重置所有数据
```bash
# 使用清理脚本
./cleanup.sh

# 然后重启服务
./start.sh
```

或者使用完全清理脚本（⚠️ 这会删除所有数据，包括本地目录！）：
```bash
./clean-docker-data.sh
# 然后重启服务
./start.sh
```

## 🔄 更新RAG-ARC

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 重新构建应用镜像
./build.sh

# 3. 重启服务
./start.sh
```

## 📚 更多信息

- [主文档](README.md) - 完整项目文档（英文）
- [中文文档](README-CN.md) - 中文项目文档
- [API文档](http://localhost:8000/docs) - 部署后可用

## 🆘 获取帮助

如果遇到问题：
1. 查看上面的[故障排除](#故障排除)部分
2. 查看容器日志：`docker logs rag-arc-app`
3. 在[GitHub](https://github.com/DataArcTech/RAG-ARC/issues)上提交issue

---

**祝您部署顺利！** 🎉

