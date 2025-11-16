<div align="center">

# 🧠 RAG-ARC：检索增强生成架构

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-GPU/CPU-FF6F00.svg)](https://github.com/facebookresearch/faiss)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-ffa000.svg)](https://docs.pydantic.dev/)

*一个模块化、高性能的检索增强生成框架，支持多路径检索、图结构提取和融合排序*

[📘 English](README.md) • [⭐ 核心特性](#核心特性) • [🏗️ 架构](#架构) • [🚀 快速开始](#快速开始)

</div>

## 🎯 项目概述

**RAG-ARC** 是一个模块化的检索增强生成（RAG）框架，旨在构建高效、可扩展的架构，支持多路径检索、图结构提取和融合排序。该系统解决了传统RAG系统在处理非结构化文档（PDF、PPT、Excel等）时的关键挑战，如信息丢失、检索准确率低和多模态内容识别困难等问题。

### 🎯 核心应用场景

🧩 **RAG 全流程支持**：
覆盖从文档解析、文本分块、向量化，到多路径检索、图结构提取、结果重排序与知识图谱管理的完整流程，实现端到端的智能检索增强生成。

📚 **知识密集型任务**：
适用于依赖大量结构化与非结构化知识的问答、推理与内容生成场景，确保高召回率与语义一致性。

🌐 **多场景适配**：
同时支持 **标准 RAG** 与 **GraphRAG** 模式，可灵活应用于学术论文分析、个人知识库、企业知识库等多种领域，配置灵活、部署简便。

## 🏗️ 架构

<div align="center">
<img src="assets/architecture.png" alt="RAG-ARC 架构" width="95%"/><br>
RAG-ARC 系统架构概览
</div>

## 🔧 核心特性

**RAG-ARC** 融合了多项关键创新，构建出一个高效、可扩展、精密集成的检索增强生成（RAG）框架：

### 📁 多格式文档解析

* 支持多种文件类型：**DOCX、PDF、PPT、Excel、HTML** 等
* 灵活解析策略：支持 **OCR** 与 **布局感知的 PDF 解析**（基于 `dots_ocr` 模块）
* 同时具备 **原生 OCR** 与 **基于多模态大模型（VLM）的 OCR 能力**

### ✂️ 文本分块与向量化

* 支持多种分块策略：按 **Token**、**语义**、**递归规则** 或 **Markdown 层级**
* 集成 **HuggingFace 嵌入模型** 生成高质量向量表示
* 分块大小与重叠比例 **可灵活配置**

### 🔍 多路径检索

* 集成 **BM25（稀疏检索）**、**Dense（Faiss-GPU）** 与 **Tantivy 全文搜索**
* 采用 **倒数排名融合（RRF）** 实现多通道结果合并
* 支持 **自定义权重与融合策略**，实现检索优化

### 🌐 图结构提取

* 基于事实的 **实体与关系抽取**，构建结构化知识图谱
* 与 **Neo4j 图数据库** 无缝集成
* 支持面向问答与推理的知识图谱管理

### 🧠 GraphRAG

* 采用简洁高效的建图方式，**支持增量更新**，便于企业级部署
* 引入 **子图 PPR（Personalized PageRank）**：相比 HippoRAG2 在全图范围内的 PPR 计算，子图 PPR 实现了 **更精准的定位与更高的推理效率**

### 📈 重排序（Rerank）

* 基于 **Qwen3 模型** 的高精度结果重排序
* 支持 **LLM 驱动** 与 **列表式策略** 的双模式重排序
* 结合 **分数归一化与元数据增强** 提升排序鲁棒性

### 🧩 模块化架构

* **工厂模式** 管理 LLM、嵌入与检索器组件创建
* **分层设计**：`config`、`core`、`encapsulation`、`application`、`api`
* **单例模式** 管理分词器与数据库连接
* **共享机制** 支持检索器与嵌入模型实例复用，提高系统性能

## 📊 性能表现

基于 **HippoRAG2** 的架构演进，**RAG-ARC** 在成本效率与召回性能方面均实现了显著突破：

* 🚀 **Token 成本降低 22.9%**
  通过精心设计的提示词策略，在保持精度的同时有效减少 Token 消耗

* 🎯 **召回率提升 5.3%**
  借助剪枝优化，显著提高了文档检索的全面性与相关性

* 🔁 **支持知识图谱的增量更新**
  无需重新构图即可更新图谱，大幅降低计算与维护成本

<div align="center">
  <h3>📊 性能对比</h3>
  <img src="assets/accuracy_comparison.png" alt="Accuracy Comparison" width="80%" style="margin-bottom: 20px;"/><br>
  <img src="assets/recall_comparison.png" alt="Recall Comparison" width="80%" style="margin-bottom: 20px;"/><br>
  <img src="assets/token_cost_comparison.png" alt="Token Cost Comparison" width="80%"/>
</div>

## 📁 项目结构

```
RAG-ARC/
├── 📁 api/                       # API层（FastAPI路由/MCP集成）
│   ├── routers/                  # API路由定义
│   ├── config_examples/          # 配置示例
│   └── mcp/                      # MCP服务器实现
│
├── 📁 application/               # 业务逻辑层
│   ├── rag_inference/            # RAG推理模块
│   ├── knowledge/                # 知识管理
│   └── account/                  # 用户账户管理
│
├── 📁 core/                      # 核心能力
│   ├── file_management/          # 文件解析和分块
│   ├── retrieval/                # 检索策略
│   ├── rerank/                   # 重排序算法
│   ├── query_rewrite/            # 查询重写
│   └── prompts/                  # 提示模板
│
├── 📁 config/                    # 配置系统
│   ├── application/              # 应用配置
│   ├── core/                     # 核心模块配置
│   └── encapsulation/            # 封装配置
│
├── 📁 encapsulation/             # 封装层
│   ├── database/                 # 数据库接口
│   ├── llm/                      # LLM接口
│   └── data_model/               # 数据模型和模式
│
├── 📁 framework/                 # 框架核心
│   ├── module.py                 # 基础模块类
│   ├── register.py               # 组件注册表
│   └── config.py                 # 配置系统
│
├── 📁 test/                      # 测试套件
│
├── main.py                      # 🎯 主应用程序入口点
├── app_registration.py          # 组件初始化
├── pyproject.toml               # 项目依赖
└── README.md                    # 项目文档
```

## 🚀 快速开始

### 🐳 Docker部署（推荐）

**三步部署：**

```bash
# 1. 克隆仓库
git clone https://github.com/DataArcTech/RAG-ARC.git
cd RAG-ARC

# 2. 构建Docker镜像（一次性设置）
./build.sh

# 3. 启动所有服务
./start.sh
```

部署包含以下服务：
- ✅ **PostgreSQL 16**：元数据存储
- ✅ **Redis 7**：缓存层
- ✅ **Neo4j**：知识图谱数据库
- ✅ **RAG-ARC应用**：支持GPU的FastAPI应用

**脚本功能说明：**

`build.sh`：
- 检查Docker环境
- 创建.env配置文件
- 选择CPU/GPU模式（自动检测NVIDIA GPU）
- 拉取基础镜像（PostgreSQL、Redis、Neo4j）
- 构建RAG-ARC应用镜像

`start.sh`：
- 创建Docker网络
- 启动全部4个容器
- 等待服务就绪
- 验证部署状态

`stop.sh`：
- 停止所有运行中的容器（保留数据）

`cleanup.sh`：
- 删除所有容器和Docker卷
- 删除Docker网络
- **保留本地数据目录**（`./data`、`./local`、`./models`）
- 适用于清理Docker资源但保留数据的情况

`clean-docker-data.sh`：
- 删除所有容器和Docker卷
- **同时删除本地数据目录**（`./data/postgresql`、`./data/neo4j`、`./data/redis`、`./data/graph_index_neo4j`）
- 适用于需要完全清理的情况（⚠️ **这将删除所有数据！**）

**访问服务：**
- API服务：http://localhost:8000
- API文档：http://localhost:8000/docs

📖 **详细说明和故障排除请参见 [Docker部署指南（中文）](README.Docker-CN.md) 或 [Docker Deployment Guide (English)](README.Docker.md)**

### 💻 本地安装

```bash
# 1. 克隆仓库
git clone https://github.com/DataArcTech/RAG-ARC.git
cd RAG-ARC

# 2. 安装uv（如果尚未安装）
# 推荐：使用国内镜像（国内更快）
curl -LsSf https://astral.ac.cn/uv/install.sh | sh
# 备选：使用官方安装器
# curl -LsSf https://astral.sh/uv/install.sh | sh
# 或添加到PATH：export PATH="$HOME/.local/bin:$PATH"

# 3. 安装依赖（uv会自动创建虚拟环境）
uv sync

# 4. 复制并配置环境变量
cp .env.example .env
# 编辑.env以配置您的设置
```

### ⚙️ 配置

RAG-ARC使用模块化配置系统。关键配置文件位于`config/json_configs/`,在这里，你可以控制选择每个模型使用的显卡，业务流程中使用的模型等不同的参数：

- `rag_inference.json`：RAG检索配置
- `knowledge.json`：知识管理配置
- `account.json`：用户账户配置

### 🏃 运行服务

```bash
# 启动FastAPI服务器（uv run会自动管理虚拟环境）
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 🧪 使用示例

```bash
# 上传文档
curl -X POST "http://localhost:8000/knowledge" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@/path/to/your/document.pdf"

# 与RAG系统对话
curl -X POST "http://localhost:8000/rag_inference/chat" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG-ARC?"}'

# 获取Token（登录）
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=YOUR_USERNAME&password=YOUR_PASSWORD"

# 注册新用户
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"name": "新用户", "user_name": "YOUR_USERNAME", "password": "YOUR_PASSWORD"}'

# 创建新对话会话
curl -X POST "http://localhost:8000/session" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# 列出会话内消息
curl -X GET "http://localhost:8000/session/YOUR_SESSION_ID/messages" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### WebSocket流式对话（Python示例，需先安装websockets库）：

```python
import asyncio
import websockets

async def chat():
    uri = 'ws://localhost:8000/rag_inference/stream_chat/YOUR_SESSION_ID'
    async with websockets.connect(uri, additional_headers=[('Cookie', 'auth_token=YOUR_ACCESS_TOKEN')]) as ws:
        await ws.send('你好，RAG-ARC!')
        print(await ws.recv())

asyncio.run(chat())
```

## 🛠️ 技术栈

- **后端**：Python 3.11+
- **框架**：FastAPI
- **向量数据库**：FAISS（GPU/CPU）
- **图数据库**：Neo4j
- **全文搜索**：Tantivy
- **机器学习框架**：HuggingFace Transformers、PyTorch
- **数据验证**：Pydantic v2
- **序列化**：Dill
- **LLM支持**：Qwen3、OpenAI API、HuggingFace模型

## 🔧 高级配置

### 多路径检索配置

RAG-ARC 支持可配置的多路径检索，包含以下组件：

1. **密集检索**：使用 FAISS 进行向量相似度搜索
2. **稀疏检索**：通过 Tantivy 实现的 BM25
3. **图检索**：基于 Neo4j 的知识图谱检索与 Pruned HippoRAG

融合方法可配置为使用：
- **倒数排名融合（RRF）**：默认的结果合并方法
- **加权求和**：为每个检索路径自定义权重
- **排名融合**：基于排名的组合方法

### GraphRAG 实现

RAG-ARC 实现了基于 HippoRAG2 的增强 GraphRAG 方法，具有以下关键改进：

1. **子图PPR**：RAG-ARC 在相关子图上计算个性化PageRank，而不是在整个图上计算，以获得更好的效率和准确性
2. **查询感知剪枝**：根据实体与查询的相关性动态调整图扩展期间保留的邻居数量
3. **增量更新**：支持在不完全重建的情况下更新知识图谱

### 文档处理管道

文档处理管道包含几个阶段：

1. **文件存储**：文档存储在可配置的存储后端（本地文件系统或云存储）
2. **解析**：多个解析器支持不同类型的文档：
   - 标准格式的原生解析器（PDF、DOCX、PPTX等）
   - 扫描文档的OCR解析器（使用DOTS-OCR或基于VLM的方法）
3. **分块**：文本使用可配置的策略分割成块：
   - 基于Token的分块
   - 语义分块
   - 递归分块
   - 基于Markdown标题的分块
4. **索引**：块在多个系统中建立索引：
   - FAISS用于密集检索
   - Tantivy用于稀疏检索
   - Neo4j用于基于图的检索

## 📊 API 端点

RAG-ARC 提供了全面的 REST API，包含以下关键端点：

### 知识管理
- `POST /knowledge`：上传文档
- `GET /knowledge/list_files`：列出用户文档
- `GET /knowledge/{doc_id}/download`：下载文档
- `DELETE /knowledge/{doc_id}`：删除文档

### RAG 推理
- `POST /rag_inference/chat`：与 RAG 系统对话
- `WebSocket /rag_inference/stream_chat/{session_id}`：基于 WebSocket 的流式对话

### 用户管理
- `POST /auth/register`：用户注册
- `POST /auth/token`：用户认证（登录）

### 会话管理
- `POST /session`：创建聊天会话
- `GET /session`：列出用户会话
- `GET /session/{session_id}`：获取会话详情
- `DELETE /session/{session_id}`：删除会话

## 🔒 安全与认证

RAG-ARC 实现了基于 JWT 的认证机制，具有以下功能：

- 用户注册和登录
- 基于角色的访问控制
- 文档级别的权限控制（VIEW/EDIT）
- 使用 bcrypt 进行安全密码哈希
- 令牌刷新机制

## 📈 监控与可观察性

RAG-ARC 包含内置的监控功能：

- 可配置级别的日志记录
- 性能指标收集
- 健康检查端点
- 索引状态监控

## 🤝 贡献

我们欢迎社区的贡献！您可以这样参与：

### 💻 代码贡献

1. 🍴 Fork仓库
2. 🌿 创建功能分支（`git checkout -b feature/AmazingFeature`）
3. 💾 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 📤 推送到分支（`git push origin feature/AmazingFeature`）
5. 🔄 打开Pull Request

### 🔧 开发指南

- **新解析策略**：实现自定义文档解析逻辑
- **检索算法**：添加新的检索方法和融合技术
- **重排序模型**：集成额外的重排序模型
- **分块方法**：实现新颖的文本分块方法

## 📞 联系方式

如有问题、建议或反馈，请在GitHub上提交issue或联系维护者。

---

## 📚 许可证

本项目采用MIT许可证 - 详情请见[LICENSE](LICENSE)文件。