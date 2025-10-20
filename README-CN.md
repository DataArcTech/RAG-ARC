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

🔗 **多跳推理与摘要**：解决需要多步推理的复杂问题<br>
📚 **知识密集型任务**：处理依赖大量结构化知识的问题<br>
🌐 **跨领域应用**：轻松支持学术论文、个人知识库、私域/企业知识库，只需最少的模式干预<br>

## 🏗️ 架构

<div align="center">
<img src="assets/architecture.png" alt="RAG-ARC 架构" width="95%"/><br>
RAG-ARC 系统架构概览
</div>

## 🔧 核心特性

RAG-ARC引入了多项关键创新，共同构建了一个精密集成的框架：

### 📁 多格式文档解析
- 支持docx、pdf、ppt、excel、html等多种文件类型
- 灵活的解析策略，支持OCR和布局感知的PDF解析（通过dots_ocr模块）
- 原生和基于VLM的OCR能力

### ✂️ 文本分块与向量化
- 多种分块策略（基于token、语义、递归、markdown标题）
- 集成HuggingFace嵌入模型进行向量表示
- 可配置的分块大小和重叠参数

### 🔍 多路径检索
- 结合BM25（稀疏检索）、Dense检索（Faiss-GPU）和Tantivy全文搜索
- 倒数排名融合（RRF）用于结果合并
- 可配置的权重和融合方法

### 🌐 图结构提取
- 基于事件的图提取能力
- 实体和关系提取用于知识图谱构建
- Neo4j图数据库集成

### 📈 重排序（Rerank）
- Qwen3模型用于精确结果重排序
- 基于LLM和列表式的重排序策略
- 分数归一化和元数据增强

### 🧩 模块化设计
- 工厂模式用于LLM、嵌入、检索器组件创建
- 分层架构：config、core、encapsulation、application、api
- 单例模式用于分词器管理和数据库连接

## 📊 性能表现

RAG-ARC在成本效率和准确性方面都带来了显著提升：

- **33.6% Token成本降低** 通过优化的检索和排序
- **16.62% 准确性提升** 通过多路径检索和融合排序
- **可扩展架构** 支持企业级部署

<div align="center">
<img src="assets/performance.png" alt="性能对比" width="90%"/>
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

### 💻 安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/RAG-ARC.git
cd RAG-ARC

# 2. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或者
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -e .

# 4. 复制并配置环境变量
cp .env.example .env
# 编辑.env以配置您的设置
```

### ⚙️ 配置

RAG-ARC使用模块化配置系统。关键配置文件位于`config/json_configs/`：

- `rag_inference.json`：RAG流水线配置
- `knowledge.json`：知识管理配置
- `account.json`：用户账户配置

### 🏃 运行服务

```bash
# 启动FastAPI服务器
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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