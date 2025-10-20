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
* 引入 **子图 PPR（Personalized PageRank）**：
  相比 HippoRAG2 在全图范围内的 PPR 计算，子图 PPR 实现了 **更精准的定位与更高的推理效率**

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

### 💻 安装

```bash
# 1. 克隆仓库
git clone https://github.com/DataArcTech/RAG-ARC.git
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
