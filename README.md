<div align="center">

# 🧠 RAG-ARC: Retrieval-Augmented Generation Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-GPU/CPU-FF6F00.svg)](https://github.com/facebookresearch/faiss)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-ffa000.svg)](https://docs.pydantic.dev/)

*A modular, high-performance Retrieval-Augmented Generation framework with multi-path retrieval, graph extraction, and fusion ranking*

[📘 中文文档](README-CN.md) • [⭐ Key Features](#key-features) • [🏗️ Architecture](#architecture) • [🚀 Quick Start](#quick-start)

</div>

## 🎯 Project Overview

**RAG-ARC** is a modular Retrieval-Augmented Generation (RAG) framework designed to build efficient, scalable architectures that support multi-path retrieval, graph structure extraction, and fusion ranking. The system addresses key challenges in traditional RAG systems when processing unstructured documents (PDF, PPT, Excel, etc.) such as information loss, low retrieval accuracy, and difficulty in recognizing multimodal content.

### 🎯 Core Use Cases

🧩 **Full RAG Pipeline Support**: Covers the complete pipeline—from document parsing, text chunking, and embedding generation to multi-path retrieval, graph extraction, reranking, and knowledge graph management.<br>
📚 **Knowledge-Intensive Tasks**: Ideal for question answering, reasoning, and content generation tasks that rely on large-scale structured and unstructured knowledge, ensuring high recall and semantic consistency.<br>
🌐 **Cross-Domain Applications**: Supports both Standard RAG and GraphRAG modes, making it adaptable for academic research, personal knowledge bases, and enterprise-level knowledge management systems.<br>

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.png" alt="RAG-ARC Architecture" width="95%"/><br>
RAG-ARC System Architecture Overview
</div>

## 🔧 Key Features

RAG-ARC introduces several key innovations that together build a sophisticated integrated framework:

### 📁 Multi-Format Document Parsing
- Support for docx, pdf, ppt, excel, html and other file types
- Flexible parsing strategies with OCR and layout-aware PDF parsing (via dots_ocr module)
- Native and VLM-based OCR capabilities

### ✂️ Text Chunking & Vectorization
- Multiple chunking strategies (token-based, semantic, recursive, markdown headers)
- Integration with HuggingFace embedding models for vector representation
- Configurable chunk size and overlap parameters

### 🔍 Multi-Path Retrieval
- Combined BM25 (sparse retrieval), Dense retrieval (Faiss-GPU), and Tantivy full-text search
- Reciprocal Rank Fusion (RRF) for result merging
- Configurable weights and fusion methods

### 🌐 Graph Structure Extraction
- Extracts entities and relations from facts to build structured knowledge graphs
- Seamlessly integrates with Neo4j graph database
- Enables knowledge-graph-driven reasoning and QA

### 🧠 GraphRAG
- Lightweight, incrementally updatable graph construction suitable for enterprise deployment
- Incorporates Subgraph PPR (Personalized PageRank):
Compared to HippoRAG2’s full-graph PPR, subgraph PPR achieves higher reasoning precision and efficiency

### 📈 Re-ranking (Rerank)
- Qwen3 model for precise result re-ranking
- LLM-based and listwise re-ranking strategies
- Score normalization and metadata enrichment

### 🧩 Modular Design
- Factory pattern for LLM, Embedding, Retriever component creation
- Layered architecture: config, core, encapsulation, application, api
- Singleton pattern for tokenizer management and database connections

## 📊 Performance

Built upon the HippoRAG2 evolution, RAG-ARC delivers significant improvements in both efficiency and recall performance:

- 🚀 22.9% Token Cost Reduction
Through optimized prompt strategies, it reduces token consumption without sacrificing accuracy.
- 🎯 5.3% Recall Rate Increase
Pruning-based optimizations yield more comprehensive and relevant retrieval.
- 🔁 Incremental Knowledge Graph Updates
Supports updating graph data without full reconstruction—reducing computational and maintenance overhead.

<div align="center">
  <h3>📊 Performance Comparison</h3>
  <img src="assets/accuracy_comparison.png" alt="Accuracy Comparison" width="80%" style="margin-bottom: 20px;"/><br>
  <img src="assets/recall_comparison.png" alt="Recall Comparison" width="80%" style="margin-bottom: 20px;"/><br>
  <img src="assets/token_cost_comparison.png" alt="Token Cost Comparison" width="80%"/>
</div>


## 📁 Project Structure

```
RAG-ARC/
├── 📁 api/                       # API layer (FastAPI routes/MCP integration)
│   ├── routers/                  # API route definitions
│   ├── config_examples/          # Configuration examples
│   └── mcp/                      # MCP server implementation
│
├── 📁 application/               # Business logic layer
│   ├── rag_inference/            # RAG inference module
│   ├── knowledge/                # Knowledge management
│   └── account/                  # User account management
│
├── 📁 core/                      # Core capabilities
│   ├── file_management/          # File parsing and chunking
│   ├── retrieval/                # Retrieval strategies
│   ├── rerank/                   # Re-ranking algorithms
│   ├── query_rewrite/            # Query rewriting
│   └── prompts/                  # Prompt templates
│
├── 📁 config/                    # Configuration system
│   ├── application/              # Application configs
│   ├── core/                     # Core module configs
│   └── encapsulation/            # Encapsulation configs
│
├── 📁 encapsulation/             # Encapsulation layer
│   ├── database/                 # Database interfaces
│   ├── llm/                      # LLM interfaces
│   └── data_model/               # Data models and schemas
│
├── 📁 framework/                 # Framework core
│   ├── module.py                 # Base module class
│   ├── register.py               # Component registry
│   └── config.py                 # Configuration system
│
├── 📁 test/                      # Test suite
│
├── main.py                      # 🎯 Main application entry point
├── app_registration.py          # Component initialization
├── pyproject.toml               # Project dependencies
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

**Two-step deployment:**

```bash
# 1. Clone the repository
git clone https://github.com/DataArcTech/RAG-ARC.git
cd RAG-ARC

# 2. Build Docker images (one-time setup)
./build.sh

# 3. Start all services
./start.sh
```

The deployment includes:
- ✅ **PostgreSQL 16**: Metadata storage
- ✅ **Redis 7**: Caching layer
- ✅ **Neo4j**: Knowledge graph database
- ✅ **RAG-ARC App**: FastAPI application with GPU support

**What the scripts do:**

`build.sh`:
- Checks Docker environment
- Creates .env configuration
- Selects CPU/GPU mode (auto-detect NVIDIA GPU)
- Pulls base images (PostgreSQL, Redis, Neo4j)
- Builds RAG-ARC application image

`start.sh`:
- Creates Docker network
- Starts all 4 containers
- Waits for services to be ready
- Verifies deployment

**Access the service:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

📖 **See [Docker Deployment Guide](README.Docker.md) for detailed instructions and troubleshooting**

### 💻 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/RAG-ARC.git
cd RAG-ARC

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -e .

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env to configure your settings
```

### ⚙️ Configuration

RAG-ARC uses a modular configuration system. Key configuration files are located in `config/json_configs/`:

- `rag_inference.json`: RAG pipeline configuration
- `knowledge.json`: Knowledge management configuration
- `account.json`: User account configuration

### 🏃 Running the Service

```bash
# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 🧪 Example Usage

```bash
# Upload a document
curl -X POST "http://localhost:8000/knowledge" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@/path/to/your/document.pdf"

# Chat with the RAG system
curl -X POST "http://localhost:8000/rag_inference/chat" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG-ARC?"}'
```

## 🛠️ Technology Stack

- **Backend**: Python 3.11+
- **Framework**: FastAPI
- **Vector Database**: FAISS (GPU/CPU)
- **Graph Database**: Neo4j
- **Full-text Search**: Tantivy
- **ML Frameworks**: HuggingFace Transformers, PyTorch
- **Data Validation**: Pydantic v2
- **Serialization**: Dill
- **LLM Support**: Qwen3, OpenAI API, HuggingFace models

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 💻 Code Contributions

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

### 🔧 Development Guidelines

- **New Parsing Strategies**: Implement custom document parsing logic
- **Retrieval Algorithms**: Add new retrieval methods and fusion techniques
- **Reranking Models**: Integrate additional reranking models
- **Chunking Methods**: Implement novel text chunking approaches

## 📞 Contact

For questions, issues, or feedback, please open an issue on GitHub or contact the maintainers.

---

## 📚 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
