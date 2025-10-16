
import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encapsulation.data_model.schema import Chunk
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig


def create_test_documents_for_users():
    """为3个用户创建测试文档"""
    user1_id = str(uuid.uuid4())
    user2_id = str(uuid.uuid4())
    user3_id = str(uuid.uuid4())
    
    chunks = []
    
    # User 1: Python 相关文档 (10个)
    python_docs = [
        "Python is a high-level programming language with dynamic typing and garbage collection.",
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "Python's standard library is extensive and includes modules for file I/O, system calls, and networking.",
        "Python decorators are a powerful feature that allows modifying function behavior without changing code.",
        "Python list comprehensions provide a concise way to create lists based on existing lists.",
        "Python generators are memory-efficient iterators that yield values one at a time.",
        "Python context managers using 'with' statement ensure proper resource management.",
        "Python asyncio enables writing concurrent code using async/await syntax.",
        "Python type hints improve code readability and enable static type checking.",
        "Python virtual environments isolate project dependencies from system packages."
    ]
    
    for i, content in enumerate(python_docs):
        chunks.append(Chunk(
            id=f"user1_python_{i}",
            content=content,
            owner_id=user1_id,
            metadata={"topic": "python", "user": "user1", "index": i}
        ))
    
    # User 2: Java 相关文档 (10个)
    java_docs = [
        "Java is a class-based, object-oriented programming language with platform independence.",
        "Java Virtual Machine (JVM) enables Java programs to run on any platform without recompilation.",
        "Java interfaces define contracts that classes must implement, supporting multiple inheritance.",
        "Java generics provide type safety and eliminate the need for type casting.",
        "Java streams API enables functional-style operations on collections of elements.",
        "Java lambda expressions provide a concise way to represent anonymous functions.",
        "Java garbage collector automatically manages memory allocation and deallocation.",
        "Java multithreading allows concurrent execution of multiple threads within a program.",
        "Java annotations provide metadata about program elements at compile time or runtime.",
        "Java modules introduced in Java 9 improve encapsulation and dependency management."
    ]
    
    for i, content in enumerate(java_docs):
        chunks.append(Chunk(
            id=f"user2_java_{i}",
            content=content,
            owner_id=user2_id,
            metadata={"topic": "java", "user": "user2", "index": i}
        ))
    
    # User 3: JavaScript 相关文档 (10个)
    js_docs = [
        "JavaScript is a dynamic, weakly-typed programming language primarily used for web development.",
        "JavaScript closures allow functions to access variables from their outer scope.",
        "JavaScript promises represent the eventual completion or failure of asynchronous operations.",
        "JavaScript async/await syntax simplifies working with promises and asynchronous code.",
        "JavaScript prototypal inheritance differs from classical inheritance in other languages.",
        "JavaScript event loop handles asynchronous callbacks and maintains non-blocking execution.",
        "JavaScript arrow functions provide a shorter syntax and lexically bind 'this' value.",
        "JavaScript destructuring assignment extracts values from arrays or properties from objects.",
        "JavaScript modules using import/export enable code organization and reusability.",
        "JavaScript spread operator expands iterables into individual elements."
    ]
    
    for i, content in enumerate(js_docs):
        chunks.append(Chunk(
            id=f"user3_js_{i}",
            content=content,
            owner_id=user3_id,
            metadata={"topic": "javascript", "user": "user3", "index": i}
        ))
    
    return chunks, user1_id, user2_id, user3_id


def test_user_isolation_with_dense_retriever():
    """测试 Dense Retriever 的用户隔离和 over-fetching"""
    print("\n" + "="*80)
    print("Test 1: Dense Retriever with User Isolation & Over-fetching")
    print("="*80)
    
    # 创建测试数据
    chunks, user1_id, user2_id, user3_id = create_test_documents_for_users()
    print(f"\n📊 Created {len(chunks)} test documents:")
    print(f"  - User 1 ({user1_id[:8]}...): 10 Python docs")
    print(f"  - User 2 ({user2_id[:8]}...): 10 Java docs")
    print(f"  - User 3 ({user3_id[:8]}...): 10 JavaScript docs")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_user_isolation_")
    faiss_index_path = os.path.join(temp_dir, "faiss_index")
    
    try:
        # 配置 Embedding (使用 Qwen 默认配置)
        print(f"\n🔧 Configuring embedding model...")
        embedding_config = QwenEmbeddingConfig(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="cuda:0",
            use_china_mirror=True,
            cache_folder="./models/Qwen"
        )
        
        # 配置 FAISS
        print(f"🔧 Configuring FAISS vector database...")
        faiss_config = FaissVectorDBConfig(
            embedding_config=embedding_config,
            index_type="flat",
            metric="cosine",
            index_path=faiss_index_path
        )
        
        # 构建索引
        print(f"🔨 Building FAISS index...")
        faiss_db = faiss_config.build()
        faiss_db.build_index(chunks)
        print(f"✓ Index built with {faiss_db.index.ntotal} vectors")
        
        # 保存索引
        faiss_db.save_index(faiss_index_path)
        print(f"✓ Index saved to {faiss_index_path}")
        
        # 配置 Dense Retriever
        print(f"\n🔧 Configuring Dense Retriever...")
        retriever_config = DenseRetrieverConfig(
            index_config=faiss_config,
            search_kwargs={"k": 5}
        )

        retriever = retriever_config.build()
        retriever._index = faiss_db
        
        # 测试场景1: User 1 查询 Python 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 1: User 1 searches for Python content")
        print("-"*80)
        
        query = "programming language features and syntax"
        print(f"Query: '{query}'")
        print(f"User: User 1 ({user1_id[:8]}...)")
        
        # 不使用 over-fetching
        print(f"\n  [Without over-fetching]")
        results_no_overfetch = retriever.similarity_search(
            query,
            owner_id=user1_id,
            k=5,
            over_fetch_multiplier=1
        )
        print(f"  Retrieved: {len(results_no_overfetch)} / 5 chunks")
        for i, chunk in enumerate(results_no_overfetch[:3]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")
        
        # 使用 over-fetching (3x)
        print(f"\n  [With over-fetching (3x)]")
        results_with_overfetch = retriever.similarity_search(
            query,
            owner_id=user1_id,
            k=5,
            over_fetch_multiplier=3
        )
        print(f"  Retrieved: {len(results_with_overfetch)} / 5 chunks")
        for i, chunk in enumerate(results_with_overfetch[:5]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")
        
        # 验证所有结果都属于 User 1
        assert all(chunk.owner_id == user1_id for chunk in results_with_overfetch), \
            "All results should belong to User 1"
        print(f"  ✓ All results belong to User 1")
        
        # 测试场景2: User 2 查询 Java 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 2: User 2 searches for Java content")
        print("-"*80)
        
        query = "object oriented programming concepts"
        print(f"Query: '{query}'")
        print(f"User: User 2 ({user2_id[:8]}...)")
        
        results_user2 = retriever.similarity_search(
            query,
            owner_id=user2_id,
            k=5,
            over_fetch_multiplier=3
        )
        print(f"  Retrieved: {len(results_user2)} / 5 chunks")
        for i, chunk in enumerate(results_user2[:5]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")
        
        assert all(chunk.owner_id == user2_id for chunk in results_user2), \
            "All results should belong to User 2"
        print(f"  ✓ All results belong to User 2")
        
        # 测试场景3: User 3 查询 JavaScript 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 3: User 3 searches for JavaScript content")
        print("-"*80)
        
        query = "asynchronous programming and functions"
        print(f"Query: '{query}'")
        print(f"User: User 3 ({user3_id[:8]}...)")
        
        results_user3 = retriever.similarity_search(
            query,
            owner_id=user3_id,
            k=5,
            over_fetch_multiplier=3
        )
        print(f"  Retrieved: {len(results_user3)} / 5 chunks")
        for i, chunk in enumerate(results_user3[:5]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")
        
        assert all(chunk.owner_id == user3_id for chunk in results_user3), \
            "All results should belong to User 3"
        print(f"  ✓ All results belong to User 3")
        
        # 测试场景4: 无 owner_id 过滤 (返回所有用户的结果)
        print(f"\n" + "-"*80)
        print("Scenario 4: Search without owner_id filter (all users)")
        print("-"*80)
        
        query = "programming language"
        print(f"Query: '{query}'")
        print(f"User: None (no filter)")
        
        results_all = retriever.similarity_search(query, k=10)
        print(f"  Retrieved: {len(results_all)} chunks from all users")
        
        # 统计每个用户的文档数量
        user_counts = {}
        for chunk in results_all:
            user_counts[chunk.owner_id] = user_counts.get(chunk.owner_id, 0) + 1
        
        for owner_id, count in user_counts.items():
            user_label = "User 1" if owner_id == user1_id else ("User 2" if owner_id == user2_id else "User 3")
            print(f"    {user_label}: {count} chunks")
        
        print(f"\n✅ Dense Retriever test passed!")
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


def test_user_isolation_with_bm25_retriever():
    """测试 BM25 Retriever 的用户隔离"""
    # 创建测试数据
    chunks, user1_id, user2_id, user3_id = create_test_documents_for_users()
    print(f"\n📊 Created {len(chunks)} test documents:")
    print(f"  - User 1 ({user1_id[:8]}...): 10 Python docs")
    print(f"  - User 2 ({user2_id[:8]}...): 10 Java docs")
    print(f"  - User 3 ({user3_id[:8]}...): 10 JavaScript docs")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_bm25_isolation_")
    bm25_index_path = os.path.join(temp_dir, "bm25_index")

    try:
        # 配置 BM25
        print(f"\n🔧 Configuring BM25 indexer...")
        bm25_config = BM25BuilderConfig(
            index_path=bm25_index_path,
            language="english"
        )

        # 构建索引
        print(f"🔨 Building BM25 index...")
        bm25_indexer = bm25_config.build()
        bm25_indexer.build_index(chunks)
        print(f"✓ Index built with {len(chunks)} documents")

        # 保存索引
        bm25_indexer.save_index(bm25_index_path)
        print(f"✓ Index saved to {bm25_index_path}")

        # 配置 BM25 Retriever
        print(f"\n🔧 Configuring BM25 Retriever...")
        retriever_config = TantivyBM25RetrieverConfig(
            index_config=bm25_config,
            search_kwargs={"k": 5}
        )

        retriever = retriever_config.build()

        # 测试场景1: User 1 查询 Python 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 1: User 1 searches for Python content")
        print("-"*80)

        query = "programming language"
        print(f"Query: '{query}'")
        print(f"User: User 1 ({user1_id[:8]}...)")

        results_user1 = retriever._get_relevant_chunks(
            query,
            k=5,
            filters={"owner_id": user1_id}
        )
        print(f"  Retrieved: {len(results_user1)} / 5 chunks")
        for i, chunk in enumerate(results_user1[:5]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")

        # 验证所有结果都属于 User 1
        assert all(chunk.owner_id == user1_id for chunk in results_user1), \
            "All results should belong to User 1"
        print(f"  ✓ All results belong to User 1")

        # 测试场景2: User 2 查询 Java 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 2: User 2 searches for Java content")
        print("-"*80)

        query = "object oriented"
        print(f"Query: '{query}'")
        print(f"User: User 2 ({user2_id[:8]}...)")

        results_user2 = retriever._get_relevant_chunks(
            query,
            k=5,
            filters={"owner_id": user2_id}
        )
        print(f"  Retrieved: {len(results_user2)} / 5 chunks")
        for i, chunk in enumerate(results_user2[:5]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")

        assert all(chunk.owner_id == user2_id for chunk in results_user2), \
            "All results should belong to User 2"
        print(f"  ✓ All results belong to User 2")

        # 测试场景3: User 3 查询 JavaScript 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 3: User 3 searches for JavaScript content")
        print("-"*80)

        query = "asynchronous"
        print(f"Query: '{query}'")
        print(f"User: User 3 ({user3_id[:8]}...)")

        results_user3 = retriever._get_relevant_chunks(
            query,
            k=5,
            filters={"owner_id": user3_id}
        )
        print(f"  Retrieved: {len(results_user3)} / 5 chunks")
        for i, chunk in enumerate(results_user3[:5]):
            print(f"    {i+1}. {chunk.id} - {chunk.content[:60]}...")

        assert all(chunk.owner_id == user3_id for chunk in results_user3), \
            "All results should belong to User 3"
        print(f"  ✓ All results belong to User 3")

        # 测试场景4: 无 owner_id 过滤
        print(f"\n" + "-"*80)
        print("Scenario 4: Search without owner_id filter (all users)")
        print("-"*80)

        query = "programming"
        print(f"Query: '{query}'")
        print(f"User: None (no filter)")

        results_all = retriever._get_relevant_chunks(query, k=10)
        print(f"  Retrieved: {len(results_all)} chunks from all users")

        # 统计每个用户的文档数量
        user_counts = {}
        for chunk in results_all:
            user_counts[chunk.owner_id] = user_counts.get(chunk.owner_id, 0) + 1

        for owner_id, count in user_counts.items():
            user_label = "User 1" if owner_id == user1_id else ("User 2" if owner_id == user2_id else "User 3")
            print(f"    {user_label}: {count} chunks")

        print(f"\n✅ BM25 Retriever test passed!")

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


def test_user_isolation_with_multipath_retriever():
    """Test 3: MultiPath Retriever with User Isolation (Dense + BM25 Fusion)"""
    print("\n" + "="*80)
    print("Test 3: MultiPath Retriever with User Isolation (Dense + BM25 Fusion)")
    print("="*80)

    temp_dir = tempfile.mkdtemp(prefix="test_multipath_isolation_")

    try:
        # 创建测试数据
        user1_id = str(uuid.uuid4())
        user2_id = str(uuid.uuid4())
        user3_id = str(uuid.uuid4())

        chunks = []

        # User 1: Python 文档
        python_docs = [
            "Python is a high-level programming language with dynamic typing",
            "Python supports multiple programming paradigms including procedural and object-oriented",
            "Python list comprehensions provide a concise way to create lists",
            "Python decorators are a powerful feature that allows modifying functions",
            "Python generators enable lazy evaluation and memory-efficient iteration",
            "Python context managers handle resource management with 'with' statement",
            "Python asyncio enables writing concurrent code using async/await syntax",
            "Python type hints improve code readability and enable static analysis",
            "Python dataclasses reduce boilerplate code for data-holding classes",
            "Python f-strings provide an elegant way to format strings"
        ]

        for i, doc in enumerate(python_docs):
            chunks.append(Chunk(
                id=f"user1_python_{i}",
                content=doc,
                owner_id=user1_id,
                metadata={"source": "python_docs", "user": "user1"}
            ))

        # User 2: Java 文档
        java_docs = [
            "Java is a class-based, object-oriented programming language designed for portability",
            "Java virtual machine enables write once, run anywhere capability",
            "Java interfaces define contracts that classes must implement",
            "Java generics provide compile-time type safety for collections",
            "Java streams API enables functional-style operations on collections",
            "Java lambda expressions simplify functional programming in Java",
            "Java Optional class helps avoid null pointer exceptions",
            "Java CompletableFuture enables asynchronous programming",
            "Java annotations provide metadata about program elements at compile time",
            "Java modules introduced in Java 9 improve encapsulation and dependency management"
        ]

        for i, doc in enumerate(java_docs):
            chunks.append(Chunk(
                id=f"user2_java_{i}",
                content=doc,
                owner_id=user2_id,
                metadata={"source": "java_docs", "user": "user2"}
            ))

        # User 3: JavaScript 文档
        js_docs = [
            "JavaScript is a dynamic, prototype-based scripting language for web development",
            "JavaScript closures allow functions to access variables from outer scope",
            "JavaScript promises represent the eventual completion or failure of async operations",
            "JavaScript async/await syntax simplifies working with promises",
            "JavaScript destructuring assignment extracts values from arrays or objects",
            "JavaScript event loop handles asynchronous callbacks and maintains execution order",
            "JavaScript arrow functions provide a shorter syntax and lexical this binding",
            "JavaScript spread operator enables array and object manipulation",
            "JavaScript modules enable code organization and reusability",
            "JavaScript Map and Set provide efficient data structures for collections"
        ]

        for i, doc in enumerate(js_docs):
            chunks.append(Chunk(
                id=f"user3_js_{i}",
                content=doc,
                owner_id=user3_id,
                metadata={"source": "js_docs", "user": "user3"}
            ))

        print(f"\n📊 Created {len(chunks)} test documents:")
        print(f"  - User 1 ({user1_id[:8]}...): {len(python_docs)} Python docs")
        print(f"  - User 2 ({user2_id[:8]}...): {len(java_docs)} Java docs")
        print(f"  - User 3 ({user3_id[:8]}...): {len(js_docs)} JavaScript docs")

        # 配置 Embedding 模型
        print("\n🔧 Configuring embedding model...")
        embedding_config = QwenEmbeddingConfig(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="cuda:0",
            use_china_mirror=True,
            cache_folder="./models/Qwen"
        )

        # 配置 FAISS 向量数据库
        print("🔧 Configuring FAISS vector database...")
        faiss_index_path = os.path.join(temp_dir, "faiss_index")
        faiss_config = FaissVectorDBConfig(
            index_path=faiss_index_path,
            metric="cosine",
            index_type="flat",
            normalize_L2=True,
            embedding_config=embedding_config
        )

        # 构建 FAISS 索引
        print("🔨 Building FAISS index...")
        faiss_db = faiss_config.build()
        faiss_db._add_chunks(chunks)
        print(f"✓ FAISS index built with {len(chunks)} vectors")
        print(f"✓ FAISS index saved to {faiss_index_path}")

        # 配置 BM25 索引
        print("\n🔧 Configuring BM25 indexer...")
        bm25_index_path = os.path.join(temp_dir, "bm25_index")
        bm25_config = BM25BuilderConfig(
            index_path=bm25_index_path
        )

        # 构建 BM25 索引
        print("🔨 Building BM25 index...")
        bm25_indexer = bm25_config.build()
        bm25_indexer.add_chunks(chunks)
        print(f"✓ BM25 index built with {len(chunks)} documents")
        print(f"✓ BM25 index saved to {bm25_index_path}")

        # 配置 MultiPath Retriever (Dense + BM25)
        print("\n🔧 Configuring MultiPath Retriever...")

        dense_retriever_config = DenseRetrieverConfig(
            index_config=faiss_config,
            search_kwargs={"k": 5}
        )

        bm25_retriever_config = TantivyBM25RetrieverConfig(
            index_config=bm25_config,
            search_kwargs={"k": 5}
        )

        multipath_config = MultiPathRetrieverConfig(
            retrievers=[dense_retriever_config, bm25_retriever_config],
            fusion_method="rrf",  # Reciprocal Rank Fusion
            rrf_k=60,
            search_kwargs={"k": 5, "with_score": True}
        )

        retriever = multipath_config.build()

        # 获取 MultiPath 信息
        info = retriever.get_multipath_info()
        print(f"✓ MultiPath Retriever configured:")
        print(f"  - Retriever count: {info['retriever_count']}")
        print(f"  - Retriever types: {info['retriever_types']}")
        print(f"  - Fusion method: {info['fusion_method']}")

        # 测试场景1: User 1 查询 Python 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 1: User 1 searches for Python content (with fusion)")
        print("-"*80)
        query = "programming language features"
        print(f"Query: '{query}'")
        print(f"User: User 1 ({user1_id[:8]}...)")

        results_user1 = retriever.invoke(
            query,
            k=5,
            owner_id=user1_id,
            over_fetch_multiplier=3  # For Dense Retriever
        )
        print(f"  Retrieved: {len(results_user1)} / 5 chunks")
        for i, chunk in enumerate(results_user1[:5]):
            score = chunk.metadata.get('score', 0.0) if chunk.metadata else 0.0
            print(f"    {i+1}. {chunk.id} (score: {score:.4f}) - {chunk.content[:60]}...")

        # 验证所有结果都属于 User 1
        assert all(chunk.owner_id == user1_id for chunk in results_user1), \
            "All results should belong to User 1"
        print(f"  ✓ All results belong to User 1")

        # 测试场景2: User 2 查询 Java 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 2: User 2 searches for Java content (with fusion)")
        print("-"*80)
        query = "object oriented programming"
        print(f"Query: '{query}'")
        print(f"User: User 2 ({user2_id[:8]}...)")

        results_user2 = retriever.invoke(
            query,
            k=5,
            owner_id=user2_id,
            over_fetch_multiplier=3
        )
        print(f"  Retrieved: {len(results_user2)} / 5 chunks")
        for i, chunk in enumerate(results_user2[:5]):
            score = chunk.metadata.get('score', 0.0) if chunk.metadata else 0.0
            print(f"    {i+1}. {chunk.id} (score: {score:.4f}) - {chunk.content[:60]}...")

        assert all(chunk.owner_id == user2_id for chunk in results_user2), \
            "All results should belong to User 2"
        print(f"  ✓ All results belong to User 2")

        # 测试场景3: User 3 查询 JavaScript 相关内容
        print(f"\n" + "-"*80)
        print("Scenario 3: User 3 searches for JavaScript content (with fusion)")
        print("-"*80)
        query = "asynchronous programming"
        print(f"Query: '{query}'")
        print(f"User: User 3 ({user3_id[:8]}...)")

        results_user3 = retriever.invoke(
            query,
            k=5,
            owner_id=user3_id,
            over_fetch_multiplier=3
        )
        print(f"  Retrieved: {len(results_user3)} / 5 chunks")
        for i, chunk in enumerate(results_user3[:5]):
            score = chunk.metadata.get('score', 0.0) if chunk.metadata else 0.0
            print(f"    {i+1}. {chunk.id} (score: {score:.4f}) - {chunk.content[:60]}...")

        assert all(chunk.owner_id == user3_id for chunk in results_user3), \
            "All results should belong to User 3"
        print(f"  ✓ All results belong to User 3")

        # 测试场景4: 无 owner_id 过滤
        print(f"\n" + "-"*80)
        print("Scenario 4: Search without owner_id filter (all users, with fusion)")
        print("-"*80)
        query = "programming language"
        print(f"Query: '{query}'")
        print(f"User: None (no filter)")

        results_all = retriever.invoke(query, k=10)
        print(f"  Retrieved: {len(results_all)} chunks from all users")

        # 统计每个用户的文档数量
        user_counts = {}
        for chunk in results_all:
            user_counts[chunk.owner_id] = user_counts.get(chunk.owner_id, 0) + 1

        for owner_id, count in user_counts.items():
            user_label = "User 1" if owner_id == user1_id else ("User 2" if owner_id == user2_id else "User 3")
            print(f"    {user_label}: {count} chunks")

        print(f"\n✅ MultiPath Retriever test passed!")

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


def main():
    try:
        test_user_isolation_with_dense_retriever()
        test_user_isolation_with_bm25_retriever()
        test_user_isolation_with_multipath_retriever()

        print(f"\n" + "="*80)
        print("✅ All tests passed!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

