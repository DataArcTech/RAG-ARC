import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import tempfile
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from datetime import timezone
from core.file_management.IndexManager import IndexManager, IndexManagerConfig, PostgreSQLDBConfig, LocalDBConfig
from core.utils.data_model import Document
from encapsulation.database.relational_db.data_schema import ChunksMetadata, ChunksStatus
from core.retrieval.dense import DenseRetrieverConfig
from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig
from encapsulation.database.vector_db.faiss import FaissIndexConfig
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig


def create_test_documents():
    """创建测试文档"""
    return [
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="人工智能技术正在快速发展，包括机器学习、深度学习和自然语言处理等领域。",
            metadata={"source": "ai_overview", "topic": "artificial_intelligence"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="机器学习算法能够从数据中自动学习模式，无需明确编程指令。",
            metadata={"source": "ml_guide", "topic": "machine_learning"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="深度学习使用多层神经网络来处理复杂的数据表示和特征提取。",
            metadata={"source": "dl_tutorial", "topic": "deep_learning"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="natural language processing is a subfield of artificial intelligence that focuses on the interaction between computers and humans using natural language.",
            metadata={"source": "nlp_basics", "topic": "natural_language_processing"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="计算机视觉技术使机器能够识别和分析图像内容。",
            metadata={"source": "cv_intro", "topic": "computer_vision"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="机器人技术结合了人工智能和机械工程，创造出能够执行任务的智能机器。",
            metadata={"source": "robotics_101", "topic": "robotics"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="区块链技术提供了一种去中心化的数据存储和传输方式。",
            metadata={"source": "blockchain_basics", "topic": "blockchain_technology"}
        ),
        Document(
            id=f"retrieval_test_{uuid.uuid4().hex[:8]}",
            content="量子计算利用量子力学原理进行信息处理，有望解决传统计算机难以解决的问题。",
            metadata={"source": "quantum_computing", "topic": "quantum_technology"}
        )
    ]


def create_chunks_for_retrieval_test(documents, index_manager):
    """为检索测试创建chunks数据"""
    created_chunks_ids = []
    
    for i, doc in enumerate(documents):
        # 创建chunks JSON数据
        chunks_data = {
            "chunks": [{
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }]
        }
        
        # 保存到文件数据库
        blob_key = f"retrieval_test_{uuid.uuid4().hex[:8]}.json"
        chunks_json = json.dumps(chunks_data, ensure_ascii=False)
        index_manager._file_db.store(blob_key, chunks_json.encode('utf-8'))
        
        # 为BM25创建chunks记录
        bm25_chunks_id = f"retrieval_bm25_{uuid.uuid4().hex[:8]}"
        bm25_chunks_metadata = ChunksMetadata(
            chunks_id=bm25_chunks_id,
            source_parsed_content_id=f"retrieval_parsed_{uuid.uuid4().hex[:8]}",
            blob_key=blob_key,
            chunks_count=1,
            content_size=len(chunks_json),
            checksum=f"retrieval_checksum_{uuid.uuid4().hex[:8]}",
            chunking_strategy="retrieval_test",
            status=ChunksStatus.CHUNKED,
            index_type="bm25_indexer",
            created_at=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
            updated_at=datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        )
        
        # 为FAISS创建chunks记录
        faiss_chunks_id = f"retrieval_faiss_{uuid.uuid4().hex[:8]}"
        faiss_chunks_metadata = ChunksMetadata(
            chunks_id=faiss_chunks_id,
            source_parsed_content_id=f"retrieval_parsed_{uuid.uuid4().hex[:8]}",
            blob_key=blob_key,
            chunks_count=1,
            content_size=len(chunks_json),
            checksum=f"retrieval_checksum_{uuid.uuid4().hex[:8]}",
            chunking_strategy="retrieval_test",
            status=ChunksStatus.CHUNKED,
            index_type="faiss",
            created_at=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
            updated_at=datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        )
        
        # 存储到数据库
        bm25_id = index_manager._relational_db.store_chunks_metadata(bm25_chunks_metadata)
        faiss_id = index_manager._relational_db.store_chunks_metadata(faiss_chunks_metadata)
        
        created_chunks_ids.extend([bm25_id, faiss_id])
        print(f"✅ 创建文档 {i+1} 的chunks: BM25({bm25_id}), FAISS({faiss_id})")
    
    return created_chunks_ids


def test_multi_index_retrieval():
    """测试多索引类型的构建和检索"""
    print("=== 多索引类型构建和检索测试 ===\n")
    
    created_chunks_ids = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 配置IndexManager
            print("1. 配置IndexManager...")
            
            bm25_index_path = os.path.join(temp_dir, "bm25_index")
            faiss_index_path = os.path.join(temp_dir, "faiss_index")
            os.makedirs(bm25_index_path, exist_ok=True)
            os.makedirs(faiss_index_path, exist_ok=True)
            
            # 创建配置对象
            relational_db_config = PostgreSQLDBConfig(
                host="localhost",
                port=5432,
                database="rag_arc_test",
                user="postgres",
                password="123"
            )

            file_db_config = LocalDBConfig(
                base_path=temp_dir
            )

            bm25_config = BM25IndexBuilderConfig(
                index_path=bm25_index_path
            )

            faiss_config = FaissIndexConfig(
                index_path=faiss_index_path,
                metric="cosine",
                index_type="flat",
                normalize_L2=True
            )

            embedding_config = HuggingFaceEmbedConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cuda:1"
            )

            config = IndexManagerConfig(
                relational_db_config=relational_db_config,
                file_db_config=file_db_config,
                indexer_configs={
                    "bm25_indexer": bm25_config,
                    "faiss": faiss_config
                },
                embedding_config=embedding_config,
                batch_size=3,
                max_concurrent_builds=2
            )
            
            # 2. 创建IndexManager并构建索引
            print("2. 创建IndexManager并构建索引...")
            index_manager = IndexManager(config)
            
            # 创建测试数据
            documents = create_test_documents()
            created_chunks_ids = create_chunks_for_retrieval_test(documents, index_manager)
            print(f"✅ 创建了 {len(documents)} 个文档，{len(created_chunks_ids)} 个chunks记录")
            
            # 构建索引
            build_stats = index_manager.build_pending_indexes()
            print(f"✅ 索引构建完成: {build_stats}")
            
            # 检查索引状态
            final_stats = index_manager.get_index_statistics()
            health_status = index_manager.get_indexer_health()
            print(f"索引统计: {final_stats}")
            print(f"BM25健康状态: {health_status['bm25_indexer']['status']}")
            print(f"FAISS健康状态: {health_status['faiss']['status']}")
            
            # 3. 测试BM25检索
            print("\n3. 测试BM25检索...")
            try:
                from core.retrieval.tantivy_bm25 import TantivyBM25Retriever
                
                bm25_config = TantivyBM25RetrieverConfig(
                    index_config=BM25IndexBuilderConfig(
                        index_path=bm25_index_path
                    ),
                    search_kwargs={
                        "k": 3,
                        "with_score": True
                    }
                )
                
                bm25_retriever = TantivyBM25Retriever(bm25_config)
                
                # 测试查询
                queries = ["机器学习", "深度学习", "人工智能"]
                for query in queries:
                    print(f"\nBM25查询: '{query}'")
                    try:
                        # 尝试不同的搜索参数
                        results = bm25_retriever.invoke(query, k=5)
                        print(f"找到 {len(results)} 个结果:")
                        for i, doc in enumerate(results, 1):
                            score = doc.metadata.get('score', 'N/A')
                            print(f"  {i}. [分数: {score}] {doc.content[:40]}...")

                        # 如果没有结果，尝试英文查询
                        if len(results) == 0 and query == "机器学习":
                            print("  尝试英文查询: 'machine learning'")
                            en_results = bm25_retriever.invoke("machine learning", k=5)
                            print(f"  英文查询找到 {len(en_results)} 个结果:")
                            for i, doc in enumerate(en_results, 1):
                                score = doc.metadata.get('score', 'N/A')
                                print(f"    {i}. [分数: {score}] {doc.content[:40]}...")

                    except Exception as e:
                        print(f"BM25检索失败: {e}")
                        
            except Exception as e:
                print(f"BM25检索器创建失败: {e}")
            
            # 检查索引文件
            print(f"\n检查索引文件...")
            print(f"FAISS索引路径: {faiss_index_path}")
            if os.path.exists(faiss_index_path):
                files = os.listdir(faiss_index_path)
                print(f"索引文件: {files}")
            else:
                print("❌ 索引路径不存在！")

            # 4. 测试FAISS检索
            print("\n4. 测试FAISS检索...")
            try:
                from core.retrieval.dense import DenseRetriever

                faiss_config = DenseRetrieverConfig(
                    index_config=FaissIndexConfig(
                        index_path=faiss_index_path,
                        metric="cosine",
                        index_type="flat",
                        normalize_L2=True
                    ),
                    embedding_config=HuggingFaceEmbedConfig(
                        type="huggingface_embedding",
                        model_name = "sentence-transformers/all-MiniLM-L6-v2",
                        device="cuda:1"
                    ),
                    search_kwargs={
                        "k": 3,
                        "with_score": True
                    }
                )

                faiss_retriever = DenseRetriever(faiss_config)

                # 比较IndexManager的索引和DenseRetriever的索引
                print(f"\n比较索引...")
                index_manager_faiss = index_manager._indexers["faiss"]
                retriever_faiss = faiss_retriever.get_index()

                print(f"IndexManager索引统计: {index_manager_faiss.get_index_stats()}")
                print(f"DenseRetriever索引统计: {retriever_faiss.get_index_stats()}")

                print(f"IndexManager docstore大小: {len(index_manager_faiss.docstore)}")
                print(f"DenseRetriever docstore大小: {len(retriever_faiss.docstore)}")

                # 检查是否是同一个对象
                print(f"是否是同一个索引对象: {index_manager_faiss is retriever_faiss}")
                print(f"是否是同一个FAISS索引: {index_manager_faiss.index is retriever_faiss.index}")

                # 检查嵌入模型
                print(f"\n检查嵌入模型...")
                index_manager_embedding = index_manager._embedding_model
                retriever_embedding = faiss_retriever.get_embedding()
                print(f"是否是同一个嵌入模型对象: {index_manager_embedding is retriever_embedding}")

                # 测试查询
                queries = ["机器学习", "深度学习", "计算机视觉技术使机器能够识别和分析图像内容。"]
                for query in queries:
                    print(f"\nFAISS查询: '{query}'")
                    try:
                        results = faiss_retriever.invoke(query)
                        print(f"最终结果 (找到 {len(results)} 个):")
                        for i, doc in enumerate(results, 1):
                            score = doc.metadata.get('score', 'N/A')
                            print(f"  {i}. [最终分数: {score}] {doc.content[:40]}...")
                    except Exception as e:
                        print(f"FAISS检索失败: {e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"FAISS检索器创建失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n=== 多索引检索测试完成 ===")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理数据
        if created_chunks_ids:
            print("\n=== 清理测试数据 ===")
            try:
                for chunks_id in created_chunks_ids:
                    success = index_manager._relational_db.delete_chunks_metadata(chunks_id)
                    if success:
                        print(f"✅ 删除chunks: {chunks_id}")
            except Exception as e:
                print(f"❌ 清理失败: {e}")
            print("=== 清理完成 ===")


if __name__ == "__main__":
    test_multi_index_retrieval()
