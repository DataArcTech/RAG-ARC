import sys
import os
import logging
import json
import time
import asyncio
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.retrieval.graph_retrieveal.graph_retrieval import GraphRetrieval
from core.file_management.extractor.graphextractor import GraphExtractor
from config.encapsulation.database.graph_db.networkx_with_embedding_config import NetworkXVectorConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.core.retrieval.graph_retrieval_config import GraphRetrievalConfig
from encapsulation.data_model.schema import Chunk, GraphData

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_networkx_configs(temp_dir: str) -> Dict[str, Any]:
    """Create NetworkX-based configurations for the complete pipeline"""
    
    # Embedding configuration using HuggingFace model
    embedding_config = QwenEmbeddingConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cuda:0",  
        use_china_mirror=True,
        cache_folder="./models/Qwen"
    )
    
    # LLM configuration for extraction and entity filtering
    llm_config = OpenAIChatConfig(
        model_name="gpt-4o-mini",
        max_tokens=2000,
        temperature=0.1
    )
    
    # NetworkX configuration with embedding support
    networkx_config = NetworkXVectorConfig(
        type="networkx_vector",
        storage_path=os.path.join(temp_dir, "networkx_pipeline_graph.pkl"),
        auto_save=True,
        similarity_threshold=0.7,
        cache_embeddings=True,
        cache_size=1000,
        embedding=embedding_config
    )
    
    # GraphExtractor configuration for entity and relationship extraction
    extractor_config = GraphExtractorConfig(
        type="graph_extractor",
        llm_config=llm_config,
        enable_cleaning=True,
        max_concurrent=100
        # enable_llm_cleaning=True,
        # max_rounds=2
    )
    
    # Graph retrieval configuration optimized for NetworkX
    retrieval_config = GraphRetrievalConfig(
        type="graph_retrieval",
        graph_config=networkx_config,  # Use NetworkX instead of Neo4j
        embedding_config=embedding_config,
        llm_config=llm_config,  # Enable LLM-based entity filtering
        k1_chunks=20,
        k2_entities=8,
        max_hops=3,
        beam_size=15,
        damping_factor=0.85,
        max_iterations=30,
        tolerance=1e-6,
        beta1=0.7,
        beta2=0.3,
        mu1=0.3,
        mu2=0.3,
        mu3=0.4,
        gamma1=0.4,
        gamma2=0.3,
        gamma3=0.3,
        lambda1=0.6,
        lambda2=0.4,
        eta=0.2,
        top_k_entities=10,
        alpha=0.6,
        beta=0.4,
        chunks_per_entity=8
    )
    
    return {
        "embedding_config": embedding_config,
        "llm_config": llm_config,
        "networkx_config": networkx_config,
        "extractor_config": extractor_config,
        "retrieval_config": retrieval_config,
    }


def load_test_documents() -> List[Chunk]:
    """Load test documents from real data file or create sample documents"""
    try:
        # Try to load real documents
        with open("./test/test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data[:5]):  # Use first 5 documents for testing
            doc = Chunk(
                id=f"doc_{i+1}",
                content=item["content"],
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} real documents from test data")
        return documents
        
    except Exception as e:
        logger.warning(f"Failed to load real documents: {e}. Using sample documents instead.")
        
        # Create sample technical documents
        sample_docs = [
            Chunk(
                id="doc_ai_tech",
                content="人工智能技术在现代工业中的应用越来越广泛。机器学习算法可以优化生产流程，"
                       "深度学习模型能够进行质量检测，神经网络技术提升了自动化水平。",
                metadata={"topic": "AI技术", "source": "技术文档"}
            ),
            Chunk(
                id="doc_hvac_system",
                content="空调系统的设计需要考虑多个因素。蒸发器的效率直接影响制冷效果，"
                       "压缩机的性能决定了系统的能耗，冷凝器的设计关系到散热效果。",
                metadata={"topic": "空调系统", "source": "设计规范"}
            ),
            Chunk(
                id="doc_quality_control",
                content="质量控制是生产过程中的关键环节。检测设备需要定期校准，"
                       "测试数据要进行统计分析，偏差控制需要建立标准流程。",
                metadata={"topic": "质量控制", "source": "操作手册"}
            ),
            Chunk(
                id="doc_installation",
                content="设备安装位置的选择至关重要。室外机需要考虑通风条件，"
                       "保温套管要符合设计要求，安装角度影响运行效果。",
                metadata={"topic": "设备安装", "source": "安装指南"}
            ),
            Chunk(
                id="doc_performance",
                content="性能参数的测试包括多个指标。制冷量是核心参数，"
                       "能效比反映节能水平，噪音控制关系到用户体验。",
                metadata={"topic": "性能测试", "source": "测试报告"}
            )
        ]
        
        logger.info(f"Created {len(sample_docs)} sample documents")
        return sample_docs


async def test_networkx_complete_pipeline():
    """Test the complete end-to-end NetworkX pipeline: Extraction → Storage → Retrieval"""
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE NETWORKX GRAPH-BASED RAG SYSTEM TEST")
    logger.info("=" * 80)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="networkx_pipeline_test_")
    logger.info(f" Created temp directory: {temp_dir}")
    
    # Performance tracking
    phase_times = {}
    
    try:
        # ===== PHASE 1: CONFIGURATION =====
        logger.info("\nPHASE 1: CONFIGURATION SETUP")
        logger.info("-" * 50)
        
        start_time = time.time()
        configs = create_networkx_configs(temp_dir)
        phase_times["configuration"] = time.time() - start_time
        
        logger.info("NetworkX configurations created")
        logger.info(f"  - Embedding model: {configs['embedding_config'].model_name}")
        logger.info(f"  - LLM model: {configs['llm_config'].model_name}")
        logger.info(f"  - NetworkX storage: {configs['networkx_config'].storage_path}")
        
        # ===== PHASE 2: DATA EXTRACTION =====
        logger.info("\nPHASE 2: DATA EXTRACTION FROM DOCUMENTS")
        logger.info("-" * 50)
        
        start_time = time.time()
        
        # Load test documents
        documents = load_test_documents()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Initialize GraphExtractor
        try:
            extractor = GraphExtractor(configs["extractor_config"])
            logger.info("GraphExtractor initialized")
            extraction_enabled = True
        except Exception as e:
            logger.warning(f"Failed to initialize GraphExtractor: {e}")
            logger.info("Skipping extraction phase, using mock graph data...")
            extraction_enabled = False
        
        # Extract or create graph data
        extracted_docs = []
        if extraction_enabled:
            # Real extraction using GraphExtractor with concurrent processing via __call__ method
            logger.info(f"  Processing {len(documents)} documents concurrently using extractor(documents) (max_concurrent={configs['extractor_config'].max_concurrent})")
            try:
                # Use extractor(documents) which internally calls extract_concurrent()
                extracted_docs = extractor(documents)

                # Log results for each document
                for i, doc in enumerate(extracted_docs, 1):
                    if doc.graph and not doc.graph.is_empty():
                        logger.info(f"  Document {i}/{len(extracted_docs)}: {doc.id}")
                        logger.info(f"    Extracted {len(doc.graph.entities)} entities, {len(doc.graph.relations)} relations")

                        # Show sample entities
                        if doc.graph.entities:
                            sample_entities = doc.graph.entities[:2]
                            for entity in sample_entities:
                                entity_name = entity.get('entity_name', entity.get('name', 'Unknown'))
                                entity_type = entity.get('entity_type', entity.get('type', 'Unknown'))
                                logger.info(f"      Entity: {entity_name} ({entity_type})")
                    else:
                        logger.warning(f"  Document {i}/{len(extracted_docs)}: {doc.id} - No graph data extracted")

            except Exception as e:
                logger.error(f"    ✗ Failed to extract from documents: {e}")
                raise
        else:
            raise Exception("GraphExtractor not available")
        
        phase_times["extraction"] = time.time() - start_time
        logger.info(f"Data extraction completed in {phase_times['extraction']:.2f}s")
        logger.info(f"  Successfully processed {len(extracted_docs)} documents")
        
        return await continue_pipeline_test(configs, extracted_docs, phase_times, temp_dir)
        
    except Exception as e:
        logger.error(f"Pipeline test failed with error: {e}")
        raise
    finally:
        # Cleanup
        logger.info(f"\n🧹 Cleaning up temp directory...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.info("✅ Cleanup completed")



async def continue_pipeline_test(configs: Dict[str, Any], extracted_docs: List[Chunk],
                                phase_times: Dict[str, float], temp_dir: str) -> Dict[str, Any]:
    """Continue the pipeline test with storage and retrieval phases"""

    # ===== PHASE 3: DATA STORAGE =====
    logger.info("\nPHASE 3: DATA STORAGE IN NETWORKX GRAPH")
    logger.info("-" * 50)

    start_time = time.time()

    # Initialize NetworkX store
    networkx_store = configs["networkx_config"].build()
    logger.info("NetworkX vector store initialized")

    # Test store health
    try:
        health_status = networkx_store.health_check()
        logger.info(f"Store health check: {health_status}")
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    # Store chunks and graph data
    stored_count = 0
    for doc in extracted_docs:
        try:
            # Store chunks with embedding
            networkx_store.add_chunk(doc)

            # Store graph data with entity embeddings
            if doc.graph and (doc.graph.entities or doc.graph.relations):
                networkx_store.add_graph_data(doc.graph, doc.id)

            stored_count += 1
            logger.info(f"  Stored chunk: {doc.id} with graph data")

        except Exception as e:
            logger.error(f"  ✗ Failed to store chunk {doc.id}: {e}")
            continue

    # Save the graph to ensure persistence for retrieval
    logger.info(" Saving graph to disk for retrieval...")
    networkx_store.save_index(temp_dir)
    logger.info("  Graph saved successfully")

    # Display graph statistics
    graph_stats = {
        "nodes": networkx_store.graph.number_of_nodes(),
        "edges": networkx_store.graph.number_of_edges()
    }

    phase_times["storage"] = time.time() - start_time
    logger.info(f"Data storage completed in {phase_times['storage']:.2f}s")
    logger.info(f"  Successfully stored {stored_count} documents with graph data")
    logger.info(f"  Graph statistics: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")

    # ===== PHASE 4: GRAPH-BASED RETRIEVAL =====
    logger.info("\nPHASE 4: GRAPH-BASED RETRIEVAL WITH NETWORKX")
    logger.info("-" * 50)

    start_time = time.time()

    # Initialize retrieval system
    retrieval_system = GraphRetrieval(configs["retrieval_config"])
    logger.info("Graph retrieval system initialized (NetworkX backend)")

    # Load the saved graph data into retrieval system
    logger.info("Loading graph data into retrieval system...")
    retrieval_system.graph_store.load_index(temp_dir)
    logger.info(f"  Loaded graph: {retrieval_system.graph_store.graph.number_of_nodes()} nodes, {retrieval_system.graph_store.graph.number_of_edges()} edges")

    # Test queries relevant to the technical documents
    test_queries = [
        "人工智能技术的应用有哪些？",
        "空调系统的主要组件包括什么？",
        "质量控制流程中需要注意什么？",
        "设备安装时要考虑哪些因素？",
        "性能参数测试包括哪些指标？"
    ]

    retrieval_results = {}
    total_query_time = 0

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n  Query {i}: {query}")
        try:
            query_start = time.time()
            results = retrieval_system.retrieve(query, top_k=3)
            query_time = time.time() - query_start
            total_query_time += query_time

            retrieval_results[query] = {
                "results": results,
                "time": query_time,
                "count": len(results)
            }

            logger.info(f"    Retrieved {len(results)} results in {query_time:.3f}s")

            # Show top result details
            if results:
                top_result = results[0]
                score = top_result.metadata.get('score', 0)
                graph_score = top_result.metadata.get('graph_score', 0)
                embedding_score = top_result.metadata.get('embedding_score', 0)
                entities = top_result.metadata.get('mentioned_entities', [])

                content_preview = top_result.content[:60] + "..." if len(top_result.content) > 60 else top_result.content
                logger.info(f"    Top result: {top_result.id} (score: {score:.3f})")
                logger.info(f"      Graph: {graph_score:.3f}, Embedding: {embedding_score:.3f}")
                logger.info(f"      Entities: {len(entities)} mentioned")
                logger.info(f"      Content: {content_preview}")

        except Exception as e:
            logger.error(f"    ✗ Query failed: {e}")
            retrieval_results[query] = {"results": [], "time": 0, "count": 0, "error": str(e)}

    avg_query_time = total_query_time / len(test_queries) if test_queries else 0
    phase_times["retrieval"] = time.time() - start_time
    logger.info(f"Graph-based retrieval completed in {phase_times['retrieval']:.2f}s")
    logger.info(f"  Average query time: {avg_query_time:.3f}s")

    # ===== PHASE 5: GRAPH ANALYSIS =====
    logger.info("\nPHASE 5: GRAPH ANALYSIS AND INSIGHTS")
    logger.info("-" * 50)

    start_time = time.time()

    # Analyze graph structure
    analysis_results = {}

    # Test entity analysis
    sample_entities = ["ai", "hvac", "qc"]
    for entity_id in sample_entities:
        try:
            entity_data = retrieval_system.get_entity_data(entity_id)
            if entity_data:
                neighbors = retrieval_system.get_entity_neighbors(entity_id)
                importance = retrieval_system.get_entity_importance(entity_id)
                specificity = retrieval_system.get_entity_specificity(entity_id)

                analysis_results[entity_id] = {
                    "name": entity_data.get('name', 'Unknown'),
                    "type": entity_data.get('type', 'Unknown'),
                    "neighbors": len(neighbors),
                    "importance": importance,
                    "specificity": specificity
                }

                logger.info(f"  Entity '{entity_id}': {entity_data.get('name', 'Unknown')} ({entity_data.get('type', 'Unknown')})")
                logger.info(f"    Neighbors: {len(neighbors)}, Importance: {importance:.3f}, Specificity: {specificity:.3f}")

        except Exception as e:
            logger.warning(f"  Failed to analyze entity '{entity_id}': {e}")

    phase_times["analysis"] = time.time() - start_time
    logger.info(f"Graph analysis completed in {phase_times['analysis']:.2f}s")

    # ===== PHASE 6: PERSISTENCE TEST =====
    logger.info("\nPHASE 6: PERSISTENCE AND LOADING TEST")
    logger.info("-" * 50)

    start_time = time.time()

    # Test save functionality
    save_path = os.path.join(temp_dir, "test_persistence.pkl")
    try:
        networkx_store.save_index(save_path)
        logger.info(f"Graph saved to {save_path}")

        # Test load functionality
        new_store = configs["networkx_config"].build()
        new_store.load_index(save_path)

        # Verify loaded data
        loaded_docs = new_store.get_chunks([doc.id for doc in extracted_docs[:3]])
        loaded_stats = {
            "nodes": new_store.graph.number_of_nodes(),
            "edges": new_store.graph.number_of_edges()
        }

        logger.info(f"Graph loaded successfully")
        logger.info(f"  Loaded {len(loaded_docs)} documents")
        logger.info(f"  Loaded graph: {loaded_stats['nodes']} nodes, {loaded_stats['edges']} edges")

        persistence_success = True

    except Exception as e:
        logger.error(f"✗ Persistence test failed: {e}")
        persistence_success = False

    phase_times["persistence"] = time.time() - start_time
    logger.info(f"Persistence test completed in {phase_times['persistence']:.2f}s")

    # ===== PHASE 7: PERFORMANCE SUMMARY =====
    logger.info("\n📊 PHASE 7: PERFORMANCE SUMMARY")
    logger.info("-" * 50)

    # Overall performance summary
    total_time = sum(phase_times.values())
    successful_queries = sum(1 for result in retrieval_results.values() if result.get("count", 0) > 0)

    logger.info(f"\nPERFORMANCE SUMMARY:")
    logger.info(f"  Configuration:     {phase_times.get('configuration', 0):.3f}s")
    logger.info(f"  Data Extraction:   {phase_times.get('extraction', 0):.3f}s")
    logger.info(f"  Data Storage:      {phase_times.get('storage', 0):.3f}s")
    logger.info(f"  Graph Retrieval:   {phase_times.get('retrieval', 0):.3f}s")
    logger.info(f"  Graph Analysis:    {phase_times.get('analysis', 0):.3f}s")
    logger.info(f"  Persistence Test:  {phase_times.get('persistence', 0):.3f}s")
    logger.info(f"  Total Time:        {total_time:.3f}s")

    logger.info(f"\nRESULTS SUMMARY:")
    logger.info(f"  Documents processed: {len(extracted_docs)}")
    logger.info(f"  Graph nodes: {graph_stats['nodes']}")
    logger.info(f"  Graph edges: {graph_stats['edges']}")
    logger.info(f"  Queries tested: {len(test_queries)}")
    logger.info(f"  Successful queries: {successful_queries}")
    logger.info(f"  Average query time: {avg_query_time:.3f}s")
    logger.info(f"  Persistence: {'Success' if persistence_success else '✗ Failed'}")

    return {
        "phase_times": phase_times,
        "retrieval_results": retrieval_results,
        "analysis_results": analysis_results,
        "graph_stats": graph_stats,
        "documents_processed": len(extracted_docs),
        "queries_tested": len(test_queries),
        "successful_queries": successful_queries,
        "avg_query_time": avg_query_time,
        "persistence_success": persistence_success,
        "total_time": total_time
    }


async def main():
    """Main function to run the complete NetworkX pipeline test"""
    logger.info("Starting Complete NetworkX Graph-Based RAG System Test")

    try:
        logger.info("\nRunning Complete End-to-End NetworkX Pipeline Test...")
        results = await test_networkx_complete_pipeline()

        logger.info("\n" + "=" * 80)
        logger.info("COMPLETE NETWORKX PIPELINE TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        # Print final summary
        logger.info(f"\nFINAL SUMMARY:")
        logger.info(f"  Documents processed: {results['documents_processed']}")
        logger.info(f"  Graph nodes created: {results['graph_stats']['nodes']}")
        logger.info(f"  Graph edges created: {results['graph_stats']['edges']}")
        logger.info(f"  Queries tested: {results['queries_tested']}")
        logger.info(f"  Successful queries: {results['successful_queries']}")
        logger.info(f"  Average query time: {results['avg_query_time']:.3f}s")
        logger.info(f"  Total pipeline time: {results['total_time']:.3f}s")
        logger.info(f"  Persistence test: {'Passed' if results['persistence_success'] else '✗ Failed'}")


        return True

    except Exception as e:
        logger.error(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the complete pipeline test
    success = asyncio.run(main())

    if success:
        print("\nComplete NetworkX Pipeline Test PASSED")
        exit(0)
    else:
        print("\nComplete NetworkX Pipeline Test FAILED")
        exit(1)
