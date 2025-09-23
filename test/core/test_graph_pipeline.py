import sys
import os
import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.retrieval.graph_retrieveal.graph_retrieval import GraphRetrieval
from core.file_management.extractor.graphextractor import GraphExtractor
from config.encapsulation.database.neo4j_with_embedding_config import Neo4jVectorConfig
from config.encapsulation.llm.huggingface_config import HuggingFaceEmbedConfig
from config.encapsulation.llm.openai_config import OpenAIConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.core.retrieval.graph_retrieval_config import GraphRetrievalConfig
from encapsulation.data_model.schema import Document, GraphData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_production_configs() -> Dict[str, Any]:
    """Create production-ready configurations for the complete pipeline"""

    # Real embedding configuration using Qwen model
    embedding_config = HuggingFaceEmbedConfig(
        type="huggingface_embedding",
        model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
        device="cuda:0",  # Use GPU for better performance
        task_types="embedding"
    )

    # Real LLM configuration for extraction and entity filtering
    llm_config = OpenAIConfig(
        type="openai",
        model_name="gpt-4o-mini",
        task_types="chat",
        api_key="sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2",  # Replace with actual API key
        base_url="https://api.gptsapi.net/v1",
        default_max_tokens=2000,
        default_temperature=0.1
    )

    # Neo4j configuration with embedding support
    neo4j_config = Neo4jVectorConfig(
        type="neo4j_vector",
        url="bolt://localhost:6550",
        username="neo4j",
        password="12345678",
        database="neo4j",
        embedding=embedding_config
    )

    # GraphExtractor configuration for entity and relationship extraction
    extractor_config = GraphExtractorConfig(
        type="graph_extractor",
        llm_config=llm_config,
        enable_cleaning=True,
        enable_llm_cleaning=True,
        max_rounds=2
    )

    # Graph retrieval configuration with LLM-based entity filtering
    retrieval_config = GraphRetrievalConfig(
        type="graph_retrieval",
        neo4j_config=neo4j_config,
        llm_config=llm_config,  # Enable LLM-based entity filtering
        k1_chunks=20,
        k2_entities=8,
        max_hops=3,
        beam_size=15,
        damping_factor=0.85,
        max_iterations=30,
        tolerance=1e-6,
        # Scoring parameters optimized for technical documents
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
        "neo4j_config": neo4j_config,
        "extractor_config": extractor_config,
        "retrieval_config": retrieval_config,
    }


def load_real_documents() -> List[Document]:
    """Load real documents from the test data file"""
    try:
        with open("test/tcl_gb_chunk.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for i, item in enumerate(data[:10]):  # Use first 10 documents for testing
            doc = Document(
                id=f"doc_{i+1}",
                content=item["content"],
                metadata=item.get("metadata", {})
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} real documents from test data")
        return documents

    except Exception as e:
        logger.warning(f"Failed to load real documents: {e}. Using sample documents instead.")





async def test_comprehensive_pipeline():
    """Test the complete end-to-end pipeline: Extraction → Storage → Retrieval"""

    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE END-TO-END GRAPH-BASED RAG SYSTEM TEST")
    logger.info("=" * 80)

    # Performance tracking
    phase_times = {}

    try:
        # ===== PHASE 1: CONFIGURATION =====
        logger.info("\n🔧 PHASE 1: CONFIGURATION SETUP")
        logger.info("-" * 50)

        start_time = time.time()
        configs = create_production_configs()
        phase_times["configuration"] = time.time() - start_time

        logger.info("✓ Production configurations created")
        logger.info(f"  - Embedding model: {configs['embedding_config'].model_name}")
        logger.info(f"  - LLM model: {configs['llm_config'].model_name}")
        logger.info(f"  - Neo4j database: {configs['neo4j_config'].url}")

        # ===== PHASE 2: DATA EXTRACTION =====
        logger.info("\n📄 PHASE 2: DATA EXTRACTION FROM REAL DOCUMENTS")
        logger.info("-" * 50)

        start_time = time.time()

        # Load real documents
        documents = load_real_documents()
        logger.info(f"✓ Loaded {len(documents)} real documents")

        # Initialize GraphExtractor
        extractor = GraphExtractor(configs["extractor_config"])
        logger.info("✓ GraphExtractor initialized")

        # Extract graph data from documents
        extracted_docs = []
        for i, doc in enumerate(documents[:5], 1):  # Process first 5 documents
            logger.info(f"  Processing document {i}/{min(5, len(documents))}: {doc.id}")
            try:
                # Extract entities and relationships
                graph_data = await extractor.extract(doc)
                doc.graph = graph_data
                extracted_docs.append(doc)

                logger.info(f"    ✓ Extracted {len(graph_data.entities)} entities, {len(graph_data.relations)} relations")

                # Show sample entities
                if graph_data.entities:
                    sample_entities = graph_data.entities[:3]
                    for entity in sample_entities:
                        logger.info(f"      Entity: {entity.get('entity_name', 'Unknown')} ({entity.get('entity_type', 'Unknown')})")

            except Exception as e:
                logger.error(f"    ✗ Failed to extract from document {doc.id}: {e}")
                continue

        phase_times["extraction"] = time.time() - start_time
        logger.info(f"✓ Data extraction completed in {phase_times['extraction']:.2f}s")
        logger.info(f"  Successfully processed {len(extracted_docs)} documents")

        # ===== PHASE 3: DATA STORAGE =====
        logger.info("\n💾 PHASE 3: DATA STORAGE IN NEO4J DATABASE")
        logger.info("-" * 50)

        start_time = time.time()

        # Initialize Neo4j store
        neo4j_store = configs["neo4j_config"].build()
        logger.info("✓ Neo4j vector store initialized")

        # Test database connection
        try:
            test_result = neo4j_store._execute_query("RETURN 1 as test")
            logger.info("✓ Database connection verified")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            logger.info("Skipping storage phase...")
            phase_times["storage"] = 0
            extracted_docs = []  # Clear for retrieval phase
        else:
            # Store documents and graph data
            stored_count = 0
            for doc in extracted_docs:
                try:
                    # Store document with embedding
                    neo4j_store.add_document(doc)

                    # Store graph data with entity embeddings
                    if doc.graph and (doc.graph.entities or doc.graph.relations):
                        neo4j_store.add_graph_data(doc.graph, doc.id)

                    stored_count += 1
                    logger.info(f"  ✓ Stored document {doc.id} with graph data")

                except Exception as e:
                    logger.error(f"  ✗ Failed to store document {doc.id}: {e}")
                    continue

            phase_times["storage"] = time.time() - start_time
            logger.info(f"✓ Data storage completed in {phase_times['storage']:.2f}s")
            logger.info(f"  Successfully stored {stored_count} documents with graph data")

        # ===== PHASE 4: RETRIEVAL WITH LLM FILTERING =====
        logger.info("\n🔍 PHASE 4: GRAPH-BASED RETRIEVAL WITH LLM FILTERING")
        logger.info("-" * 50)

        start_time = time.time()

        # Initialize retrieval system with LLM filtering
        retrieval_system_llm = GraphRetrieval(configs["retrieval_config"])
        logger.info("✓ Graph retrieval system initialized (with LLM filtering)")

        # Test queries relevant to the technical documents
        test_queries = [
            "蒸发器设计规范有哪些要求？",
            "实验室测试偏差的原因是什么？",
            "TCL空调的性能参数包括哪些？",
            "如何解决室外机安装位置问题？",
            "保温套管的设计要求是什么？"
        ]

        llm_results = {}
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n  Query {i}: {query}")
            try:
                query_start = time.time()
                results = retrieval_system_llm.retrieve(query)
                query_time = time.time() - query_start

                llm_results[query] = {
                    "results": results,
                    "time": query_time,
                    "count": len(results)
                }

                logger.info(f"    ✓ Retrieved {len(results)} results in {query_time:.2f}s")

                # Show top result
                if results:
                    top_result = results[0]
                    # Handle both Document objects and dictionaries
                    if hasattr(top_result, 'metadata'):
                        # Document object
                        score = top_result.metadata.get('score', 0)
                        content = top_result.content
                    else:
                        # Dictionary
                        score = top_result.get('score', 0)
                        content = top_result.get('content', '')

                    content_preview = content
                    logger.info(f"    Top result (score: {score:.3f}): {content_preview}")

            except Exception as e:
                logger.error(f"    ✗ Query failed: {e}")
                llm_results[query] = {"results": [], "time": 0, "count": 0, "error": str(e)}

        phase_times["retrieval_llm"] = time.time() - start_time
        logger.info(f"✓ LLM-enhanced retrieval completed in {phase_times['retrieval_llm']:.2f}s")

        # ===== PHASE 5: PERFORMANCE ANALYSIS =====
        logger.info("\n📊 PHASE 5: PERFORMANCE ANALYSIS")
        logger.info("-" * 50)

        # Overall performance summary
        total_time = sum(phase_times.values())
        logger.info(f"\n⏱️  PERFORMANCE SUMMARY:")
        logger.info(f"  Configuration:     {phase_times.get('configuration', 0):.2f}s")
        logger.info(f"  Data Extraction:   {phase_times.get('extraction', 0):.2f}s")
        logger.info(f"  Data Storage:      {phase_times.get('storage', 0):.2f}s")
        logger.info(f"  LLM Retrieval:     {phase_times.get('retrieval_llm', 0):.2f}s")
        logger.info(f"  Total Time:        {total_time:.2f}s")

        return {
            "phase_times": phase_times,
            "llm_results": llm_results,
            "documents_processed": len(extracted_docs),
            "queries_tested": len(test_queries)
        }

    except Exception as e:
        logger.error(f"Comprehensive test failed with error: {e}")
        raise



async def main():
    """Main function to run all tests"""
    logger.info("Starting Graph-Based RAG System Tests")

    # Run comprehensive end-to-end test
    try:
        logger.info("\nRunning Comprehensive End-to-End Pipeline Test...")
        comprehensive_results = await test_comprehensive_pipeline()
        logger.info("Comprehensive test completed successfully!")

        # Print summary
        logger.info(f"\n📋 FINAL SUMMARY:")
        logger.info(f"  Documents processed: {comprehensive_results['documents_processed']}")
        logger.info(f"  Queries tested: {comprehensive_results['queries_tested']}")
        logger.info(f"  Total time: {sum(comprehensive_results['phase_times'].values()):.2f}s")

    except Exception as e:
        logger.error(f"❌ Comprehensive test failed: {e}")


if __name__ == "__main__":
    # Run all tests
    asyncio.run(main())
