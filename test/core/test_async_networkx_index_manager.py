"""
与test_async_index_manager.py的区别在于：
这份代码仅针对graph构建索引
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
from config.core.file_management.index_manager_config import IndexManagerConfig
from config.core.file_management.storage.file_storage import FileStorageConfig
from config.core.file_management.storage.parsed_content_storage import ParsedContentStorageConfig
from config.core.file_management.storage.chunk_storage import ChunkStorageConfig
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

from config.core.file_management.parser_combinator_config import ParserCombinatorConfig
from config.core.file_management.chunker.chunker_config import RecursiveChunkerConfig

from config.core.file_management.indexing.graph_indexing.networkx_indexing_config import NetworkXGraphIndexerConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.encapsulation.database.graph_db.networkx_with_embedding_config import NetworkXVectorConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig

# Test file path
TEST_FILE = 'test/test_docx.docx'


def create_storage_instances():
    """Create all storage instances with shared database configuration"""
    logger.info("Creating storage instances...")

    # Shared configurations
    file_db_config = LocalDBConfig(
        base_path="./test_output/networkx_index_test"
    )

    postgresql_config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="rag_test",
        user="chenmingzhen",
        password="123"
    )

    # Create all three storage configs
    file_storage_config = FileStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=postgresql_config
    )

    parsed_content_storage_config = ParsedContentStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=postgresql_config
    )

    chunk_storage_config = ChunkStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=postgresql_config
    )

    # Build all storage instances
    file_storage = file_storage_config.build()
    parsed_content_storage = parsed_content_storage_config.build()
    chunk_storage = chunk_storage_config.build()

    return file_storage, parsed_content_storage, chunk_storage


def create_index_manager_with_networkx():
    """Create IndexManager with NetworkX Graph indexing"""
    logger.info("Creating IndexManager with NetworkX Graph indexing...")

    # Parser configuration
    parser_config = ParserCombinatorConfig()
    
    # Chunker configuration
    chunker_config = RecursiveChunkerConfig(
        chunk_size=400,
        chunk_overlap=40
    )

    # LLM configuration for graph extraction
    llm_config = OpenAIChatConfig(
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    # Embedding configuration for NetworkX
    embedding_config = QwenEmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_china_mirror=True,
        cache_folder="./models"
    )

    # GraphExtractor configuration
    extractor_config = GraphExtractorConfig(
        type="graph_extractor",
        llm_config=llm_config,
        enable_cleaning=True,
        enable_llm_cleaning=False,
        max_rounds=1,
        max_concurrent=10
    )

    # NetworkX Vector Graph Store configuration
    networkx_config = NetworkXVectorConfig(
        type="networkx_vector",
        storage_path="./test_output/networkx_index_test/graph_store",
        index_name="docx_graph_index",
        auto_save=True,
        similarity_threshold=0.5,
        cache_embeddings=True,
        embedding_cache_size=1000,
        embedding=embedding_config
    )

    # NetworkX Graph Indexer configuration
    networkx_indexer_config = NetworkXGraphIndexerConfig(
        type="networkx_graph_indexer",
        extractor_config=extractor_config,
        graph_store_config=networkx_config
    )

    # Create IndexManager with NetworkX indexer
    config = IndexManagerConfig(
        parser_config=parser_config,
        chunker_config=chunker_config,
        indexer_configs=[networkx_indexer_config]
    )

    return config.build()


async def test_async_networkx_indexing():
    """Test the async index_file method with NetworkX Graph indexing for DOCX file"""
    logger.info("=== Testing Async NetworkX Graph Indexing ===")

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set it in your .env file to run the tests")
        return False

    try:
        # Create all storage instances
        file_storage, parsed_content_storage, chunk_storage = create_storage_instances()

        # Create IndexManager with NetworkX indexing
        index_manager = create_index_manager_with_networkx()
        index_manager.file_storage = file_storage
        index_manager.parsed_content_storage = parsed_content_storage
        index_manager.chunk_storage = chunk_storage

        # Check if test file exists
        if not os.path.exists(TEST_FILE):
            logger.error(f"Test file not found: {TEST_FILE}")
            return False

        logger.info(f"\nTesting DOCX file: {os.path.basename(TEST_FILE)}")

        # Read file content
        with open(TEST_FILE, 'rb') as f:
            file_content = f.read()

        # Upload file
        filename = "test_networkx_docx.docx"
        content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

        file_id = file_storage.upload_file(
            filename=filename,
            file_data=file_content,
            content_type=content_type
        )

        logger.info(f"  Uploaded: {filename} -> {file_id} ({len(file_content)} bytes)")

        # Test async index_file method
        logger.info("\nStarting indexing pipeline...")
        result = await index_manager.index_file(file_id)

        if result["success"]:
            logger.info("\n" + "=" * 80)
            logger.info("DOCX indexing with NetworkX succeeded!")
            logger.info("=" * 80)
            logger.info(f"  - Parsed content ID: {result['parsed_content_id']}")
            logger.info(f"  - Number of chunks: {len(result['chunk_ids'])}")
            logger.info(f"  - Parser used: {result['metadata']['parser_type']}")
            logger.info(f"  - Chunker used: {result['metadata']['chunker_type']}")

            # Show indexing results
            if result['indexing_results']:
                logger.info("\n  Indexing results:")
                for indexer_name, indexer_result in result['indexing_results'].items():
                    if indexer_result['success']:
                        logger.info(f"    ✓ {indexer_name}: {indexer_result.get('indexed_count', 0)} chunks indexed")
                        
                        # Show graph statistics if available
                        if 'graph_stats' in indexer_result:
                            stats = indexer_result['graph_stats']
                            logger.info(f"      - Graph nodes: {stats.get('num_nodes', 0)}")
                            logger.info(f"      - Graph edges: {stats.get('num_edges', 0)}")
                            logger.info(f"      - Total entities: {stats.get('total_entities', 0)}")
                            logger.info(f"      - Total relations: {stats.get('total_relations', 0)}")
                    else:
                        logger.error(f"    ✗ {indexer_name}: Failed")

            # Verify graph store
            logger.info("\n" + "=" * 80)
            logger.info("Verifying NetworkX Graph Store...")
            logger.info("=" * 80)
            
            # Access the NetworkX indexer
            networkx_indexer = index_manager.indexers[0]
            networkx_store = networkx_indexer.networkx_store
            
            # Check graph statistics
            num_nodes = networkx_store.graph.number_of_nodes()
            num_edges = networkx_store.graph.number_of_edges()
            num_chunks = len(networkx_store.chunks)
            
            logger.info(f"  Graph statistics:")
            logger.info(f"    - Nodes: {num_nodes}")
            logger.info(f"    - Edges: {num_edges}")
            logger.info(f"    - Chunks stored: {num_chunks}")
            
            # Retrieve and display sample chunk data
            if result['chunk_ids']:
                logger.info("\n  Sample chunk graph data:")
                sample_chunks = networkx_store.get_chunks(result['chunk_ids'][:3])
                
                for i, chunk in enumerate(sample_chunks, 1):
                    logger.info(f"\n    Chunk {i} (ID: {chunk.id}):")
                    logger.info(f"      Content preview: {chunk.content[:100]}...")
                    
                    if chunk.graph:
                        logger.info(f"      Entities: {len(chunk.graph.entities)}")
                        logger.info(f"      Relations: {len(chunk.graph.relations)}")
                        
                        # Show sample entities
                        if chunk.graph.entities:
                            logger.info(f"      Sample entities:")
                            for entity in chunk.graph.entities[:3]:
                                logger.info(f"        - {entity.get('entity_name')} ({entity.get('entity_type')})")
                        
                        # Show sample relations
                        if chunk.graph.relations:
                            logger.info(f"      Sample relations:")
                            for relation in chunk.graph.relations[:3]:
                                logger.info(f"        - {relation[0]} --[{relation[1]}]--> {relation[2]}")
            
            # Check persistence
            logger.info("\n" + "=" * 80)
            logger.info("Checking persistence...")
            logger.info("=" * 80)
            
            storage_path = Path("./test_output/networkx_index_test/graph_store")
            graph_file = storage_path / "docx_graph_index_graph.pkl"
            docs_file = storage_path / "docx_graph_index_docs.pkl"
            
            if graph_file.exists() and docs_file.exists():
                logger.info(f"  ✓ Index files saved successfully:")
                logger.info(f"    - Graph file: {graph_file}")
                logger.info(f"    - Docs file: {docs_file}")
            else:
                logger.warning(f"  ⚠ Index files not found (auto_save might be disabled)")
            
            logger.info("\n" + "=" * 80)
            logger.info("Test completed successfully!")
            logger.info("=" * 80)
            
            return True
        else:
            logger.error(f"\nDOCX indexing failed: {result['error_message']}")
            return False

    except Exception as e:
        logger.error(f"Async test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("\nNetworkX Graph Indexing Test for DOCX")
    logger.info("=" * 80)
    
    # Run the test
    try:
        result = asyncio.run(test_async_networkx_indexing())
        if result:
            logger.info("\n✓ All tests passed!")
            sys.exit(0)
        else:
            logger.info("\n✗ Test failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

