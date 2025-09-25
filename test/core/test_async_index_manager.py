import os
import sys
import logging
import json
import asyncio
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
from core.file_management.index_manager import IndexManager
from config.core.file_management.index_manager_config import IndexManagerConfig
from core.file_management.file_storage import FileStorage
from config.core.file_management.file_storage_config import FileStorageConfig
from config.encapsulation.database.file_store_config import FileStoreConfig
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

from config.core.file_management.parser.standard_parser_config import StandardParserConfig
from config.core.file_management.chunker.chunker_config import TokenChunkerConfig, RecursiveChunkerConfig

from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig

from config.encapsulation.faiss import FaissConfig
from config.encapsulation.llm.huggingface_embedding import HuggingFaceEmbeddingConfig

# Test file paths
TEST_FILES = {
    'pdf': '/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/test/test_pdf.pdf',
    'docx': '/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/test/test_docx.docx',
    'xlsx': '/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/test/test_xlsx.xlsx'
}


def create_file_storage():
    """Create FileStorage instance with real database configuration"""
    logger.info("Creating FileStorage instance...")

    # Local file storage config
    file_db_config = LocalDBConfig(
        base_path="./test_output/async_index_test"
    )

    # PostgreSQL config
    postgresql_config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="rag_test",
        user="chenmingzhen",
        password=""  # No password needed for local user
    )

    # File store config
    file_store_config = FileStoreConfig(
        file_db_config=file_db_config,
        relational_db_config=postgresql_config
    )

    # File storage config
    file_storage_config = FileStorageConfig(
        data_store_config=file_store_config
    )

    return file_storage_config.build()


def create_index_manager_with_hybrid():
    """Create IndexManager with both BM25 and FAISS indexing"""
    logger.info("Creating IndexManager with hybrid (BM25 + FAISS) indexing...")

    parser_config = StandardParserConfig()
    chunker_config = RecursiveChunkerConfig(
        chunk_size=400,
        chunk_overlap=40
    )

    # BM25 indexer config
    bm25_builder_config = BM25BuilderConfig(
        index_path="./test_output/async_index_test/bm25_index"
    )
    bm25_indexer_config = BM25IndexerConfig(
        index_config=bm25_builder_config
    )



    embedding_config = HuggingFaceEmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    faiss_config = FaissConfig(
        index_path="./test_output/async_index_test/faiss_index",
        embedding_config=embedding_config
    )
    faiss_indexer_config = FaissIndexerConfig(
        index_config=faiss_config
    )

    config = IndexManagerConfig(
        parser_config=parser_config,
        chunker_config=chunker_config,
        indexer_configs=[bm25_indexer_config, faiss_indexer_config]
    )

    return config.build()


async def test_async_index_file():
    """Test the async index_file method with multiple file types"""
    logger.info("=== Testing Async index_file Method ===")

    try:
        # Create FileStorage
        file_storage = create_file_storage()

        # Create IndexManager with file_storage
        index_manager = create_index_manager_with_hybrid()
        index_manager.file_storage = file_storage

        # Content type mapping
        content_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }

        # Test each file type
        for file_type, file_path in TEST_FILES.items():
            if not os.path.exists(file_path):
                logger.warning(f" Test file not found: {file_path}")
                continue

            logger.info(f"\n Testing {file_type.upper()} file: {os.path.basename(file_path)}")

            try:
                # Read file content
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                # Upload file
                filename = f"test_async_{file_type}.{file_type}"
                content_type = content_types.get(file_type, 'application/octet-stream')

                file_id = file_storage.upload_file(
                    filename=filename,
                    file_data=file_content,
                    content_type=content_type
                )

                logger.info(f"    Uploaded: {filename} -> {file_id} ({len(file_content)} bytes)")

                # Test async index_file method
                result = await index_manager.index_file(file_id)

                if result["success"]:
                    logger.info(f"    {file_type.upper()} indexing succeeded!")
                    logger.info(f"      - Parsed content ID: {result['parsed_content_id']}")
                    logger.info(f"      - Number of chunks: {len(result['chunk_ids'])}")
                    logger.info(f"      - Parser used: {result['metadata']['parser_type']}")
                    logger.info(f"      - Chunker used: {result['metadata']['chunker_type']}")

                    # Show indexing results
                    if result['indexing_results']:
                        logger.info("      - Indexing results:")
                        for indexer_name, indexer_result in result['indexing_results'].items():
                            status = "" if indexer_result['success'] else ""
                            logger.info(f"        {status} {indexer_name}: {indexer_result.get('indexed_count', 0)} docs indexed")
                else:
                    logger.error(f"    {file_type.upper()} indexing failed: {result['error_message']}")

            except Exception as e:
                logger.error(f"    Failed to process {file_type.upper()} file: {e}")

        logger.info("\nAll file types tested!")

    except Exception as e:
        logger.error(f" Async test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_async_index_file())
