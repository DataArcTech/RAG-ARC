import os
import sys
import logging
import asyncio
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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
from config.core.file_management.parser.native import NativeParserConfig
from config.core.file_management.parser.vlm_ocr import VLMOcrParserConfig
from config.core.file_management.chunker.chunker_config import RecursiveChunkerConfig

from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig

from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.parse.vlm_ocr import VLMOcrConfig

# Test file paths (using Native Parser supported formats)
TEST_FILES = {
    'pdf': 'test/test_pdf.pdf',
    # 'html': 'test/test2.html', 
    # 'docx': 'test/test_docx.docx',
    # 'xlsx': 'test/test_xlsx.xlsx'
}




async def test_async_index_file():
    """Test the async index_file method with multiple file types"""
    logger.info("=== Testing Async index_file Method ===")

    try:
        # Create all storage instances
        file_db_config = LocalDBConfig(
            base_path="./test_output/async_index_test"
        )

        # Use PostgreSQLConfig with environment variables (from .env)
        postgresql_config = PostgreSQLConfig()

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

        # Build file storage instance (others will be built by IndexManager)
        file_storage = file_storage_config.build()

        # Configure Native parser (no API calls, faster and more reliable for testing)
        native_parser_config = NativeParserConfig()
        ocr_parser_config = VLMOcrParserConfig(
            vlm_ocr=VLMOcrConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
                model_name="gpt-4o-mini"
            )
        )

        parser_config = ParserCombinatorConfig(
            base_output_dir="./test_output/async_index_test/parsed_files",
            native_parser=native_parser_config,
            ocr_parser=ocr_parser_config
        )
        # Configure chunker
        chunker_config = RecursiveChunkerConfig(
            chunk_size=400,
            chunk_overlap=40
        )

        # Configure BM25 indexer
        bm25_builder_config = BM25BuilderConfig(
            index_path="./test_output/async_index_test/bm25_index"
        )
        bm25_indexer_config = BM25IndexerConfig(
            index_config=bm25_builder_config
        )

        # Configure FAISS indexer with embedding
        embedding_config = QwenEmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_china_mirror=True,
            cache_folder="./models"
        )
        faiss_config = FaissVectorDBConfig(
            index_path="./test_output/async_index_test/faiss_index",
            embedding_config=embedding_config
        )
        faiss_indexer_config = FaissIndexerConfig(
            index_config=faiss_config
        )

        # Create IndexManager config
        config = IndexManagerConfig(
            parser_config=parser_config,
            chunker_config=chunker_config,
            indexer_configs=[bm25_indexer_config, faiss_indexer_config],
            file_storage_config=file_storage_config,
            chunk_storage_config=chunk_storage_config,
            parsed_content_storage_config=parsed_content_storage_config
        )

        # Build IndexManager
        index_manager = config.build()

        # Create test user in database
        test_user_id = str(uuid.uuid4())
        test_username = f"test_user_{int(asyncio.get_event_loop().time())}"

        # Insert user into database (using correct column names: id, user_name, hashed_password)
        with postgresql_config.build().engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(
                text("INSERT INTO \"user\" (id, user_name, hashed_password, created_at, updated_at) VALUES (:id, :user_name, :hashed_password, NOW(), NOW()) ON CONFLICT DO NOTHING"),
                {"id": test_user_id, "user_name": test_username, "hashed_password": "test_password_hash"}
            )
            conn.commit()

        logger.info(f"✓ Created test user: {test_username} (ID: {test_user_id})")

        # Content type mapping
        content_types = {
            'pdf': 'application/pdf',
            'html': 'text/html',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }

        # Test each file type
        for file_type, file_path in TEST_FILES.items():
            if not os.path.exists(file_path):
                logger.warning(f"⚠️  Test file not found: {file_path}")
                continue

            logger.info(f"\n📄 Testing {file_type.upper()} file: {os.path.basename(file_path)}")

            try:
                # Read file content
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                # Upload file with owner_id and unique filename
                # Use timestamp to make filename unique
                import time
                timestamp = int(time.time() * 1000)
                filename = f"test_async_{file_type}_{timestamp}.{file_type}"
                content_type = content_types.get(file_type, 'application/octet-stream')

                file_id = file_storage.upload_file(
                    filename=filename,
                    file_data=file_content,
                    content_type=content_type,
                    owner_id=test_user_id
                )

                logger.info(f"  ✓ Uploaded: {filename} -> {file_id} ({len(file_content)} bytes)")

                # Test async index_file method
                logger.info(f"  🔄 Starting async indexing...")
                result = await index_manager.index_file(file_id)

                if result["success"]:
                    logger.info(f"  ✅ {file_type.upper()} indexing succeeded!")
                    logger.info(f"    - Parsed content ID: {result['parsed_content_id']}")
                    logger.info(f"    - Number of chunks: {len(result['chunk_ids'])}")
                    logger.info(f"    - Parser used: {result['metadata']['parser_type']}")
                    logger.info(f"    - Chunker used: {result['metadata']['chunker_type']}")

                    # Show indexing results
                    if result['indexing_results']:
                        logger.info("    - Indexing results:")
                        for indexer_name, indexer_result in result['indexing_results'].items():
                            status = "✅" if indexer_result['success'] else "❌"
                            logger.info(f"      {status} {indexer_name}: {indexer_result.get('indexed_count', 0)} docs indexed")
                else:
                    logger.error(f"  ❌ {file_type.upper()} indexing failed: {result['error_message']}")

            except Exception as e:
                logger.error(f"  ❌ Failed to process {file_type.upper()} file: {e}", exc_info=True)

        logger.info("\n🎉 All file types tested!")

    except Exception as e:
        logger.error(f"❌ Async test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_async_index_file())
