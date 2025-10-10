"""
Test script for file deletion functionality in IndexManager
"""
import sys
import os
import tempfile
import shutil
import json
from dotenv import load_dotenv
load_dotenv() 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.core.file_management.index_manager_config import IndexManagerConfig
from config.core.file_management.storage.file_storage import FileStorageConfig
from config.core.file_management.storage.parsed_content_storage import ParsedContentStorageConfig
from config.core.file_management.storage.chunk_storage import ChunkStorageConfig
from config.core.file_management.parser.native import NativeParserConfig
from config.core.file_management.parser_combinator_config import ParserCombinatorConfig
from config.core.file_management.chunker.chunker_config import TokenChunkerConfig
from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig
from config.core.file_management.indexing.graph_indexing.networkx_indexing_config import NetworkXGraphIndexerConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.database.graph_db.networkx_config import NetworkXConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

def test_file_deletion():
    """Test file deletion functionality"""
    print("=" * 80)
    print("Testing File Deletion Functionality")
    print("=" * 80)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    file_db_path = os.path.join(temp_dir, "file_db")
    parsed_db_path = os.path.join(temp_dir, "parsed_db")
    chunk_db_path = os.path.join(temp_dir, "chunk_db")
    bm25_index_path = os.path.join(temp_dir, "bm25_index")
    faiss_index_path = os.path.join(temp_dir, "faiss_index")
    graph_index_path = os.path.join(temp_dir, "graph_index")

    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # 1. Setup configurations
        print("\n1. Setting up configurations...")
        
        # Database configs
        file_db_config = LocalDBConfig(storage_path=file_db_path)
        parsed_db_config = LocalDBConfig(storage_path=parsed_db_path)
        chunk_db_config = LocalDBConfig(storage_path=chunk_db_path)

        pg_config = PostgreSQLConfig(
            host="127.0.0.1",
            port=18080, 
            database="rag_arc_filestore_test",
            user="postgres",
            password="123"
        )
        
        # Storage configs
        file_storage_config = FileStorageConfig(
            file_db_config=file_db_config,
            relational_db_config=pg_config
        )
        parsed_content_storage_config = ParsedContentStorageConfig(
            file_db_config=parsed_db_config,
            relational_db_config=pg_config
        )
        chunk_storage_config = ChunkStorageConfig(
            file_db_config=chunk_db_config,
            relational_db_config=pg_config
        )
        
        # Build storage instances
        file_storage = file_storage_config.build()
        parsed_content_storage = parsed_content_storage_config.build()
        chunk_storage = chunk_storage_config.build()

        # Setup database schema
        print("  Setting up database schema...")
        from encapsulation.data_model.orm_models import Base
        Base.metadata.drop_all(file_storage.metadata_store.engine)
        Base.metadata.create_all(file_storage.metadata_store.engine)
        print("  Database schema recreated")
        
        # Parser and chunker configs
        native_parser_config = NativeParserConfig()
        parser_config = ParserCombinatorConfig(parser=native_parser_config)

        # Use TokenChunker to avoid embedding dependencies
        chunker_config = TokenChunkerConfig(
            chunk_size=500,
            chunk_overlap=50
        )

        # Embedding config for FAISS
        embedding_config = QwenEmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_china_mirror=True,
            cache_folder="./models"
        )

        # Indexer configs - BM25, FAISS, and NetworkX Graph
        bm25_config = BM25BuilderConfig(index_path=bm25_index_path)
        bm25_indexer_config = BM25IndexerConfig(index_config=bm25_config)

        faiss_config = FaissVectorDBConfig(
            index_path=faiss_index_path,
            embedding_config=embedding_config,
            index_type="flat",
            metric="cosine"
        )
        faiss_indexer_config = FaissIndexerConfig(index_config=faiss_config)

        # LLM config for Graph extractor
        llm_config = OpenAIChatConfig(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        # Graph extractor and store config
        graph_extractor_config = GraphExtractorConfig(
            llm_config=llm_config
        )
        graph_store_config = NetworkXConfig(
            storage_path=graph_index_path,
            index_name="test_graph"
        )
        graph_indexer_config = NetworkXGraphIndexerConfig(
            extractor_config=graph_extractor_config,
            graph_store_config=graph_store_config
        )

        # IndexManager config - use BM25, FAISS, and Graph
        index_manager_config = IndexManagerConfig(
            parser_config=parser_config,
            chunker_config=chunker_config,
            indexer_configs=[bm25_indexer_config, faiss_indexer_config, graph_indexer_config]
        )

        # Build IndexManager with storage instances
        from core.file_management.index_manager import IndexManager
        index_manager = IndexManager(
            config=index_manager_config,
            file_storage=file_storage,
            parsed_content_storage=parsed_content_storage,
            chunk_storage=chunk_storage
        )
        
        print("Configurations set up successfully")
        
        # 2. Use existing test docx file and create 3 copies
        print("\n2. Loading and storing 3 test files...")
        test_docx_path = "/home/dataarc/chenmingzhen/RAG-ARC/test/test_docx.docx"

        with open(test_docx_path, 'rb') as f:
            file_content = f.read()

        file_ids = []
        for i in range(3):
            file_id = file_storage.upload_file(
                filename=f"test_deletion_{i+1}.docx",
                file_data=file_content,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            file_ids.append(file_id)
            print(f"  Stored test file {i+1} with ID: {file_id}")

        print(f"Stored {len(file_ids)} test files")
        
        # 3. Index all files
        print("\n3. Indexing all files...")
        all_parsed_content_ids = []
        all_chunk_ids_by_file = {}

        for i, file_id in enumerate(file_ids):
            print(f"\n  Indexing file {i+1}/{len(file_ids)}...")
            index_result = index_manager.process_file(
                file_id=file_id,
                file_storage=file_storage,
                parsed_content_storage=parsed_content_storage,
                chunk_storage=chunk_storage
            )

            if not index_result["success"]:
                raise Exception(f"Indexing failed for file {i+1}: {index_result.get('error_message')}")

            parsed_content_id = index_result["parsed_content_id"]
            chunk_ids = index_result["chunk_ids"]

            all_parsed_content_ids.append(parsed_content_id)
            all_chunk_ids_by_file[file_id] = chunk_ids

            print(f"  Indexed file {i+1} successfully")
            print(f"    Parsed content ID: {parsed_content_id}")
            print(f"    Number of chunks: {len(chunk_ids)}")

        total_chunks = sum(len(chunks) for chunks in all_chunk_ids_by_file.values())
        print(f"\nIndexed all {len(file_ids)} files successfully")
        print(f"  Total parsed content: {len(all_parsed_content_ids)}")
        print(f"  Total chunks: {total_chunks}")
        
        # 4. Verify data exists before deletion
        print("\n4. Verifying data exists before deletion...")

        for i, file_id in enumerate(file_ids):
            file_metadata = file_storage.get_file_metadata(file_id)
            assert file_metadata is not None, f"File {i+1} metadata should exist"
        print(f"  All {len(file_ids)} file metadata exist")

        for i, parsed_content_id in enumerate(all_parsed_content_ids):
            parsed_metadata = parsed_content_storage.get_parsed_content_metadata(parsed_content_id)
            assert parsed_metadata is not None, f"Parsed content {i+1} metadata should exist"
        print(f"  All {len(all_parsed_content_ids)} parsed content metadata exist")

        for file_id, chunk_ids in all_chunk_ids_by_file.items():
            for chunk_id in chunk_ids:
                chunk_metadata = chunk_storage.get_chunk_metadata(chunk_id)
                assert chunk_metadata is not None, f"Chunk metadata should exist for {chunk_id}"
        print(f"  All {total_chunks} chunk metadata exist")
        
        # 5. Delete the first file (parsed content, chunks and index entries)
        # Deletion order: indexers -> chunks -> parsed_data
        print("\n5. Deleting the first file...")
        file_to_delete = file_ids[0]
        delete_result = index_manager.delete_file_data(
            file_id=file_to_delete,
            parsed_content_storage=parsed_content_storage,
            chunk_storage=chunk_storage
        )

        if not delete_result["success"]:
            raise Exception(f"Deletion failed: {delete_result.get('error_message')}")

        print(f"Deleted file 1 successfully")
        print(f"  Deleted {len(delete_result['deleted_parsed_content_ids'])} parsed content")
        print(f"  Deleted {len(delete_result['deleted_chunk_ids'])} chunks")
        print(f"  Indexer deletion results:")
        for indexer_name, result in delete_result['indexer_deletion_results'].items():
            print(f"    - {indexer_name}: {result}")
        
        # 6. Verify first file's data is deleted
        print("\n6. Verifying first file's data is deleted...")

        # Check file (should still exist since we didn't delete it)
        file_metadata = file_storage.get_file_metadata(file_to_delete)
        assert file_metadata is not None, "File metadata should still exist"
        print(f"File metadata still exists (as expected)")

        # Check parsed content (should be deleted)
        deleted_parsed_content_id = all_parsed_content_ids[0]
        parsed_metadata = parsed_content_storage.get_parsed_content_metadata(deleted_parsed_content_id)
        assert parsed_metadata is None, "Parsed content metadata should be deleted"
        print(f"Parsed content metadata deleted")

        # Check chunks of deleted file (should be deleted)
        deleted_chunk_ids = all_chunk_ids_by_file[file_to_delete]
        for chunk_id in deleted_chunk_ids:
            chunk_metadata = chunk_storage.get_chunk_metadata(chunk_id)
            assert chunk_metadata is None, f"Chunk metadata should be deleted for {chunk_id}"
        print(f"All {len(deleted_chunk_ids)} chunk metadata of file 1 deleted")

        # Check chunks of other files (should still exist)
        other_files_chunks = 0
        for i, file_id in enumerate(file_ids[1:], start=2):
            for chunk_id in all_chunk_ids_by_file[file_id]:
                chunk_metadata = chunk_storage.get_chunk_metadata(chunk_id)
                assert chunk_metadata is not None, f"Chunk metadata of file {i} should still exist"
                other_files_chunks += 1
        print(f"All {other_files_chunks} chunk metadata of other files still exist")

        # 7. Test the synchronous delete_file method with the second file
        print("\n7. Testing synchronous delete_file method with the second file...")
        file_to_delete_2 = file_ids[1]
        delete_result_2 = index_manager.delete_file(file_to_delete_2)

        if not delete_result_2["success"]:
            raise Exception(f"Deletion via delete_file failed: {delete_result_2.get('error_message')}")

        print(f"Deleted file 2 successfully using delete_file method")
        print(f"  Deleted {len(delete_result_2['deleted_parsed_content_ids'])} parsed content")
        print(f"  Deleted {len(delete_result_2['deleted_chunk_ids'])} chunks")
        print(f"  Indexer deletion results:")
        for indexer_name, result in delete_result_2['indexer_deletion_results'].items():
            print(f"    - {indexer_name}: {result}")

        # Verify second file's data is deleted
        deleted_parsed_content_id_2 = all_parsed_content_ids[1]
        parsed_metadata_2 = parsed_content_storage.get_parsed_content_metadata(deleted_parsed_content_id_2)
        assert parsed_metadata_2 is None, "Parsed content metadata of file 2 should be deleted"
        print(f"Verified file 2's parsed content metadata deleted")

        deleted_chunk_ids_2 = all_chunk_ids_by_file[file_to_delete_2]
        for chunk_id in deleted_chunk_ids_2:
            chunk_metadata = chunk_storage.get_chunk_metadata(chunk_id)
            assert chunk_metadata is None, f"Chunk metadata of file 2 should be deleted for {chunk_id}"
        print(f"Verified all {len(deleted_chunk_ids_2)} chunk metadata of file 2 deleted")

        print("\n✓ All tests passed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_file_deletion()

