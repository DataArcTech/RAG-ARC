"""
Simple test to understand how FileStore coordinated database operations work
"""

import sys
import os
import json

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import AbstractConfig
from encapsulation.database.file_store import FileStore
from encapsulation.database.file_db.local import LocalDB
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.data_model.orm_models import (
    FileMetadata, FileStatus,
    ParsedContentMetadata, ParsedContentStatus,
    ChunkMetadata, ChunkIndexStatus
)

from typing import Literal

class LocalFileConfig(AbstractConfig):
    """Configuration for Local file storage"""
    type: Literal["local_file_store"] = "local_file_store"
    base_path: str = "./test_output"

    def build(self) -> LocalDB:
        return LocalDB(self)

class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL Database"""
    type: Literal["postgresql"] = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_arc_filestore_test"
    user: str = "postgres"
    password: str = "123"

    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)

class FileStoreConfig(AbstractConfig):
    """Configuration for FileStore"""
    type: Literal["file_store"] = "file_store"
    file_db_config: LocalFileConfig
    relational_db_config: PostgreSQLConfig

    def build(self) -> FileStore:
        return FileStore(self)

def test_file_operations(file_store: FileStore):
    """Test complete CRUD operations for Files"""
    print("\n=== FILE OPERATIONS ===")

    # Test file data
    file_data = b"This is a test document for coordinated file storage operations."
    filename = "test_document.txt"
    content_type = "text/plain"

    # Test CREATE
    print("1. Testing file storage...")
    file_metadata = file_store.store_file(
        file_data=file_data,
        filename=filename,
        content_type=content_type
    )
    print(f"  Stored file with ID: {file_metadata.file_id}")
    print(f"  Filename: {file_metadata.filename}")
    print(f"  Status: {file_metadata.status}")
    print(f"  Blob key: {file_metadata.blob_key}")

    # Test READ
    print("2. Testing file metadata retrieval...")
    retrieved_metadata = file_store.get_file_metadata(file_metadata.file_id)
    if retrieved_metadata:
        print(f"  Retrieved: {retrieved_metadata.filename} - Status: {retrieved_metadata.status}")
        print(f"  File size: {retrieved_metadata.file_size} bytes")
    else:
        print("  Failed to retrieve file metadata")

    # Test file content retrieval
    print("3. Testing file content retrieval...")
    retrieved_content = file_store.get_file_content(file_metadata.file_id)
    if retrieved_content:
        print(f"  Retrieved content length: {len(retrieved_content)} bytes")
        assert retrieved_content == file_data
        print("  Content matches original data")
    else:
        print("  Failed to retrieve file content")

    # Test validation
    print("4. Testing file upload validation...")
    is_valid = file_store.validate_file_upload(file_metadata.file_id)
    if is_valid:
        print("  File upload validation successful")
    else:
        print("  File upload validation failed")

    return file_metadata

def test_parsed_content_operations(file_store: FileStore, source_file: FileMetadata):
    """Test complete CRUD operations for Parsed Content"""
    print("\n=== PARSED CONTENT OPERATIONS ===")

    # Test parsed content data
    parsed_data = b"# Test Document\n\nThis is parsed content from the source file.\n\n## Section 1\nSome content here."
    parser_type = "markdown_parser"

    # Test CREATE
    print("1. Testing parsed content storage...")
    parsed_metadata = file_store.store_parsed_content(
        parsed_data=parsed_data,
        source_file_id=source_file.file_id,
        parser_type=parser_type,
        content_type="text/markdown"
    )
    print(f"  Stored parsed content with ID: {parsed_metadata.parsed_content_id}")
    print(f"  Source file ID: {parsed_metadata.source_file_id}")
    print(f"  Parser type: {parsed_metadata.parser_type}")
    print(f"  Status: {parsed_metadata.status}")

    # Test READ
    print("2. Testing parsed content metadata retrieval...")
    retrieved_metadata = file_store.get_parsed_content_metadata(parsed_metadata.parsed_content_id)
    if retrieved_metadata:
        print(f"  Retrieved: {retrieved_metadata.parser_type} - Status: {retrieved_metadata.status}")
        print(f"  Content type: {retrieved_metadata.content_type}")
    else:
        print("  Failed to retrieve parsed content metadata")

    # Test parsed content retrieval
    print("3. Testing parsed content retrieval...")
    retrieved_content = file_store.get_parsed_content(parsed_metadata.parsed_content_id)
    if retrieved_content:
        print(f"  Retrieved content length: {len(retrieved_content)} bytes")
        assert retrieved_content == parsed_data
        print("  Content matches original data")
    else:
        print("  Failed to retrieve parsed content")

    # Test validation
    print("4. Testing parsed content validation...")
    is_valid = file_store.validate_parsed_content_upload(parsed_metadata.parsed_content_id)
    if is_valid:
        print("  Parsed content validation successful")
    else:
        print("  Parsed content validation failed")

    return parsed_metadata

def test_chunk_operations(file_store: FileStore, source_parsed: ParsedContentMetadata):
    """Test complete CRUD operations for Chunks"""
    print("\n=== CHUNK OPERATIONS ===")

    # Test chunk data (JSON format)
    chunk_data = json.dumps({
        "chunk_id": 0,
        "content": "This is the first chunk of the parsed content.",
        "metadata": {
            "start_char": 0,
            "end_char": 46,
            "tokens": 10,
            "section": "introduction"
        }
    }).encode('utf-8')
    chunker_type = "semantic_chunker"

    # Test CREATE
    print("1. Testing chunk storage...")
    chunk_metadata = file_store.store_chunk(
        chunk_data=chunk_data,
        source_parsed_content_id=source_parsed.parsed_content_id,
        chunker_type=chunker_type
    )
    print(f"  Stored chunk with ID: {chunk_metadata.chunk_id}")
    print(f"  Source parsed content ID: {chunk_metadata.source_parsed_content_id}")
    print(f"  Chunker type: {chunk_metadata.chunker_type}")
    print(f"  Index status: {chunk_metadata.index_status}")

    # Test READ
    print("2. Testing chunk metadata retrieval...")
    retrieved_metadata = file_store.get_chunk_metadata(chunk_metadata.chunk_id)
    if retrieved_metadata:
        print(f"  Retrieved: {retrieved_metadata.chunker_type} - Status: {retrieved_metadata.index_status}")
        print(f"  Created at: {retrieved_metadata.created_at}")
    else:
        print("  Failed to retrieve chunk metadata")

    # Test chunk content retrieval
    print("3. Testing chunk content retrieval...")
    retrieved_content = file_store.get_chunk(chunk_metadata.chunk_id)
    if retrieved_content:
        print(f"  Retrieved content length: {len(retrieved_content)} bytes")
        assert retrieved_content == chunk_data
        print("  Content matches original data")

        # Parse and verify JSON structure
        chunk_json = json.loads(retrieved_content.decode('utf-8'))
        print(f"  Chunk content: {chunk_json['content'][:50]}...")
    else:
        print("  Failed to retrieve chunk content")

    # Test validation
    print("4. Testing chunk validation...")
    is_valid = file_store.validate_chunk_upload(chunk_metadata.chunk_id)
    if is_valid:
        print("  Chunk validation successful")
    else:
        print("  Chunk validation failed")

    return chunk_metadata

def test_error_handling(file_store: FileStore):
    """Test error handling for non-existent resources"""
    print("\n=== ERROR HANDLING ===")

    nonexistent_id = "00000000-0000-0000-0000-000000000000"

    # Test file operations with non-existent ID
    print("1. Testing file error handling...")
    file_metadata = file_store.get_file_metadata(nonexistent_id)
    assert file_metadata is None
    print("  Non-existent file metadata returns None")

    file_content = file_store.get_file_content(nonexistent_id)
    assert file_content is None
    print("  Non-existent file content returns None")

    # Test parsed content operations with non-existent ID
    print("2. Testing parsed content error handling...")
    parsed_metadata = file_store.get_parsed_content_metadata(nonexistent_id)
    assert parsed_metadata is None
    print("  Non-existent parsed content metadata returns None")

    parsed_content = file_store.get_parsed_content(nonexistent_id)
    assert parsed_content is None
    print("  Non-existent parsed content returns None")

    # Test chunk operations with non-existent ID
    print("3. Testing chunk error handling...")
    chunk_metadata = file_store.get_chunk_metadata(nonexistent_id)
    assert chunk_metadata is None
    print("  Non-existent chunk metadata returns None")

    chunk_content = file_store.get_chunk(nonexistent_id)
    assert chunk_content is None
    print("  Non-existent chunk content returns None")

    # Test validation with non-existent IDs
    print("4. Testing validation error handling...")
    file_valid = file_store.validate_file_upload(nonexistent_id)
    assert file_valid is False
    print("  Non-existent file validation returns False")

    parsed_valid = file_store.validate_parsed_content_upload(nonexistent_id)
    assert parsed_valid is False
    print("  Non-existent parsed content validation returns False")

    chunk_valid = file_store.validate_chunk_upload(nonexistent_id)
    assert chunk_valid is False
    print("  Non-existent chunk validation returns False")

def test_cleanup_operations(file_store: FileStore):
    """Test cleanup by creating and deleting test data"""
    print("\n=== CLEANUP OPERATIONS ===")

    # Create test data chain
    print("1. Creating test data for cleanup...")

    # Create file
    file_data = b"Test file for cleanup operations"
    file_metadata = file_store.store_file(
        file_data=file_data,
        filename="cleanup_test.txt",
        content_type="text/plain"
    )
    print(f"  Created file: {file_metadata.file_id}")

    # Create parsed content
    parsed_data = b"# Cleanup Test\n\nParsed content for cleanup testing."
    parsed_metadata = file_store.store_parsed_content(
        parsed_data=parsed_data,
        source_file_id=file_metadata.file_id,
        parser_type="cleanup_parser"
    )
    print(f"  Created parsed content: {parsed_metadata.parsed_content_id}")

    # Create chunk
    chunk_data = json.dumps({
        "chunk_id": 0,
        "content": "Cleanup test chunk",
        "metadata": {"test": True}
    }).encode('utf-8')
    chunk_metadata = file_store.store_chunk(
        chunk_data=chunk_data,
        source_parsed_content_id=parsed_metadata.parsed_content_id,
        chunker_type="cleanup_chunker"
    )
    print(f"  Created chunk: {chunk_metadata.chunk_id}")

    # Verify all exist
    assert file_store.get_file_metadata(file_metadata.file_id) is not None
    assert file_store.get_parsed_content_metadata(parsed_metadata.parsed_content_id) is not None
    assert file_store.get_chunk_metadata(chunk_metadata.chunk_id) is not None
    print("  All test data verified to exist")

    # Test cleanup in reverse order (chunk -> parsed -> file)
    print("2. Testing cleanup operations...")

    # Cleanup chunk
    chunk_cleanup = file_store.cleanup_failed_chunk_upload(chunk_metadata.chunk_id)
    assert chunk_cleanup is True
    assert file_store.get_chunk_metadata(chunk_metadata.chunk_id) is None
    print("  Chunk cleanup successful")

    # Cleanup parsed content
    parsed_cleanup = file_store.cleanup_failed_parsed_content_upload(parsed_metadata.parsed_content_id)
    assert parsed_cleanup is True
    assert file_store.get_parsed_content_metadata(parsed_metadata.parsed_content_id) is None
    print("  Parsed content cleanup successful")

    # Cleanup file
    file_cleanup = file_store.cleanup_failed_file_upload(file_metadata.file_id)
    assert file_cleanup is True
    assert file_store.get_file_metadata(file_metadata.file_id) is None
    print("  File cleanup successful")

def main():
    print("Testing FileStore Operations with coordinated blob and metadata storage...")

    # Use configured base path instead of temporary directory
    file_db_config = LocalFileConfig()
    base_path = file_db_config.base_path
    print(f"Using blob storage directory: {base_path}")

    try:
        # Create file store configuration
        relational_db_config = PostgreSQLConfig()
        config = FileStoreConfig(
            file_db_config=file_db_config,
            relational_db_config=relational_db_config
        )

        # Create file store instance
        file_store = config.build()

        print(f"FileStore initialized:")
        print(f"  Blob storage: {base_path}")
        print(f"  Database: {relational_db_config.host}:{relational_db_config.port}/{relational_db_config.database}")

        # Drop and recreate all tables to ensure schema is up-to-date
        print("\n=== SCHEMA SETUP ===")
        print("Dropping and recreating all tables...")
        from encapsulation.data_model.orm_models import Base
        Base.metadata.drop_all(file_store.metadata_store.engine)
        Base.metadata.create_all(file_store.metadata_store.engine)
        print("  Tables recreated with latest schema")

        # Run tests
        print("\n" + "="*60)
        print("STARTING FILESTORE TESTS")
        print("="*60)

        # Test file operations
        file_metadata = test_file_operations(file_store)

        # Test parsed content operations (depends on file)
        parsed_metadata = test_parsed_content_operations(file_store, file_metadata)

        # Test chunk operations (depends on parsed content)
        chunk_metadata = test_chunk_operations(file_store, parsed_metadata)

        # Test error handling
        test_error_handling(file_store)

        # Show final state
        print("\n=== FINAL STATE VERIFICATION ===")
        final_file = file_store.get_file_metadata(file_metadata.file_id)
        final_parsed = file_store.get_parsed_content_metadata(parsed_metadata.parsed_content_id)
        final_chunk = file_store.get_chunk_metadata(chunk_metadata.chunk_id)

        print(f"File status: {final_file.status} (type: {type(final_file.status)})")
        print(f"Parsed status: {final_parsed.status} (type: {type(final_parsed.status)})")
        print(f"Chunk status: {final_chunk.index_status} (type: {type(final_chunk.index_status)})")

        # Test cleanup operations (commented out to preserve test data)
        # test_cleanup_operations(file_store)

        print("\nAll FileStore tests completed successfully!")
        print(f"\nFiles preserved in: {base_path}")
        print("You can inspect the stored files and directory structure")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()