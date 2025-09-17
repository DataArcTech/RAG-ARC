"""
Test for FileStore - testing coordinated file and metadata storage operations
"""

import tempfile
import shutil
import os
import pytest
from io import BytesIO
from typing import Literal

from framework.config import AbstractConfig
from encapsulation.database.file_store import FileStore
from encapsulation.database.file_db.local import LocalDB
from encapsulation.database.relational_db.postgresql import PostgreSQLDB


class LocalFileConfig(AbstractConfig):
    """Configuration for Local file storage testing"""
    type: Literal["local_file_store"] = "local_file_store"
    base_path: str = "/root/RAG-ARC/test_file"
    cleanup_empty_dirs: bool = True
    
    def build(self) -> LocalDB:
        return LocalDB(self)


class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL testing"""
    type: Literal["postgresql_store"] = "postgresql_store"
    host: str = "localhost"
    port: int = 5432
    database: str = "test_filestore_db"
    user: str = "postgres"
    password: str = "123"
    pool_size: int = 5      # Optional
    max_overflow: int = 10  # Optional
    echo_sql: bool = False  # Optional
    
    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)


class FileStoreConfig(AbstractConfig):
    """Configuration for FileStore testing"""
    type: Literal["file_store"] = "file_store"
    blob_store_config: LocalFileConfig
    relational_db_config: PostgreSQLConfig
    
    def build(self) -> FileStore:
        return FileStore(self)


def create_test_config() -> FileStoreConfig:
    """Create test configuration for file store"""
    temp_dir = tempfile.mkdtemp(prefix="filestore_test_")
    
    blob_config = LocalFileConfig(base_path=temp_dir)
    db_config = PostgreSQLConfig()
    
    return FileStoreConfig(
        blob_store_config=blob_config,
        relational_db_config=db_config
    ), temp_dir


@pytest.fixture
def file_store():
    """Create FileStore instance for testing"""
    config, temp_dir = create_test_config()
    
    try:
        store = config.build()
        yield store
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Sample test data"""
    return {
        "text_data": b"Hello, this is test content for coordinated file storage!",
        "binary_data": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",
        "json_data": b'{"test": "coordinated storage", "number": 42}',
        "markdown_data": b"# Test Document\n\nThis is a **test** document for parsed content.",
        "chunks_data": b'[{"chunk_id": 1, "content": "First chunk of text", "metadata": {"start": 0, "end": 20}}, {"chunk_id": 2, "content": "Second chunk of text", "metadata": {"start": 21, "end": 42}}]',
        "chunks_data_large": b'[{"chunk_id": 1, "content": "Large chunk 1 with more content", "metadata": {"start": 0, "end": 32}}, {"chunk_id": 2, "content": "Large chunk 2 with even more content", "metadata": {"start": 33, "end": 70}}, {"chunk_id": 3, "content": "Large chunk 3 final section", "metadata": {"start": 71, "end": 99}}]',
    }


class TestFileStore:
    """Test cases for coordinated file storage"""
    
    def test_store_and_retrieve_file(self, file_store, sample_data):
        """Test storing and retrieving file with metadata coordination"""
        print("\n--- Testing coordinated file storage ---")
        filename = "test_document.txt"
        file_data = sample_data["text_data"]
        content_type = "text/plain"
        original_path = "/home/user/documents/test_document.txt"
        
        # Store file (coordinates blob + metadata)
        metadata = file_store.store_file(
            file_data=file_data,
            filename=filename,
            content_type=content_type,
            original_path=original_path
        )
        
        print(f"✓ Stored file with asset_id: {metadata.asset_id}")
        print(f"  - Filename: {metadata.filename}")
        print(f"  - Status: {metadata.status}")
        print(f"  - Blob key: {metadata.blob_key}")
        print(f"  - File size: {metadata.file_size}")
        print(f"  - Checksum: {metadata.checksum}")
        
        # Verify metadata was stored
        retrieved_metadata = file_store.get_file_metadata(metadata.asset_id)
        assert retrieved_metadata is not None
        assert retrieved_metadata.asset_id == metadata.asset_id
        assert retrieved_metadata.filename == filename
        assert retrieved_metadata.content_type == content_type
        assert retrieved_metadata.original_path == original_path
        assert retrieved_metadata.file_size == len(file_data)
        print("✓ Metadata correctly stored and retrieved")
        
        # Verify file content was stored
        retrieved_content = file_store.get_file_content(metadata.asset_id)
        assert retrieved_content is not None
        assert retrieved_content == file_data
        print("✓ File content correctly stored and retrieved")
        
        # Validate upload
        is_valid = file_store.validate_file_upload(metadata.asset_id)
        assert is_valid is True
        print("✓ File upload validation passed")
    
    def test_store_file_stream(self, file_store, sample_data):
        """Test storing file from stream with metadata coordination"""
        print("\n--- Testing coordinated file stream storage ---")
        filename = "stream_test.bin"
        file_data = sample_data["binary_data"]
        content_type = "application/octet-stream"
        
        # Create stream
        file_stream = BytesIO(file_data)
        
        # Store file stream
        metadata = file_store.store_file_stream(
            file_stream=file_stream,
            filename=filename,
            content_type=content_type
        )
        
        print(f"✓ Stored file stream with asset_id: {metadata.asset_id}")
        
        # Verify content
        retrieved_content = file_store.get_file_content(metadata.asset_id)
        assert retrieved_content == file_data
        print("✓ Stream content correctly stored and retrieved")
    
    def test_store_parsed_content(self, file_store, sample_data):
        """Test storing parsed content with metadata coordination"""
        print("\n--- Testing coordinated parsed content storage ---")
        
        # First store a source file
        source_filename = "source_document.pdf"
        source_data = sample_data["binary_data"]
        
        source_metadata = file_store.store_file(
            file_data=source_data,
            filename=source_filename,
            content_type="application/pdf"
        )
        print(f"✓ Stored source file: {source_metadata.asset_id}")
        
        # Now store parsed content
        parsed_data = sample_data["markdown_data"]
        parser_type = "dots_ocr"
        parser_version = "1.0.0"
        
        parsed_metadata = file_store.store_parsed_content(
            parsed_data=parsed_data,
            source_asset_id=source_metadata.asset_id,
            parser_type=parser_type,
            parser_version=parser_version,
            content_type="text/markdown",
            page_count=1,
            language="en"
        )
        
        print(f"✓ Stored parsed content: {parsed_metadata.parsed_content_id}")
        print(f"  - Source asset: {parsed_metadata.source_asset_id}")
        print(f"  - Parser: {parsed_metadata.parser_type} v{parsed_metadata.parser_version}")
        print(f"  - Blob key: {parsed_metadata.blob_key}")
        
        # Verify parsed content metadata
        retrieved_parsed_metadata = file_store.get_parsed_content_metadata(parsed_metadata.parsed_content_id)
        assert retrieved_parsed_metadata is not None
        assert retrieved_parsed_metadata.source_asset_id == source_metadata.asset_id
        assert retrieved_parsed_metadata.parser_type == parser_type
        assert retrieved_parsed_metadata.parser_version == parser_version
        print("✓ Parsed content metadata correctly stored")
        
        # Verify parsed content
        retrieved_parsed_content = file_store.get_parsed_content(parsed_metadata.parsed_content_id)
        assert retrieved_parsed_content == parsed_data
        print("✓ Parsed content correctly stored and retrieved")
        
        # Validate parsed content upload
        is_valid = file_store.validate_parsed_content_upload(parsed_metadata.parsed_content_id)
        assert is_valid is True
        print("✓ Parsed content upload validation passed")

    def test_store_chunks(self, file_store, sample_data):
        """Test storing chunks with metadata coordination"""
        print("\n--- Testing coordinated chunks storage ---")

        # First store a source file
        source_filename = "source_document.pdf"
        source_data = sample_data["binary_data"]

        source_metadata = file_store.store_file(
            file_data=source_data,
            filename=source_filename,
            content_type="application/pdf"
        )
        print(f"✓ Stored source file: {source_metadata.asset_id}")

        # Store parsed content from the source file
        parsed_data = sample_data["markdown_data"]
        parser_type = "dots_ocr"
        parser_version = "1.0.0"

        parsed_metadata = file_store.store_parsed_content(
            parsed_data=parsed_data,
            source_asset_id=source_metadata.asset_id,
            parser_type=parser_type,
            parser_version=parser_version,
            content_type="text/markdown"
        )
        print(f"✓ Stored parsed content: {parsed_metadata.parsed_content_id}")

        # Now store chunks from the parsed content
        chunks_data = sample_data["chunks_data"]
        chunking_strategy = "fixed_512"
        chunking_version = "1.2.0"
        processing_time_ms = 150

        chunks_metadata = file_store.store_chunks(
            chunks_data=chunks_data,
            source_parsed_content_id=parsed_metadata.parsed_content_id,
            chunking_strategy=chunking_strategy,
            chunking_version=chunking_version,
            content_type="application/json",
            processing_time_ms=processing_time_ms,
            chunking_config='{"chunk_size": 512, "overlap": 50}'
        )

        print(f"✓ Stored chunks: {chunks_metadata.chunks_id}")
        print(f"  - Source parsed content: {chunks_metadata.source_parsed_content_id}")
        print(f"  - Chunking strategy: {chunks_metadata.chunking_strategy}")
        print(f"  - Chunks count: {chunks_metadata.chunks_count}")
        print(f"  - Blob key: {chunks_metadata.blob_key}")
        print(f"  - Processing time: {chunks_metadata.processing_time_ms}ms")

        # Verify chunks metadata
        retrieved_chunks_metadata = file_store.get_chunks_metadata(chunks_metadata.chunks_id)
        assert retrieved_chunks_metadata is not None
        assert retrieved_chunks_metadata.source_parsed_content_id == parsed_metadata.parsed_content_id
        assert retrieved_chunks_metadata.chunking_strategy == chunking_strategy
        assert retrieved_chunks_metadata.chunking_version == chunking_version
        assert retrieved_chunks_metadata.chunks_count == 2  # Should be calculated from test data
        assert retrieved_chunks_metadata.processing_time_ms == processing_time_ms
        print("✓ Chunks metadata correctly stored")

        # Verify chunks content
        retrieved_chunks_content = file_store.get_chunks_content(chunks_metadata.chunks_id)
        assert retrieved_chunks_content == chunks_data
        print("✓ Chunks content correctly stored and retrieved")

        # Validate chunks upload
        is_valid = file_store.validate_chunks_upload(chunks_metadata.chunks_id)
        assert is_valid is True
        print("✓ Chunks upload validation passed")

    def test_store_chunks_stream(self, file_store, sample_data):
        """Test storing chunks from stream with metadata coordination"""
        print("\n--- Testing coordinated chunks stream storage ---")

        # First create the prerequisite chain: file -> parsed content
        source_metadata = file_store.store_file(
            file_data=sample_data["binary_data"],
            filename="stream_source.pdf",
            content_type="application/pdf"
        )

        parsed_metadata = file_store.store_parsed_content(
            parsed_data=sample_data["markdown_data"],
            source_asset_id=source_metadata.asset_id,
            parser_type="test_parser",
            parser_version="1.0.0"
        )
        print(f"✓ Created prerequisite chain: file -> parsed content")

        # Create chunks stream
        chunks_data = sample_data["chunks_data_large"]
        chunks_stream = BytesIO(chunks_data)

        # Store chunks stream
        chunks_metadata = file_store.store_chunks_stream(
            chunks_stream=chunks_stream,
            source_parsed_content_id=parsed_metadata.parsed_content_id,
            chunking_strategy="semantic_0.8",
            chunking_version="2.0.0",
            processing_time_ms=300
        )

        print(f"✓ Stored chunks stream: {chunks_metadata.chunks_id}")

        # Verify content
        retrieved_content = file_store.get_chunks_content(chunks_metadata.chunks_id)
        assert retrieved_content == chunks_data
        print("✓ Stream chunks content correctly stored and retrieved")

    def test_chunks_validation_and_cleanup(self, file_store, sample_data):
        """Test chunks validation and cleanup operations"""
        print("\n--- Testing chunks validation and cleanup ---")

        # Create the full chain: file -> parsed content -> chunks
        source_metadata = file_store.store_file(
            file_data=sample_data["text_data"],
            filename="validation_source.txt",
            content_type="text/plain"
        )

        parsed_metadata = file_store.store_parsed_content(
            parsed_data=sample_data["markdown_data"],
            source_asset_id=source_metadata.asset_id,
            parser_type="validation_parser",
            parser_version="1.0.0"
        )

        chunks_metadata = file_store.store_chunks(
            chunks_data=sample_data["chunks_data"],
            source_parsed_content_id=parsed_metadata.parsed_content_id,
            chunking_strategy="test_validation",
            chunking_version="1.0.0"
        )
        print(f"✓ Created full storage chain for validation test")

        # Test validation
        is_valid = file_store.validate_chunks_upload(chunks_metadata.chunks_id)
        assert is_valid is True
        print("✓ Chunks validation successful")

        # Verify chunks exist before cleanup
        assert file_store.get_chunks_metadata(chunks_metadata.chunks_id) is not None
        assert file_store.get_chunks_content(chunks_metadata.chunks_id) is not None
        print("✓ Chunks verified to exist before cleanup")

        # Test cleanup
        cleanup_success = file_store.cleanup_failed_chunks_upload(chunks_metadata.chunks_id)
        assert cleanup_success is True
        print("✓ Chunks cleanup successful")

        # Verify chunks are completely removed
        assert file_store.get_chunks_metadata(chunks_metadata.chunks_id) is None
        assert file_store.get_chunks_content(chunks_metadata.chunks_id) is None
        print("✓ Chunks completely removed after cleanup")
    
    
    
    def test_upload_status_tracking(self, file_store, sample_data):
        """Test upload status tracking throughout the process"""
        print("\n--- Testing upload status tracking ---")
        filename = "status_test.txt"
        file_data = sample_data["text_data"]
        
        # Store file
        metadata = file_store.store_file(
            file_data=file_data,
            filename=filename,
            content_type="text/plain"
        )
        
        # Check status
        status = file_store.get_upload_status(metadata.asset_id)
        print(f"✓ Upload status: {status}")
        assert status in ["uploaded", "uploading"]  # Should be "uploaded" after successful store
        
        # Validate upload
        is_valid = file_store.validate_file_upload(metadata.asset_id)
        assert is_valid is True
        print("✓ Upload validation successful")
    
    def test_error_handling_nonexistent_files(self, file_store):
        """Test error handling for non-existent files"""
        print("\n--- Testing error handling ---")
        nonexistent_id = "00000000-0000-0000-0000-000000000000"
        
        # Test metadata retrieval for non-existent file
        metadata = file_store.get_file_metadata(nonexistent_id)
        assert metadata is None
        print("✓ Non-existent file metadata returns None")
        
        # Test content retrieval for non-existent file
        content = file_store.get_file_content(nonexistent_id)
        assert content is None
        print("✓ Non-existent file content returns None")
        
        # Test status for non-existent file
        status = file_store.get_upload_status(nonexistent_id)
        assert status is None
        print("✓ Non-existent file status returns None")
    
    def test_parsed_content_error_handling(self, file_store, sample_data):
        """Test error handling for parsed content with invalid source"""
        print("\n--- Testing parsed content error handling ---")
        nonexistent_source_id = "00000000-0000-0000-0000-000000000000"
        parsed_data = sample_data["markdown_data"]
        
        try:
            file_store.store_parsed_content(
                parsed_data=parsed_data,
                source_asset_id=nonexistent_source_id,
                parser_type="test_parser",
                parser_version="1.0.0"
            )
            assert False, "Should have raised ValueError for non-existent source"
        except ValueError as e:
            assert nonexistent_source_id in str(e)
            print(f"✓ ValueError raised for non-existent source: {e}")

    def test_chunks_error_handling(self, file_store, sample_data):
        """Test error handling for chunks with invalid source parsed content"""
        print("\n--- Testing chunks error handling ---")
        nonexistent_parsed_content_id = "00000000-0000-0000-0000-000000000000"
        chunks_data = sample_data["chunks_data"]

        try:
            file_store.store_chunks(
                chunks_data=chunks_data,
                source_parsed_content_id=nonexistent_parsed_content_id,
                chunking_strategy="test_strategy",
                chunking_version="1.0.0"
            )
            assert False, "Should have raised ValueError for non-existent parsed content"
        except ValueError as e:
            assert nonexistent_parsed_content_id in str(e)
            print(f"✓ ValueError raised for non-existent parsed content: {e}")

        # Test retrieval error handling for non-existent chunks
        nonexistent_chunks_id = "00000000-0000-0000-0000-000000000000"

        # Test chunks metadata retrieval for non-existent chunks
        chunks_metadata = file_store.get_chunks_metadata(nonexistent_chunks_id)
        assert chunks_metadata is None
        print("✓ Non-existent chunks metadata returns None")

        # Test chunks content retrieval for non-existent chunks
        chunks_content = file_store.get_chunks_content(nonexistent_chunks_id)
        assert chunks_content is None
        print("✓ Non-existent chunks content returns None")

        # Test validation for non-existent chunks
        is_valid = file_store.validate_chunks_upload(nonexistent_chunks_id)
        assert is_valid is False
        print("✓ Non-existent chunks validation returns False")

        # Test cleanup for non-existent chunks (should not fail)
        cleanup_success = file_store.cleanup_failed_chunks_upload(nonexistent_chunks_id)
        # Cleanup might succeed even for non-existent chunks as it's idempotent
        print(f"✓ Cleanup for non-existent chunks handled gracefully: {cleanup_success}")
    
    def test_cleanup_operations(self, file_store, sample_data):
        """Test cleanup operations for failed uploads"""
        print("\n--- Testing cleanup operations ---")
        filename = "cleanup_test.txt"
        file_data = sample_data["text_data"]
        
        # Store file normally
        metadata = file_store.store_file(
            file_data=file_data,
            filename=filename,
            content_type="text/plain"
        )
        print(f"✓ Stored file for cleanup test: {metadata.asset_id}")
        
        # Verify file exists
        assert file_store.get_file_metadata(metadata.asset_id) is not None
        assert file_store.get_file_content(metadata.asset_id) is not None
        
        # Cleanup the file
        cleanup_success = file_store.cleanup_failed_file_upload(metadata.asset_id)
        assert cleanup_success is True
        print("✓ File cleanup successful")
        
        # Verify file is gone
        assert file_store.get_file_metadata(metadata.asset_id) is None
        assert file_store.get_file_content(metadata.asset_id) is None
        print("✓ File completely removed after cleanup")


def main():
    """Main test function"""
    print("Testing FileStore (Coordinated File + Metadata Storage)...")
    print("Note: This test requires PostgreSQL server running on localhost:5432")
    print("Database 'test_filestore_db' will be created if it doesn't exist")
    
    try:
        # Create file store instance using proper config pattern
        config, temp_dir = create_test_config()
        file_store = config.build()
        
        # Sample data
        sample_data = {
            "text_data": b"Hello, this is test content for coordinated file storage!",
            "binary_data": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",
            "json_data": b'{"test": "coordinated storage", "number": 42}',
            "markdown_data": b"# Test Document\n\nThis is a **test** document for parsed content.",
            "chunks_data": b'[{"chunk_id": 1, "content": "First chunk of text", "metadata": {"start": 0, "end": 20}}, {"chunk_id": 2, "content": "Second chunk of text", "metadata": {"start": 21, "end": 42}}]',
            "chunks_data_large": b'[{"chunk_id": 1, "content": "Large chunk 1 with more content", "metadata": {"start": 0, "end": 32}}, {"chunk_id": 2, "content": "Large chunk 2 with even more content", "metadata": {"start": 33, "end": 70}}, {"chunk_id": 3, "content": "Large chunk 3 final section", "metadata": {"start": 71, "end": 99}}]',
        }

        # Run tests
        test_instance = TestFileStore()

        test_instance.test_store_and_retrieve_file(file_store, sample_data)
        test_instance.test_store_file_stream(file_store, sample_data)
        test_instance.test_store_parsed_content(file_store, sample_data)
        test_instance.test_store_chunks(file_store, sample_data)
        test_instance.test_store_chunks_stream(file_store, sample_data)
        test_instance.test_chunks_validation_and_cleanup(file_store, sample_data)
        test_instance.test_upload_status_tracking(file_store, sample_data)
        test_instance.test_error_handling_nonexistent_files(file_store)
        test_instance.test_parsed_content_error_handling(file_store, sample_data)
        test_instance.test_chunks_error_handling(file_store, sample_data)
        test_instance.test_cleanup_operations(file_store, sample_data)
        
        print(f"\n🎉 All FileStore tests passed!")
        print(f"Test directory: {temp_dir}")
        
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up test directory")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure PostgreSQL server is running and accessible")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()