"""
Test for PostgreSQL Metadata Store - testing file and parsed content metadata operations
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
import uuid

from framework.config import AbstractConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.database.relational_db.models.file_metadata import FileMetadata, FileStatus
from encapsulation.database.relational_db.models.parsed_content_metadata import ParsedContentMetadata, ParsedContentStatus
from encapsulation.database.relational_db.models.chunks_metadata import ChunksMetadata, ChunksStatus


class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL metadata store testing"""
    type: Literal["postgresql_metadata"] = "postgresql_metadata"
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_arc_test"
    user: str = "postgres"
    password: str = "123"
    pool_size: int = 5       # Optional
    max_overflow: int = 10   # Optional
    echo_sql: bool = False   # Optional
    
    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)


def create_test_config() -> PostgreSQLConfig:
    """Create PostgreSQL configuration for testing"""
    return PostgreSQLConfig()


@pytest.fixture
def metadata_store():
    """Create PostgreSQL metadata store instance for testing"""
    config = create_test_config()
    return config.build()


@pytest.fixture
def sample_file_metadata():
    """Sample file metadata for testing"""
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    return FileMetadata(
        asset_id=str(uuid.uuid4()),
        blob_key="test/sample_document.pdf",
        filename="sample_document.pdf",
        status=FileStatus.UPLOADED,
        file_size=1024000,
        content_type="application/pdf",
        checksum="abc123def456",
        created_at=now,
        updated_at=now,
        original_path="/uploads/sample_document.pdf"
    )


@pytest.fixture
def sample_parsed_content_metadata():
    """Sample parsed content metadata for testing"""
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    return ParsedContentMetadata(
        parsed_content_id=str(uuid.uuid4()),
        source_asset_id=str(uuid.uuid4()),
        blob_key="test/parsed_content.md",
        content_size=50000,
        checksum="def456ghi789",
        parser_type="dots_ocr",
        parser_version="1.0.0",
        status=ParsedContentStatus.PARSED,
        created_at=now,
        updated_at=now,
        content_type="text/markdown",
        parsing_config='{"confidence_threshold": 0.8}',
        page_count=5,
        language="en"
    )


@pytest.fixture
def sample_chunks_metadata():
    """Sample chunks metadata for testing"""
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    return ChunksMetadata(
        chunks_id=str(uuid.uuid4()),
        source_parsed_content_id=str(uuid.uuid4()),
        blob_key="chunks/ch/chunks-123/chunks.json",
        chunks_count=10,
        content_size=25000,
        checksum="ghi789jkl012",
        chunking_strategy="fixed_1000",
        chunking_version="1.0.0",
        status=ChunksStatus.CHUNKED,
        created_at=now,
        updated_at=now,
        content_type="application/json",
        processing_time_ms=1500,
        chunking_config='{"chunk_size": 1000, "overlap": 100}'
    )


class TestPostgreSQLMetadataStore:
    """Test cases for PostgreSQL metadata store"""
    
    def test_store_and_get_file_metadata(self, metadata_store, sample_file_metadata):
        """Test storing and retrieving file metadata"""
        print("\n--- Testing file metadata storage and retrieval ---")
        
        # Store metadata
        stored_asset_id = metadata_store.store_file_metadata(sample_file_metadata)
        assert stored_asset_id == sample_file_metadata.asset_id
        print(f"✓ Stored file metadata with asset_id: {stored_asset_id}")
        
        # Retrieve metadata
        retrieved_metadata = metadata_store.get_file_metadata(sample_file_metadata.asset_id)
        assert retrieved_metadata is not None
        assert retrieved_metadata.asset_id == sample_file_metadata.asset_id
        assert retrieved_metadata.filename == sample_file_metadata.filename
        assert retrieved_metadata.status == sample_file_metadata.status
        assert retrieved_metadata.file_size == sample_file_metadata.file_size
        print(f"✓ Retrieved file metadata: {retrieved_metadata.filename}")
        
        # Clean up
        deleted = metadata_store.delete_file_metadata(sample_file_metadata.asset_id)
        assert deleted is True
        print(f"✓ Deleted file metadata: {deleted}")
    
    def test_update_file_metadata(self, metadata_store, sample_file_metadata):
        """Test updating file metadata"""
        print("\n--- Testing file metadata updates ---")
        
        # Store initial metadata
        metadata_store.store_file_metadata(sample_file_metadata)
        print(f"✓ Stored initial metadata")
        
        # Update filename and file size
        updates = {
            "filename": "updated_document.pdf",
            "file_size": 2048000
        }
        updated = metadata_store.update_file_metadata(sample_file_metadata.asset_id, updates)
        assert updated is True
        print(f"✓ Updated metadata fields")
        
        # Verify updates
        retrieved_metadata = metadata_store.get_file_metadata(sample_file_metadata.asset_id)
        assert retrieved_metadata.filename == "updated_document.pdf"
        assert retrieved_metadata.file_size == 2048000
        # assert retrieved_metadata.updated_at > sample_file_metadata.updated_at
        print(f"✓ Verified updates: {retrieved_metadata.filename}, {retrieved_metadata.file_size} bytes")
        
        # Clean up
        metadata_store.delete_file_metadata(sample_file_metadata.asset_id)
    
    def test_update_file_status(self, metadata_store, sample_file_metadata):
        """Test updating file processing status"""
        print("\n--- Testing file status updates ---")
        
        # Store initial metadata
        metadata_store.store_file_metadata(sample_file_metadata)
        print(f"✓ Initial status: {sample_file_metadata.status.value}")
        
        # Update status to PARSING
        updated = metadata_store.update_file_status(sample_file_metadata.asset_id, FileStatus.PARSING)
        assert updated is True
        print(f"✓ Updated status to: {FileStatus.PARSING.value}")
        
        # Verify status update
        retrieved_metadata = metadata_store.get_file_metadata(sample_file_metadata.asset_id)
        assert retrieved_metadata.status == FileStatus.PARSING
        
        # Update status to PARSED
        updated = metadata_store.update_file_status(sample_file_metadata.asset_id, FileStatus.PARSED)
        assert updated is True
        print(f"✓ Updated status to: {FileStatus.PARSED.value}")
        
        # Verify final status
        retrieved_metadata = metadata_store.get_file_metadata(sample_file_metadata.asset_id)
        assert retrieved_metadata.status == FileStatus.PARSED
        print(f"✓ Final status confirmed: {retrieved_metadata.status.value}")
        
        # Clean up
        metadata_store.delete_file_metadata(sample_file_metadata.asset_id)
    
    def test_list_file_metadata(self, metadata_store):
        """Test listing file metadata with filtering"""
        print("\n--- Testing file metadata listing ---")
        
        # Create multiple test files with different statuses
        test_files = []
        statuses = [FileStatus.UPLOADED, FileStatus.PARSING, FileStatus.PARSED, FileStatus.FAILED]
        
        for i, status in enumerate(statuses):
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            file_metadata = FileMetadata(
                asset_id=str(uuid.uuid4()),
                blob_key=f"test/file_{i}.pdf",
                filename=f"file_{i}.pdf",
                status=status,
                file_size=1000 * (i + 1),
                content_type="application/pdf",
                checksum=f"hash{i}",
                created_at=now,
                updated_at=now,
                original_path=f"/uploads/file_{i}.pdf"
            )
            metadata_store.store_file_metadata(file_metadata)
            test_files.append(file_metadata)
        
        print(f"✓ Created {len(test_files)} test files")
        
        # List all files
        all_files = metadata_store.list_file_metadata()
        assert len(all_files) >= 4
        print(f"✓ Listed {len(all_files)} total files")
        
        # List files by status
        uploaded_files = metadata_store.list_file_metadata(status=FileStatus.UPLOADED)
        assert len(uploaded_files) >= 1
        print(f"✓ Found {len(uploaded_files)} UPLOADED files")
        
        parsed_files = metadata_store.list_file_metadata(status=FileStatus.PARSED)
        assert len(parsed_files) >= 1
        print(f"✓ Found {len(parsed_files)} PARSED files")
        
        # Test pagination
        limited_files = metadata_store.list_file_metadata(limit=2)
        assert len(limited_files) == 2
        print(f"✓ Limited listing returned {len(limited_files)} files")
        
        # Clean up
        for file_metadata in test_files:
            metadata_store.delete_file_metadata(file_metadata.asset_id)
        print(f"✓ Cleaned up {len(test_files)} test files")
    
    def test_delete_file_metadata(self, metadata_store, sample_file_metadata):
        """Test deleting file metadata"""
        print("\n--- Testing file metadata deletion ---")
        
        # Store metadata
        metadata_store.store_file_metadata(sample_file_metadata)
        print(f"✓ Stored metadata for deletion test")
        
        # Verify it exists
        retrieved = metadata_store.get_file_metadata(sample_file_metadata.asset_id)
        assert retrieved is not None
        print(f"✓ Confirmed metadata exists")
        
        # Delete metadata
        deleted = metadata_store.delete_file_metadata(sample_file_metadata.asset_id)
        assert deleted is True
        print(f"✓ Deleted metadata successfully")
        
        # Verify it's gone
        retrieved = metadata_store.get_file_metadata(sample_file_metadata.asset_id)
        assert retrieved is None
        print(f"✓ Confirmed metadata is deleted")
        
        # Try to delete again (should return False)
        deleted_again = metadata_store.delete_file_metadata(sample_file_metadata.asset_id)
        assert deleted_again is False
        print(f"✓ Second deletion correctly returned False")
    
    def test_duplicate_asset_id_error(self, metadata_store, sample_file_metadata):
        """Test error handling for duplicate asset IDs"""
        print("\n--- Testing duplicate asset ID handling ---")
        
        # Store initial metadata
        metadata_store.store_file_metadata(sample_file_metadata)
        print(f"✓ Stored initial metadata")
        
        # Try to store duplicate
        try:
            metadata_store.store_file_metadata(sample_file_metadata)
            assert False, "Should have raised ValueError for duplicate asset_id"
        except ValueError as e:
            assert "already exists" in str(e)
            print(f"✓ Correctly raised ValueError: {e}")
        
        # Clean up
        metadata_store.delete_file_metadata(sample_file_metadata.asset_id)
    
    def test_nonexistent_file_operations(self, metadata_store):
        """Test operations on non-existent files"""
        print("\n--- Testing non-existent file operations ---")
        
        fake_asset_id = str(uuid.uuid4())
        
        # Get non-existent file
        result = metadata_store.get_file_metadata(fake_asset_id)
        assert result is None
        print(f"✓ Get non-existent file returned None")
        
        # Update non-existent file
        updated = metadata_store.update_file_metadata(fake_asset_id, {"filename": "new.pdf"})
        assert updated is False
        print(f"✓ Update non-existent file returned False")
        
        # Delete non-existent file
        deleted = metadata_store.delete_file_metadata(fake_asset_id)
        assert deleted is False
        print(f"✓ Delete non-existent file returned False")
    
    # Parsed Content Metadata Tests
    
    def test_store_and_get_parsed_content_metadata(self, metadata_store, sample_parsed_content_metadata):
        """Test storing and retrieving parsed content metadata"""
        print("\n--- Testing parsed content metadata storage and retrieval ---")
        
        # Store metadata
        stored_id = metadata_store.store_parsed_content_metadata(sample_parsed_content_metadata)
        assert stored_id == sample_parsed_content_metadata.parsed_content_id
        print(f"✓ Stored parsed content metadata with ID: {stored_id}")
        
        # Retrieve metadata
        retrieved_metadata = metadata_store.get_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
        assert retrieved_metadata is not None
        assert retrieved_metadata.parsed_content_id == sample_parsed_content_metadata.parsed_content_id
        assert retrieved_metadata.source_asset_id == sample_parsed_content_metadata.source_asset_id
        assert retrieved_metadata.parser_type == sample_parsed_content_metadata.parser_type
        assert retrieved_metadata.status == sample_parsed_content_metadata.status
        print(f"✓ Retrieved parsed content metadata: {retrieved_metadata.parser_type}")
        
        # Clean up
        deleted = metadata_store.delete_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
        assert deleted is True
        print(f"✓ Deleted parsed content metadata: {deleted}")
    
    def test_update_parsed_content_metadata(self, metadata_store, sample_parsed_content_metadata):
        """Test updating parsed content metadata"""
        print("\n--- Testing parsed content metadata updates ---")
        
        # Store initial metadata
        metadata_store.store_parsed_content_metadata(sample_parsed_content_metadata)
        print(f"✓ Stored initial parsed content metadata")
        
        # Update parser version and content size
        updates = {
            "parser_version": "1.1.0",
            "content_size": 75000
        }
        updated = metadata_store.update_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id, updates)
        assert updated is True
        print(f"✓ Updated metadata fields")
        
        # Verify updates
        retrieved_metadata = metadata_store.get_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
        assert retrieved_metadata.parser_version == "1.1.0"
        assert retrieved_metadata.content_size == 75000
        # assert retrieved_metadata.updated_at > sample_parsed_content_metadata.updated_at
        print(f"✓ Verified updates: {retrieved_metadata.parser_version}, {retrieved_metadata.content_size} bytes")
        
        # Clean up
        metadata_store.delete_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
    
    def test_update_parsed_content_status(self, metadata_store, sample_parsed_content_metadata):
        """Test updating parsed content processing status"""
        print("\n--- Testing parsed content status updates ---")
        
        # Store initial metadata
        metadata_store.store_parsed_content_metadata(sample_parsed_content_metadata)
        print(f"✓ Initial status: {sample_parsed_content_metadata.status.value}")
        
        # Update status to INDEXED
        updated = metadata_store.update_parsed_content_status(sample_parsed_content_metadata.parsed_content_id, ParsedContentStatus.INDEXED)
        assert updated is True
        print(f"✓ Updated status to: {ParsedContentStatus.INDEXED.value}")
        
        # Verify status update
        retrieved_metadata = metadata_store.get_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
        assert retrieved_metadata.status == ParsedContentStatus.INDEXED
        print(f"✓ Status confirmed: {retrieved_metadata.status.value}")
        
        # Clean up
        metadata_store.delete_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
    
    def test_list_parsed_content_metadata(self, metadata_store):
        """Test listing parsed content metadata with filtering"""
        print("\n--- Testing parsed content metadata listing ---")
        
        # Create multiple test parsed contents with different attributes
        source_asset_id = str(uuid.uuid4())
        test_parsed_contents = []
        parsers = ["dots_ocr", "pypdf", "unstructured"]
        statuses = [ParsedContentStatus.PARSED, ParsedContentStatus.INDEXED, ParsedContentStatus.FAILED]
        
        for i, (parser, status) in enumerate(zip(parsers, statuses)):
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            parsed_metadata = ParsedContentMetadata(
                parsed_content_id=str(uuid.uuid4()),
                source_asset_id=source_asset_id if i < 2 else str(uuid.uuid4()),  # First 2 share same source
                blob_key=f"test/parsed_{i}.md",
                content_size=1000 * (i + 1),
                checksum=f"hash{i}",
                parser_type=parser,
                parser_version="1.0.0",
                status=status,
                created_at=now,
                updated_at=now,
                content_type="text/markdown"
            )
            metadata_store.store_parsed_content_metadata(parsed_metadata)
            test_parsed_contents.append(parsed_metadata)
        
        print(f"✓ Created {len(test_parsed_contents)} test parsed contents")
        
        # List all parsed contents
        all_parsed = metadata_store.list_parsed_content_metadata()
        assert len(all_parsed) >= 3
        print(f"✓ Listed {len(all_parsed)} total parsed contents")
        
        # List by source asset ID
        source_parsed = metadata_store.list_parsed_content_metadata(source_asset_id=source_asset_id)
        assert len(source_parsed) >= 2
        print(f"✓ Found {len(source_parsed)} parsed contents for source asset")
        
        # List by status
        indexed_parsed = metadata_store.list_parsed_content_metadata(status=ParsedContentStatus.INDEXED)
        assert len(indexed_parsed) >= 1
        print(f"✓ Found {len(indexed_parsed)} INDEXED parsed contents")
        
        # List by parser type
        ocr_parsed = metadata_store.list_parsed_content_metadata(parser_type="dots_ocr")
        assert len(ocr_parsed) >= 1
        print(f"✓ Found {len(ocr_parsed)} dots_ocr parsed contents")
        
        # Test pagination
        limited_parsed = metadata_store.list_parsed_content_metadata(limit=2)
        assert len(limited_parsed) == 2
        print(f"✓ Limited listing returned {len(limited_parsed)} parsed contents")
        
        # Clean up
        for parsed_metadata in test_parsed_contents:
            metadata_store.delete_parsed_content_metadata(parsed_metadata.parsed_content_id)
        print(f"✓ Cleaned up {len(test_parsed_contents)} test parsed contents")
    
    def test_duplicate_parsed_content_id_error(self, metadata_store, sample_parsed_content_metadata):
        """Test error handling for duplicate parsed content IDs"""
        print("\n--- Testing duplicate parsed content ID handling ---")
        
        # Store initial metadata
        metadata_store.store_parsed_content_metadata(sample_parsed_content_metadata)
        print(f"✓ Stored initial parsed content metadata")
        
        # Try to store duplicate
        try:
            metadata_store.store_parsed_content_metadata(sample_parsed_content_metadata)
            assert False, "Should have raised ValueError for duplicate parsed_content_id"
        except ValueError as e:
            assert "already exists" in str(e)
            print(f"✓ Correctly raised ValueError: {e}")
        
        # Clean up
        metadata_store.delete_parsed_content_metadata(sample_parsed_content_metadata.parsed_content_id)
    
    def test_nonexistent_parsed_content_operations(self, metadata_store):
        """Test operations on non-existent parsed content"""
        print("\n--- Testing non-existent parsed content operations ---")
        
        fake_parsed_id = str(uuid.uuid4())
        
        # Get non-existent parsed content
        result = metadata_store.get_parsed_content_metadata(fake_parsed_id)
        assert result is None
        print(f"✓ Get non-existent parsed content returned None")
        
        # Update non-existent parsed content
        updated = metadata_store.update_parsed_content_metadata(fake_parsed_id, {"parser_version": "2.0.0"})
        assert updated is False
        print(f"✓ Update non-existent parsed content returned False")
        
        # Delete non-existent parsed content
        deleted = metadata_store.delete_parsed_content_metadata(fake_parsed_id)
        assert deleted is False
        print(f"✓ Delete non-existent parsed content returned False")

    # Chunks Metadata Tests

    def test_store_and_get_chunks_metadata(self, metadata_store, sample_chunks_metadata):
        """Test storing and retrieving chunks metadata"""
        print("\n--- Testing chunks metadata storage and retrieval ---")

        # Store metadata
        stored_id = metadata_store.store_chunks_metadata(sample_chunks_metadata)
        assert stored_id == sample_chunks_metadata.chunks_id
        print(f"✓ Stored chunks metadata with ID: {stored_id}")

        # Retrieve metadata
        retrieved_metadata = metadata_store.get_chunks_metadata(sample_chunks_metadata.chunks_id)
        assert retrieved_metadata is not None
        assert retrieved_metadata.chunks_id == sample_chunks_metadata.chunks_id
        assert retrieved_metadata.source_parsed_content_id == sample_chunks_metadata.source_parsed_content_id
        assert retrieved_metadata.chunking_strategy == sample_chunks_metadata.chunking_strategy
        assert retrieved_metadata.status == sample_chunks_metadata.status
        assert retrieved_metadata.chunks_count == sample_chunks_metadata.chunks_count
        print(f"✓ Retrieved chunks metadata: {retrieved_metadata.chunking_strategy}")

        # Clean up
        deleted = metadata_store.delete_chunks_metadata(sample_chunks_metadata.chunks_id)
        assert deleted is True
        print(f"✓ Deleted chunks metadata: {deleted}")

    def test_update_chunks_metadata(self, metadata_store, sample_chunks_metadata):
        """Test updating chunks metadata"""
        print("\n--- Testing chunks metadata updates ---")

        # Store initial metadata
        metadata_store.store_chunks_metadata(sample_chunks_metadata)
        print(f"✓ Stored initial chunks metadata")

        # Update chunking version and content size
        updates = {
            "chunking_version": "1.1.0",
            "content_size": 30000
        }
        updated = metadata_store.update_chunks_metadata(sample_chunks_metadata.chunks_id, updates)
        assert updated is True
        print(f"✓ Updated metadata fields")

        # Verify updates
        retrieved_metadata = metadata_store.get_chunks_metadata(sample_chunks_metadata.chunks_id)
        assert retrieved_metadata.chunking_version == "1.1.0"
        assert retrieved_metadata.content_size == 30000
        # assert retrieved_metadata.updated_at > sample_chunks_metadata.updated_at
        print(f"✓ Verified updates: {retrieved_metadata.chunking_version}, {retrieved_metadata.content_size} bytes")

        # Clean up
        metadata_store.delete_chunks_metadata(sample_chunks_metadata.chunks_id)

    def test_update_chunks_status(self, metadata_store, sample_chunks_metadata):
        """Test updating chunks processing status"""
        print("\n--- Testing chunks status updates ---")

        # Store initial metadata
        metadata_store.store_chunks_metadata(sample_chunks_metadata)
        print(f"✓ Initial status: {sample_chunks_metadata.status.value}")

        # Update status to INDEXED
        updated = metadata_store.update_chunks_status(sample_chunks_metadata.chunks_id, ChunksStatus.INDEXED)
        assert updated is True
        print(f"✓ Updated status to: {ChunksStatus.INDEXED.value}")

        # Verify status update
        retrieved_metadata = metadata_store.get_chunks_metadata(sample_chunks_metadata.chunks_id)
        assert retrieved_metadata.status == ChunksStatus.INDEXED
        print(f"✓ Status confirmed: {retrieved_metadata.status.value}")

        # Clean up
        metadata_store.delete_chunks_metadata(sample_chunks_metadata.chunks_id)

    def test_list_chunks_metadata(self, metadata_store):
        """Test listing chunks metadata with filtering"""
        print("\n--- Testing chunks metadata listing ---")

        # Create multiple test chunks with different attributes
        source_parsed_content_id = str(uuid.uuid4())
        test_chunks = []
        strategies = ["fixed_1000", "semantic_0.8", "recursive_500"]
        statuses = [ChunksStatus.CHUNKED, ChunksStatus.INDEXED, ChunksStatus.FAILED]

        for i, (strategy, status) in enumerate(zip(strategies, statuses)):
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            chunks_metadata = ChunksMetadata(
                chunks_id=str(uuid.uuid4()),
                source_parsed_content_id=source_parsed_content_id if i < 2 else str(uuid.uuid4()),  # First 2 share same source
                blob_key=f"chunks/ch/chunks_{i}.json",
                chunks_count=5 + i * 2,
                content_size=1000 * (i + 1),
                checksum=f"hash{i}",
                chunking_strategy=strategy,
                chunking_version="1.0.0",
                status=status,
                created_at=now,
                updated_at=now,
                content_type="application/json"
            )
            metadata_store.store_chunks_metadata(chunks_metadata)
            test_chunks.append(chunks_metadata)

        print(f"✓ Created {len(test_chunks)} test chunks")

        # List all chunks
        all_chunks = metadata_store.list_chunks_metadata()
        assert len(all_chunks) >= 3
        print(f"✓ Listed {len(all_chunks)} total chunks")

        # List by source parsed content ID
        source_chunks = metadata_store.list_chunks_metadata(source_parsed_content_id=source_parsed_content_id)
        assert len(source_chunks) >= 2
        print(f"✓ Found {len(source_chunks)} chunks for source parsed content")

        # List by status
        indexed_chunks = metadata_store.list_chunks_metadata(status=ChunksStatus.INDEXED)
        assert len(indexed_chunks) >= 1
        print(f"✓ Found {len(indexed_chunks)} INDEXED chunks")

        # List by chunking strategy
        fixed_chunks = metadata_store.list_chunks_metadata(chunking_strategy="fixed_1000")
        assert len(fixed_chunks) >= 1
        print(f"✓ Found {len(fixed_chunks)} fixed_1000 chunks")

        # Test pagination
        limited_chunks = metadata_store.list_chunks_metadata(limit=2)
        assert len(limited_chunks) == 2
        print(f"✓ Limited listing returned {len(limited_chunks)} chunks")

        # Clean up
        for chunks_metadata in test_chunks:
            metadata_store.delete_chunks_metadata(chunks_metadata.chunks_id)
        print(f"✓ Cleaned up {len(test_chunks)} test chunks")

    def test_duplicate_chunks_id_error(self, metadata_store, sample_chunks_metadata):
        """Test error handling for duplicate chunks IDs"""
        print("\n--- Testing duplicate chunks ID handling ---")

        # Store initial metadata
        metadata_store.store_chunks_metadata(sample_chunks_metadata)
        print(f"✓ Stored initial chunks metadata")

        # Try to store duplicate
        try:
            metadata_store.store_chunks_metadata(sample_chunks_metadata)
            assert False, "Should have raised ValueError for duplicate chunks_id"
        except ValueError as e:
            assert "already exists" in str(e)
            print(f"✓ Correctly raised ValueError: {e}")

        # Clean up
        metadata_store.delete_chunks_metadata(sample_chunks_metadata.chunks_id)

    def test_nonexistent_chunks_operations(self, metadata_store):
        """Test operations on non-existent chunks"""
        print("\n--- Testing non-existent chunks operations ---")

        fake_chunks_id = str(uuid.uuid4())

        # Get non-existent chunks
        result = metadata_store.get_chunks_metadata(fake_chunks_id)
        assert result is None
        print(f"✓ Get non-existent chunks returned None")

        # Update non-existent chunks
        updated = metadata_store.update_chunks_metadata(fake_chunks_id, {"chunking_version": "2.0.0"})
        assert updated is False
        print(f"✓ Update non-existent chunks returned False")

        # Delete non-existent chunks
        deleted = metadata_store.delete_chunks_metadata(fake_chunks_id)
        assert deleted is False
        print(f"✓ Delete non-existent chunks returned False")


def main():
    """Main test function"""
    print("Testing PostgreSQL Metadata Store...")
    print("Note: This test requires PostgreSQL server running with test database")
    print("Connection: postgresql://postgres:password@localhost:5432/rag_arc_test")
    
    try:
        # Create metadata store instance
        config = create_test_config()
        metadata_store = config.build()
        
        # Sample file metadata
        now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        sample_file_metadata = FileMetadata(
            asset_id=str(uuid.uuid4()),
            blob_key="test/sample_document.pdf",
            filename="sample_document.pdf",
            status=FileStatus.UPLOADED,
            file_size=1024000,
            content_type="application/pdf",
            checksum="abc123def456",
            created_at=now,
            updated_at=now,
            original_path="/uploads/sample_document.pdf"
        )
        
        # Sample parsed content metadata
        sample_parsed_content_metadata = ParsedContentMetadata(
            parsed_content_id=str(uuid.uuid4()),
            source_asset_id=str(uuid.uuid4()),
            blob_key="test/parsed_content.md",
            content_size=50000,
            checksum="def456ghi789",
            parser_type="dots_ocr",
            parser_version="1.0.0",
            status=ParsedContentStatus.PARSED,
            created_at=now,
            updated_at=now,
            content_type="text/markdown",
            parsing_config='{"confidence_threshold": 0.8}',
            page_count=5,
            language="en"
        )

        # Sample chunks metadata
        sample_chunks_metadata = ChunksMetadata(
            chunks_id=str(uuid.uuid4()),
            source_parsed_content_id=str(uuid.uuid4()),
            blob_key="chunks/ch/chunks-123/chunks.json",
            chunks_count=10,
            content_size=25000,
            checksum="ghi789jkl012",
            chunking_strategy="fixed_1000",
            chunking_version="1.0.0",
            status=ChunksStatus.CHUNKED,
            created_at=now,
            updated_at=now,
            content_type="application/json",
            processing_time_ms=1500,
            chunking_config='{"chunk_size": 1000, "overlap": 100}'
        )
        
        # Run tests
        test_instance = TestPostgreSQLMetadataStore()
        
        print("\n=== File Metadata Tests ===")
        test_instance.test_store_and_get_file_metadata(metadata_store, sample_file_metadata)
        test_instance.test_update_file_metadata(metadata_store, sample_file_metadata)
        test_instance.test_update_file_status(metadata_store, sample_file_metadata)
        test_instance.test_list_file_metadata(metadata_store)
        test_instance.test_delete_file_metadata(metadata_store, sample_file_metadata)
        test_instance.test_duplicate_asset_id_error(metadata_store, sample_file_metadata)
        test_instance.test_nonexistent_file_operations(metadata_store)
        
        print("\n=== Parsed Content Metadata Tests ===")
        test_instance.test_store_and_get_parsed_content_metadata(metadata_store, sample_parsed_content_metadata)
        test_instance.test_update_parsed_content_metadata(metadata_store, sample_parsed_content_metadata)
        test_instance.test_update_parsed_content_status(metadata_store, sample_parsed_content_metadata)
        test_instance.test_list_parsed_content_metadata(metadata_store)
        test_instance.test_duplicate_parsed_content_id_error(metadata_store, sample_parsed_content_metadata)
        test_instance.test_nonexistent_parsed_content_operations(metadata_store)

        print("\n=== Chunks Metadata Tests ===")
        test_instance.test_store_and_get_chunks_metadata(metadata_store, sample_chunks_metadata)
        test_instance.test_update_chunks_metadata(metadata_store, sample_chunks_metadata)
        test_instance.test_update_chunks_status(metadata_store, sample_chunks_metadata)
        test_instance.test_list_chunks_metadata(metadata_store)
        test_instance.test_duplicate_chunks_id_error(metadata_store, sample_chunks_metadata)
        test_instance.test_nonexistent_chunks_operations(metadata_store)

        print("\n🎉 All PostgreSQL metadata store tests passed!")
        print("✓ File metadata operations working correctly")
        print("✓ Parsed content metadata operations working correctly")
        print("✓ Chunks metadata operations working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure PostgreSQL server is running and test database exists")
        print("Database will be created automatically if it doesn't exist")


if __name__ == "__main__":
    main()