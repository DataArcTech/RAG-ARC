"""
Simple test to understand how PostgreSQL database operations work with SQLAlchemy ORM
"""

import sys
import os

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import AbstractConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.database.relational_db.data_schema import (
    FileMetadata, FileStatus,
    ParsedContentMetadata, ParsedContentStatus,
    ChunksMetadata, ChunksStatus
)
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal

class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL Database"""
    type: Literal["postgresql"] = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_arc_test"
    user: str = "postgres"
    password: str = "123"

    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)

def test_file_metadata_operations(db: PostgreSQLDB):
    """Test complete CRUD operations for FileMetadata"""
    print("\n=== FILE METADATA OPERATIONS ===")

    # Create test file metadata
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    file_metadata = FileMetadata(
        asset_id="test-file-123",
        blob_key="assets/te/test-file-123/document.pdf",
        filename="document.pdf",
        status=FileStatus.UPLOADING,  # Using enum directly
        file_size=1024000,
        content_type="application/pdf",
        checksum="abc123def456",
        created_at=now,
        updated_at=now,
        original_path="/uploads/document.pdf"
    )

    # Test CREATE
    print("1. Testing file metadata creation...")
    stored_id = db.store_file_metadata(file_metadata)
    print(f"   ✓ Stored file metadata with ID: {stored_id}")

    # Test READ
    print("2. Testing file metadata retrieval...")
    retrieved = db.get_file_metadata("test-file-123")
    if retrieved:
        print(f"   ✓ Retrieved: {retrieved.filename} - Status: {retrieved.status}")
        print(f"   ✓ Status type: {type(retrieved.status)} (should be FileStatus enum)")
    else:
        print("   ✗ Failed to retrieve file metadata")

    # Test UPDATE
    print("3. Testing file metadata update...")
    success = db.update_file_metadata("test-file-123", {
        "status": FileStatus.UPLOADED,  # Using enum directly
        "blob_key": "assets/te/test-file-123/document-v2.pdf"
    })
    if success:
        updated = db.get_file_metadata("test-file-123")
        print(f"   ✓ Updated status to: {updated.status}")
        print(f"   ✓ Updated blob_key to: {updated.blob_key}")
    else:
        print("   ✗ Failed to update file metadata")

    # Test LIST
    print("4. Testing file metadata listing...")
    file_list = db.list_file_metadata(status=FileStatus.UPLOADED, limit=5)
    print(f"   ✓ Found {len(file_list)} uploaded files")
    for f in file_list:
        print(f"     - {f.filename} ({f.status})")

    return retrieved

def test_parsed_content_operations(db: PostgreSQLDB, source_file: FileMetadata):
    """Test complete CRUD operations for ParsedContentMetadata"""
    print("\n=== PARSED CONTENT METADATA OPERATIONS ===")

    # Create test parsed content metadata
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    parsed_metadata = ParsedContentMetadata(
        parsed_content_id="parsed-123",
        source_asset_id=source_file.asset_id,
        blob_key="parsed/te/test-file-123/parsed-123.markdown",
        content_size=50000,
        checksum="def456ghi789",
        parser_type="pdf_parser",
        parser_version="1.0.0",
        status=ParsedContentStatus.PARSING,  # Using enum directly
        created_at=now,
        updated_at=now,
        content_type="text/markdown",
        parsing_config='{"extract_images": true}',
        page_count=10,
        language="en"
    )

    # Test CREATE
    print("1. Testing parsed content creation...")
    stored_id = db.store_parsed_content_metadata(parsed_metadata)
    print(f"   ✓ Stored parsed content with ID: {stored_id}")

    # Test READ
    print("2. Testing parsed content retrieval...")
    retrieved = db.get_parsed_content_metadata("parsed-123")
    if retrieved:
        print(f"   ✓ Retrieved: {retrieved.parser_type} - Status: {retrieved.status}")
        print(f"   ✓ Page count: {retrieved.page_count}, Language: {retrieved.language}")
    else:
        print("   ✗ Failed to retrieve parsed content metadata")

    # Test UPDATE
    print("3. Testing parsed content update...")
    success = db.update_parsed_content_metadata("parsed-123", {
        "status": ParsedContentStatus.PARSED,  # Using enum directly
        "page_count": 12
    })
    if success:
        updated = db.get_parsed_content_metadata("parsed-123")
        print(f"   ✓ Updated status to: {updated.status}")
        print(f"   ✓ Updated page count to: {updated.page_count}")
    else:
        print("   ✗ Failed to update parsed content metadata")

    # Test LIST
    print("4. Testing parsed content listing...")
    parsed_list = db.list_parsed_content_metadata(
        source_asset_id=source_file.asset_id,
        status=ParsedContentStatus.PARSED,
        limit=5
    )
    print(f"   ✓ Found {len(parsed_list)} parsed content for source file")

    return retrieved

def test_chunks_operations(db: PostgreSQLDB, source_parsed: ParsedContentMetadata):
    """Test complete CRUD operations for ChunksMetadata"""
    print("\n=== CHUNKS METADATA OPERATIONS ===")

    # Create test chunks metadata
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    chunks_metadata = ChunksMetadata(
        chunks_id="chunks-123",
        source_parsed_content_id=source_parsed.parsed_content_id,
        blob_key="chunks/pa/parsed-123/chunks-123.semantic",
        chunks_count=25,
        content_size=75000,
        checksum="ghi789jkl012",
        chunking_strategy="semantic_0.8",
        chunking_version="2.0.0",
        status=ChunksStatus.CHUNKING,  # Using enum directly
        created_at=now,
        updated_at=now,
        content_type="application/json",
        processing_time_ms=1500,
        chunking_config='{"similarity_threshold": 0.8}',
        index_type=None  # Not indexed yet
    )

    # Test CREATE
    print("1. Testing chunks creation...")
    stored_id = db.store_chunks_metadata(chunks_metadata)
    print(f"   ✓ Stored chunks with ID: {stored_id}")

    # Test READ
    print("2. Testing chunks retrieval...")
    retrieved = db.get_chunks_metadata("chunks-123")
    if retrieved:
        print(f"   ✓ Retrieved: {retrieved.chunking_strategy} - Status: {retrieved.status}")
        print(f"   ✓ Chunks count: {retrieved.chunks_count}, Index type: {retrieved.index_type}")
    else:
        print("   ✗ Failed to retrieve chunks metadata")

    # Test UPDATE (simulate indexing)
    print("3. Testing chunks update (indexing simulation)...")
    success = db.update_chunks_metadata("chunks-123", {
        "status": ChunksStatus.INDEXED,  # Using enum directly
        "index_type": "faiss"  # Now indexed in FAISS
    })
    if success:
        updated = db.get_chunks_metadata("chunks-123")
        print(f"   ✓ Updated status to: {updated.status}")
        print(f"   ✓ Updated index_type to: {updated.index_type}")
    else:
        print("   ✗ Failed to update chunks metadata")

    # Test LIST
    print("4. Testing chunks listing...")
    chunks_list = db.list_chunks_metadata(
        source_parsed_content_id=source_parsed.parsed_content_id,
        status=ChunksStatus.INDEXED,
        limit=5
    )
    print(f"   ✓ Found {len(chunks_list)} indexed chunks for source parsed content")

    return retrieved

def test_cleanup_operations(db: PostgreSQLDB):
    """Test cleanup by deleting all test data"""
    print("\n=== CLEANUP OPERATIONS ===")

    # Delete in reverse order (chunks -> parsed -> file) due to dependencies
    print("1. Deleting chunks metadata...")
    chunks_deleted = db.delete_chunks_metadata("chunks-123")
    print(f"   ✓ Chunks deleted: {chunks_deleted}")

    print("2. Deleting parsed content metadata...")
    parsed_deleted = db.delete_parsed_content_metadata("parsed-123")
    print(f"   ✓ Parsed content deleted: {parsed_deleted}")

    print("3. Deleting file metadata...")
    file_deleted = db.delete_file_metadata("test-file-123")
    print(f"   ✓ File metadata deleted: {file_deleted}")

def main():
    print("Testing PostgreSQL Database Operations with SQLAlchemy ORM...")

    # Create database instance using configuration injection
    config = PostgreSQLConfig()
    db = config.build()

    print(f"Database connected to: {config.host}:{config.port}/{config.database}")
    print(f"Engine info: {db.engine}")
    print(f"SessionMaker: {db.SessionMaker}")

    # Drop and recreate all tables to ensure schema is up-to-date
    print("\n=== SCHEMA SETUP ===")
    print("Dropping and recreating all tables...")
    from encapsulation.database.relational_db.data_schema import Base
    Base.metadata.drop_all(db.engine)
    Base.metadata.create_all(db.engine)
    print("✓ Tables recreated with latest schema")

    try:
        # Test file metadata operations
        file_metadata = test_file_metadata_operations(db)

        # Test parsed content operations (depends on file)
        parsed_metadata = test_parsed_content_operations(db, file_metadata)

        # Test chunks operations (depends on parsed content)
        chunks_metadata = test_chunks_operations(db, parsed_metadata)

        # Show final state
        print("\n=== FINAL STATE VERIFICATION ===")
        final_file = db.get_file_metadata("test-file-123")
        final_parsed = db.get_parsed_content_metadata("parsed-123")
        final_chunks = db.get_chunks_metadata("chunks-123")

        print(f"File status: {final_file.status} (type: {type(final_file.status)})")
        print(f"Parsed status: {final_parsed.status} (type: {type(final_parsed.status)})")
        print(f"Chunks status: {final_chunks.status}, Index: {final_chunks.index_type}")

        # Cleanup test data
        test_cleanup_operations(db)

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Try cleanup even if tests failed
        try:
            test_cleanup_operations(db)
        except:
            print("⚠️  Cleanup also failed - you may need to manually clean test data")

if __name__ == "__main__":
    main()