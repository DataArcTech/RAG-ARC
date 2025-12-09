"""
Simple test to understand how PostgreSQL database operations work with SQLAlchemy ORM
"""

import sys
import os
import pytest

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAGARC_POSTGRES_TESTS") != "1",
    reason="Requires a running PostgreSQL instance; set RUN_RAGARC_POSTGRES_TESTS=1 to enable.",
)

from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.data_model.orm_models import (
    FileMetadata, FileStatus,
    ParsedContentMetadata, ParsedContentStatus,
    ChunkMetadata, ChunkIndexStatus
)
from datetime import datetime
from zoneinfo import ZoneInfo

from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
import uuid


def _build_postgres_config() -> PostgreSQLConfig:
    """Build PostgreSQL config using environment overrides."""
    return PostgreSQLConfig(
        host=os.getenv("POSTGRES_HOST", PostgreSQLConfig.model_fields["host"].default),
        port=str(os.getenv("POSTGRES_PORT", PostgreSQLConfig.model_fields["port"].default)),
        database=os.getenv("POSTGRES_DB", PostgreSQLConfig.model_fields["database"].default),
        user=os.getenv("POSTGRES_USER", PostgreSQLConfig.model_fields["user"].default),
        password=os.getenv("POSTGRES_PASSWORD", PostgreSQLConfig.model_fields["password"].default),
    )


@pytest.fixture(scope="module")
def db():
    """Provide a PostgreSQL connection or skip if unavailable."""
    config = _build_postgres_config()
    try:
        return config.build()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL unavailable: {exc}")


@pytest.fixture
def source_file(db: PostgreSQLDB):
    """Insert a temporary file metadata row for downstream tests."""
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    file_id = f"test-file-{uuid.uuid4()}"
    file_metadata = FileMetadata(
        file_id=file_id,
        blob_key=f"assets/te/{file_id}/document.pdf",
        filename="document.pdf",
        status=FileStatus.STORED,
        file_size=1024000,
        content_type="application/pdf",
        created_at=now,
        updated_at=now
    )
    db.store_file_metadata(file_metadata)
    try:
        yield file_metadata
    finally:
        db.delete_file_metadata(file_id)


@pytest.fixture
def source_parsed(db: PostgreSQLDB, source_file: FileMetadata):
    """Insert a temporary parsed content row for chunk tests."""
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    parsed_id = f"parsed-{uuid.uuid4()}"
    parsed_metadata = ParsedContentMetadata(
        parsed_content_id=parsed_id,
        source_file_id=source_file.file_id,
        blob_key=f"parsed/te/{source_file.file_id}/{parsed_id}.markdown",
        parser_type="pdf_parser",
        status=ParsedContentStatus.STORED,
        content_type="text/markdown",
        created_at=now,
        updated_at=now
    )
    db.store_parsed_content_metadata(parsed_metadata)
    try:
        yield parsed_metadata
    finally:
        db.delete_parsed_content_metadata(parsed_id)

def test_file_metadata_operations(db: PostgreSQLDB):
    """Test complete CRUD operations for FileMetadata"""
    print("\n=== FILE METADATA OPERATIONS ===")

    # Create test file metadata
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    file_id = f"test-file-{uuid.uuid4()}"
    file_metadata = FileMetadata(
        file_id=file_id,
        blob_key=f"assets/te/{file_id}/document.pdf",
        filename="document.pdf",
        status=FileStatus.STORED,  # Using new enum value
        file_size=1024000,
        content_type="application/pdf",
        created_at=now,
        updated_at=now
    )

    # Test CREATE
    print("1. Testing file metadata creation...")
    stored_id = db.store_file_metadata(file_metadata)
    print(f"    Stored file metadata with ID: {stored_id}")

    # Test READ
    print("2. Testing file metadata retrieval...")
    retrieved = db.get_file_metadata(file_id)
    if retrieved:
        print(f"  Retrieved: {retrieved.filename} - Status: {retrieved.status}")
        print(f"  Status type: {type(retrieved.status)} (should be FileStatus enum)")
    else:
        print("  Failed to retrieve file metadata")

    # Test UPDATE
    print("3. Testing file metadata update...")
    success = db.update_file_metadata(file_id, {
        "status": FileStatus.PARSED,  # Using new enum progression
        "blob_key": f"assets/te/{file_id}/document-v2.pdf"
    })
    if success:
        updated = db.get_file_metadata("test-file-123")
        print(f"  Updated status to: {updated.status}")
        print(f"  Updated blob_key to: {updated.blob_key}")
    else:
        print("  Failed to update file metadata")

    # Test LIST
    print("4. Testing file metadata listing...")
    file_list = db.list_file_metadata(status=FileStatus.PARSED, limit=5)
    print(f"  Found {len(file_list)} parsed files")
    for f in file_list:
        print(f"     - {f.filename} ({f.status})")

    # Cleanup
    db.delete_file_metadata(file_id)
    return retrieved

def test_parsed_content_operations(db: PostgreSQLDB, source_file: FileMetadata):
    """Test complete CRUD operations for ParsedContentMetadata"""
    print("\n=== PARSED CONTENT METADATA OPERATIONS ===")

    # Create test parsed content metadata
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    parsed_id = f"parsed-{uuid.uuid4()}"
    parsed_metadata = ParsedContentMetadata(
        parsed_content_id=parsed_id,
        source_file_id=source_file.file_id,
        blob_key=f"parsed/te/{source_file.file_id}/{parsed_id}.markdown",
        parser_type="pdf_parser",
        status=ParsedContentStatus.STORED,  # Using new enum value
        content_type="text/markdown",
        created_at=now,
        updated_at=now
    )

    # Test CREATE
    print("1. Testing parsed content creation...")
    stored_id = db.store_parsed_content_metadata(parsed_metadata)
    print(f"  Stored parsed content with ID: {stored_id}")

    # Test READ
    print("2. Testing parsed content retrieval...")
    retrieved = db.get_parsed_content_metadata(parsed_id)
    if retrieved:
        print(f"  Retrieved: {retrieved.parser_type} - Status: {retrieved.status}")
        print(f"  Content type: {retrieved.content_type}")
    else:
        print("  Failed to retrieve parsed content metadata")

    # Test UPDATE
    print("3. Testing parsed content update...")
    success = db.update_parsed_content_metadata(parsed_id, {
        "status": ParsedContentStatus.PARSED,  # Progress to next status
        "parser_type": "updated_pdf_parser"
    })
    if success:
        updated = db.get_parsed_content_metadata("parsed-123")
        print(f"  Updated status to: {updated.status}")
        print(f"  Updated parser_type to: {updated.parser_type}")
    else:
        print("  Failed to update parsed content metadata")

    # Test LIST
    print("4. Testing parsed content listing...")
    parsed_list = db.list_parsed_content_metadata(
        source_file_id=source_file.file_id,
        status=ParsedContentStatus.PARSED,
        limit=5
    )
    print(f"  Found {len(parsed_list)} parsed content for source file")

    db.delete_parsed_content_metadata(parsed_id)
    return retrieved

def test_chunk_operations(db: PostgreSQLDB, source_parsed: ParsedContentMetadata):
    """Test complete CRUD operations for ChunkMetadata"""
    print("\n=== CHUNK METADATA OPERATIONS ===")

    # Create test chunk metadata
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    chunk_id = f"chunk-{uuid.uuid4()}"
    chunk_metadata = ChunkMetadata(
        chunk_id=chunk_id,
        source_parsed_content_id=source_parsed.parsed_content_id,  # Back to ParsedContent reference
        blob_key=f"chunks/pa/{source_parsed.parsed_content_id}/{chunk_id}.json",
        chunker_type="semantic",
        index_status=ChunkIndexStatus.STORED,  # Using new enum value
        created_at=now,
        indexed_at=None  # Not indexed yet
    )

    # Test CREATE
    print("1. Testing chunk creation...")
    stored_id = db.store_chunk_metadata(chunk_metadata)
    print(f"  Stored chunk with ID: {stored_id}")
    print(f"  Chunk linked to parsed content: {chunk_metadata.source_parsed_content_id}")

    # Test READ
    print("2. Testing chunk retrieval...")
    retrieved = db.get_chunk_metadata(chunk_id)
    if retrieved:
        print(f"  Retrieved: {retrieved.chunker_type} - Status: {retrieved.index_status}")
        print(f"  Created at: {retrieved.created_at}")
        print(f"  Linked to parsed content: {retrieved.source_parsed_content_id}")
    else:
        print("  Failed to retrieve chunk metadata")

    # Test UPDATE (simulate indexing)
    print("3. Testing chunk update (indexing simulation)...")
    success = db.update_chunk_metadata(chunk_id, {
        "index_status": ChunkIndexStatus.INDEXED,  # Using enum directly
        "indexed_at": datetime.now(tz=datetime.now().astimezone().tzinfo)
    })
    if success:
        updated = db.get_chunk_metadata("chunk-123")
        print(f"  Updated status to: {updated.index_status}")
        print(f"  Indexed at: {updated.indexed_at}")
    else:
        print("  Failed to update chunk metadata")

    # Test LIST
    print("4. Testing chunk listing...")
    chunk_list = db.list_chunk_metadata(
        source_parsed_content_id=source_parsed.parsed_content_id,  # Back to parsed content parameter
        index_status=ChunkIndexStatus.INDEXED,
        limit=5
    )
    print(f"    Found {len(chunk_list)} indexed chunks for source parsed content")

    db.delete_chunk_metadata(chunk_id)
    return retrieved

def test_cleanup_operations(db: PostgreSQLDB):
    """Test cleanup by deleting all test data"""
    print("\n=== CLEANUP OPERATIONS ===")

    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    file_id = f"cleanup-file-{uuid.uuid4()}"
    parsed_id = f"cleanup-parsed-{uuid.uuid4()}"
    chunk_id = f"cleanup-chunk-{uuid.uuid4()}"

    db.store_file_metadata(FileMetadata(
        file_id=file_id,
        blob_key=f"assets/cleanup/{file_id}/document.pdf",
        filename="cleanup.pdf",
        status=FileStatus.STORED,
        file_size=1024,
        content_type="application/pdf",
        created_at=now,
        updated_at=now,
    ))

    db.store_parsed_content_metadata(ParsedContentMetadata(
        parsed_content_id=parsed_id,
        source_file_id=file_id,
        blob_key=f"parsed/cleanup/{file_id}/{parsed_id}.markdown",
        parser_type="pdf_parser",
        status=ParsedContentStatus.STORED,
        content_type="text/markdown",
        created_at=now,
        updated_at=now,
    ))

    db.store_chunk_metadata(ChunkMetadata(
        chunk_id=chunk_id,
        source_parsed_content_id=parsed_id,
        blob_key=f"chunks/cleanup/{parsed_id}/{chunk_id}.json",
        chunker_type="semantic",
        index_status=ChunkIndexStatus.STORED,
        created_at=now,
        indexed_at=None,
    ))

    print("1. Deleting chunk metadata...")
    chunk_deleted = db.delete_chunk_metadata(chunk_id)
    print(f"  Chunk deleted: {chunk_deleted}")

    print("2. Deleting parsed content metadata...")
    parsed_deleted = db.delete_parsed_content_metadata(parsed_id)
    print(f"  Parsed content deleted: {parsed_deleted}")

    print("3. Deleting file metadata...")
    file_deleted = db.delete_file_metadata(file_id)
    print(f"  File metadata deleted: {file_deleted}")

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
    from encapsulation.data_model.orm_models import Base
    Base.metadata.drop_all(db.engine)
    Base.metadata.create_all(db.engine)
    print(" Tables recreated with latest schema")

    try:
        # Test file metadata operations
        file_metadata = test_file_metadata_operations(db)

        # Test parsed content operations (depends on file)
        parsed_metadata = test_parsed_content_operations(db, file_metadata)

        # Test chunk operations (depends on parsed content)
        chunk_metadata = test_chunk_operations(db, parsed_metadata)

        # Show final state
        print("\n=== FINAL STATE VERIFICATION ===")
        final_file = db.get_file_metadata("test-file-123")
        final_parsed = db.get_parsed_content_metadata("parsed-123")
        final_chunk = db.get_chunk_metadata("chunk-123")

        print(f"File status: {final_file.status} (type: {type(final_file.status)})")
        print(f"Parsed status: {final_parsed.status} (type: {type(final_parsed.status)})")
        print(f"Chunk status: {final_chunk.index_status}, Indexed at: {final_chunk.indexed_at}")

        # Cleanup test data
        # test_cleanup_operations(db)

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Try cleanup even if tests failed
        try:
            test_cleanup_operations(db)
        except:
            print("Cleanup also failed - you may need to manually clean test data")

if __name__ == "__main__":
    main()
