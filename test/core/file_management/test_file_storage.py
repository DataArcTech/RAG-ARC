"""
Test for FileStorage - testing the core file management interface methods
"""

import json

from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.core.file_management.file_storage_config import FileStorageConfig


def main():
    print("Testing FileStorage - Core File Management Interface Methods")

    file_db_config = LocalDBConfig()
    relational_db_config = PostgreSQLConfig()
    config = FileStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=relational_db_config
    )

    try:
        """Test the FileStorage interface methods"""
        print("=== Testing FileStorage Interface ===")

        # Test data
        test_file_data = b"This is a test document for file storage operations."
        test_filename = "test_document.txt"
        test_content_type = "text/plain"

        # 1. Test build and initialization
        print("\n--- Test 1: build ---")
        file_storage = config.build()
        print(f"  FileStorage built from config")
        print(f"  Blob store initialized: {file_storage.blob_store is not None}")
        print(f"  Metadata store initialized: {file_storage.metadata_store is not None}")
        print(f"  Blob storage path: {file_db_config.base_path}")
        print(f"  Database: {relational_db_config.database}")

        # Setup database schema
        print("  Setting up database schema...")
        from encapsulation.data_model.orm_models import Base
        Base.metadata.drop_all(file_storage.metadata_store.engine)
        Base.metadata.create_all(file_storage.metadata_store.engine)
        print("  Database schema recreated")

        # 2. Test upload_file
        print("\n--- Test 2: upload_file ---")
        file_id = file_storage.upload_file(
            filename=test_filename,
            file_data=test_file_data,
            content_type=test_content_type,
            validate_after_store=True
        )
        print(f"  Uploaded file: {test_filename}")
        print(f"  File ID: {file_id}")

        # Get file metadata to display details
        file_metadata = file_storage.get_file_metadata(file_id)
        print(f"  Status: {file_metadata.status}")
        print(f"  File size: {file_metadata.file_size} bytes")
        print(f"  Content type: {file_metadata.content_type}")
        print(f"  Blob key: {file_metadata.blob_key}")

        # 3. Test get_file_metadata
        print("\n--- Test 3: get_file_metadata ---")
        retrieved_metadata = file_storage.get_file_metadata(file_id)
        print(f"  Retrieved metadata for file ID: {file_id}")
        print(f"  Filename: {retrieved_metadata.filename}")
        print(f"  Status: {retrieved_metadata.status}")
        print(f"  Created at: {retrieved_metadata.created_at}")
        print(f"  Updated at: {retrieved_metadata.updated_at}")

        # Test non-existent file
        nonexistent_metadata = file_storage.get_file_metadata("00000000-0000-0000-0000-000000000000")
        print(f"  Non-existent file metadata: {nonexistent_metadata}")

        # 4. Test get_file_content
        print("\n--- Test 4: get_file_content ---")
        retrieved_content = file_storage.get_file_content(file_id)
        print(f"  Retrieved content length: {len(retrieved_content)} bytes")
        print(f"  Content matches original: {retrieved_content == test_file_data}")
        print(f"  Content preview: {retrieved_content[:50].decode('utf-8')}...")

        # Test non-existent file content
        nonexistent_content = file_storage.get_file_content("00000000-0000-0000-0000-000000000000")
        print(f"  Non-existent file content: {nonexistent_content}")

        # 5. Test additional file uploads for testing other methods
        print("\n--- Test 5: upload_additional_files ---")
        uploaded_files = []

        additional_files = [
            ("doc1.txt", b"Content of document 1", "text/plain"),
            ("doc2.md", b"# Document 2\nMarkdown content", "text/markdown"),
            ("doc3.txt", b"Content of document 3", "text/plain")
        ]

        for filename, file_data, content_type in additional_files:
            try:
                file_id = file_storage.upload_file(
                    filename=filename,
                    file_data=file_data,
                    content_type=content_type,
                    validate_after_store=True
                )
                file_metadata = file_storage.get_file_metadata(file_id)
                uploaded_files.append(file_metadata)
                print(f"  Uploaded {filename}: {file_id}")
            except Exception as e:
                print(f"  Failed to upload {filename}: {e}")

        print(f"  Successfully uploaded files: {len(uploaded_files)}")
        print(f"  File IDs: {[f.file_id for f in uploaded_files]}")

        # 6. Test store_parsed_content
        print("\n--- Test 6: store_parsed_content ---")
        parsed_data = b"# Parsed Content\n\nThis is parsed content from the original file."
        parsed_content_id = file_storage.store_parsed_content(
            source_file_id=file_id,
            parser_type="test_parser",
            parsed_data=parsed_data,
            content_type="text/markdown",
            validate_after_store=True
        )
        print(f"  Stored parsed content ID: {parsed_content_id}")

        # Get parsed content metadata to display details
        parsed_metadata = file_storage.get_parsed_content_metadata(parsed_content_id)
        print(f"  Source file ID: {parsed_metadata.source_file_id}")
        print(f"  Parser type: {parsed_metadata.parser_type}")
        print(f"  Status: {parsed_metadata.status}")
        print(f"  Content type: {parsed_metadata.content_type}")

        # 7. Test get_parsed_content_metadata
        print("\n--- Test 7: get_parsed_content_metadata ---")
        retrieved_parsed_metadata = file_storage.get_parsed_content_metadata(parsed_content_id)
        print(f"  Retrieved parsed metadata for ID: {parsed_content_id}")
        print(f"  Parser type: {retrieved_parsed_metadata.parser_type}")
        print(f"  Source file ID: {retrieved_parsed_metadata.source_file_id}")
        print(f"  Created at: {retrieved_parsed_metadata.created_at}")

        # 8. Test get_parsed_content
        print("\n--- Test 8: get_parsed_content ---")
        retrieved_parsed_content = file_storage.get_parsed_content(parsed_content_id)
        print(f"  Retrieved parsed content length: {len(retrieved_parsed_content)} bytes")
        print(f"  Content matches original: {retrieved_parsed_content == parsed_data}")
        print(f"  Content preview: {retrieved_parsed_content[:50].decode('utf-8')}...")

        # 9. Test additional parsed content storage
        print("\n--- Test 9: store_additional_parsed_content ---")
        parsed_contents = []

        # Use first 2 uploaded files for testing
        for i, uploaded_file in enumerate(uploaded_files[:2]):
            try:
                parsed_data = f"Batch parsed content {i+1}".encode('utf-8')
                parsed_content_id = file_storage.store_parsed_content(
                    source_file_id=uploaded_file.file_id,
                    parser_type="batch_parser",
                    parsed_data=parsed_data,
                    content_type="text/markdown",
                    validate_after_store=True
                )
                parsed_metadata = file_storage.get_parsed_content_metadata(parsed_content_id)
                parsed_contents.append(parsed_metadata)
                print(f"  Stored parsed content {i+1}: {parsed_content_id}")
            except Exception as e:
                print(f"  Failed to store parsed content {i+1}: {e}")

        print(f"  Successfully stored parsed contents: {len(parsed_contents)}")
        print(f"  Parsed content IDs: {[p.parsed_content_id for p in parsed_contents]}")

        # 10. Test store_chunk
        print("\n--- Test 10: store_chunk ---")
        chunk_data = {
            "chunk_id": 0,
            "content": "This is the first chunk of parsed content.",
            "metadata": {
                "start_pos": 0,
                "end_pos": 42,
                "tokens": 8,
                "source": parsed_content_id
            }
        }
        chunk_bytes = json.dumps(chunk_data).encode('utf-8')

        chunk_id = file_storage.store_chunk(
            source_parsed_content_id=parsed_content_id,
            chunker_type="test_chunker",
            chunk_data=chunk_bytes,
            validate_after_store=True
        )
        print(f"  Stored chunk ID: {chunk_id}")

        # Get chunk metadata to display details
        chunk_metadata = file_storage.get_chunk_metadata(chunk_id)
        print(f"  Source parsed content ID: {chunk_metadata.source_parsed_content_id}")
        print(f"  Chunker type: {chunk_metadata.chunker_type}")
        print(f"  Index status: {chunk_metadata.index_status}")

        # 11. Test get_chunk_metadata
        print("\n--- Test 11: get_chunk_metadata ---")
        retrieved_chunk_metadata = file_storage.get_chunk_metadata(chunk_id)
        print(f"  Retrieved chunk metadata for ID: {chunk_id}")
        print(f"  Chunker type: {retrieved_chunk_metadata.chunker_type}")
        print(f"  Source parsed content ID: {retrieved_chunk_metadata.source_parsed_content_id}")
        print(f"  Created at: {retrieved_chunk_metadata.created_at}")

        # 12. Test get_chunk_content
        print("\n--- Test 12: get_chunk_content ---")
        retrieved_chunk_content = file_storage.get_chunk_content(chunk_id)
        print(f"  Retrieved chunk content length: {len(retrieved_chunk_content)} bytes")
        print(f"  Content matches original: {retrieved_chunk_content == chunk_bytes}")

        # Parse and verify JSON structure
        chunk_json = json.loads(retrieved_chunk_content.decode('utf-8'))
        print(f"  Chunk content: {chunk_json['content']}")
        print(f"  Chunk metadata: {chunk_json['metadata']}")

        # 13. Test additional chunk storage
        print("\n--- Test 13: store_additional_chunks ---")
        stored_chunks = []

        # Store multiple chunks individually
        for i in range(3):
            try:
                chunk_data = {
                    "chunk_id": i,
                    "content": f"Batch chunk {i+1} content",
                    "metadata": {"batch_test": True, "chunk_index": i}
                }
                chunk_bytes = json.dumps(chunk_data).encode('utf-8')

                chunk_id = file_storage.store_chunk(
                    source_parsed_content_id=parsed_content_id,
                    chunker_type="batch_chunker",
                    chunk_data=chunk_bytes,
                    validate_after_store=True
                )
                chunk_metadata = file_storage.get_chunk_metadata(chunk_id)
                stored_chunks.append(chunk_metadata)
                print(f"  Stored chunk {i+1}: {chunk_id}")
            except Exception as e:
                print(f"  Failed to store chunk {i+1}: {e}")

        print(f"  Successfully stored chunks: {len(stored_chunks)}")
        print(f"  Chunk IDs: {[c.chunk_id for c in stored_chunks]}")

        # 14. Test delete operations
        print("\n--- Test 14: delete operations ---")

        # Test delete_chunk
        if stored_chunks:
            test_chunk = stored_chunks[0]
            delete_chunk_result = file_storage.delete_chunk(test_chunk.chunk_id)
            print(f"  Delete chunk result: {delete_chunk_result}")

            # Verify deletion
            deleted_chunk_metadata = file_storage.get_chunk_metadata(test_chunk.chunk_id)
            print(f"  Deleted chunk metadata: {deleted_chunk_metadata}")

        # Test delete_parsed_content
        if parsed_contents:
            test_parsed = parsed_contents[0]
            delete_parsed_result = file_storage.delete_parsed_content(test_parsed.parsed_content_id)
            print(f"  Delete parsed content result: {delete_parsed_result}")

        # Test delete_file
        if uploaded_files:
            test_file = uploaded_files[0]
            delete_file_result = file_storage.delete_file(test_file.file_id)
            print(f"  Delete file result: {delete_file_result}")

        print("\n All FileStorage methods tested successfully!")
        print(f"\n Files stored in: {file_db_config.base_path}")
        print("You can inspect the stored files and directory structure")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        print("Make sure PostgreSQL is running with the configured credentials:")
        print(f"  Host: {relational_db_config.host}:{relational_db_config.port}")
        print(f"  Database: {relational_db_config.database}")
        print(f"  User: {relational_db_config.user}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()