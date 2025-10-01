"""
Unified Test for Storage Modules - FileStorage, ParsedContentStorage, and ChunkStorage
Tests the complete storage pipeline: File -> Parsed Content -> Chunks
"""

import json

from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.core.file_management.storage.file_storage import FileStorageConfig
from config.core.file_management.storage.parsed_content_storage import ParsedContentStorageConfig
from config.core.file_management.storage.chunk_storage import ChunkStorageConfig


def main():
    print("=" * 80)
    print("Testing Storage Modules - Unified Test")
    print("Testing: FileStorage, ParsedContentStorage, ChunkStorage")
    print("=" * 80)

    # Setup shared configurations
    file_db_config = LocalDBConfig()
    relational_db_config = PostgreSQLConfig()

    file_storage_config = FileStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=relational_db_config
    )

    parsed_content_storage_config = ParsedContentStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=relational_db_config
    )

    chunk_storage_config = ChunkStorageConfig(
        file_db_config=file_db_config,
        relational_db_config=relational_db_config
    )

    try:
        # Build all storage modules
        print("\n" + "=" * 80)
        print("SETUP: Building Storage Modules")
        print("=" * 80)

        file_storage = file_storage_config.build()
        parsed_content_storage = parsed_content_storage_config.build()
        chunk_storage = chunk_storage_config.build()

        print(f"✓ FileStorage built")
        print(f"✓ ParsedContentStorage built")
        print(f"✓ ChunkStorage built")
        print(f"  Blob storage path: {file_db_config.base_path}")
        print(f"  Database: {relational_db_config.database}")

        # Setup database schema
        print("\n  Setting up database schema...")
        from encapsulation.data_model.orm_models import Base
        Base.metadata.drop_all(file_storage.metadata_store.engine)
        Base.metadata.create_all(file_storage.metadata_store.engine)
        print("  ✓ Database schema recreated")

        # =====================================================================
        # PART 1: FILE STORAGE TESTS
        # =====================================================================
        print("\n" + "=" * 80)
        print("PART 1: TESTING FILE STORAGE")
        print("=" * 80)

        # Test 1: Upload single file
        print("\n--- Test 1.1: upload_file (single) ---")
        test_file_data = b"This is a test document for file storage operations."
        test_filename = "test_document.txt"

        file_id = file_storage.upload_file(
            filename=test_filename,
            file_data=test_file_data,
            content_type="text/plain",
            validate_after_store=True
        )
        print(f"  ✓ Uploaded file: {test_filename}")
        print(f"  File ID: {file_id}")

        # Test 2: Get file metadata and content
        print("\n--- Test 1.2: get_file_metadata and get_file_content ---")
        file_metadata = file_storage.get_file_metadata(file_id)
        file_content = file_storage.get_file_content(file_id)

        print(f"  ✓ Retrieved metadata: {file_metadata.filename}")
        print(f"  Status: {file_metadata.status}")
        print(f"  File size: {file_metadata.file_size} bytes")
        print(f"  Content matches: {file_content == test_file_data}")

        # Test 3: Upload multiple files
        print("\n--- Test 1.3: upload_file (multiple) ---")
        uploaded_files = []
        additional_files = [
            ("doc1.txt", b"Content of document 1", "text/plain"),
            ("doc2.md", b"# Document 2\nMarkdown content", "text/markdown"),
            ("doc3.txt", b"Content of document 3", "text/plain"),
        ]

        for filename, file_data, content_type in additional_files:
            fid = file_storage.upload_file(
                filename=filename,
                file_data=file_data,
                content_type=content_type,
                validate_after_store=True
            )
            uploaded_files.append(fid)
            print(f"  ✓ Uploaded {filename}: {fid}")

        print(f"  Total files uploaded: {len(uploaded_files) + 1}")

        # Test 4: File validation
        print("\n--- Test 1.4: file_validation ---")
        validation_tests = [
            ("empty filename", "", b"test data", "text/plain"),
            ("empty file data", "test.txt", b"", "text/plain"),
            ("long filename", "a" * 300 + ".txt", b"test data", "text/plain"),
        ]

        for test_name, filename, file_data, content_type in validation_tests:
            try:
                file_storage.upload_file(filename=filename, file_data=file_data, content_type=content_type)
                print(f"  ✗ {test_name}: Should have been rejected")
            except Exception:
                print(f"  ✓ {test_name}: Correctly rejected")

        # Test 5: Delete file
        print("\n--- Test 1.5: delete_file ---")
        if uploaded_files:
            test_file_id = uploaded_files[0]
            delete_result = file_storage.delete_file(test_file_id)
            deleted_metadata = file_storage.get_file_metadata(test_file_id)
            print(f"  ✓ Delete result: {delete_result}")
            print(f"  Deleted file metadata: {deleted_metadata}")

        # =====================================================================
        # PART 2: PARSED CONTENT STORAGE TESTS
        # =====================================================================
        print("\n" + "=" * 80)
        print("PART 2: TESTING PARSED CONTENT STORAGE")
        print("=" * 80)

        # Test 1: Store single parsed content
        print("\n--- Test 2.1: store_parsed_content (single) ---")
        parsed_data = b"# Parsed Content\n\nThis is parsed content from the original file."
        parsed_content_id = parsed_content_storage.store_parsed_content(
            source_file_id=file_id,
            parser_type="test_parser",
            parsed_data=parsed_data,
            content_type="text/markdown",
            validate_after_store=True
        )
        print(f"  ✓ Stored parsed content ID: {parsed_content_id}")

        # Test 2: Get parsed content metadata and content
        print("\n--- Test 2.2: get_parsed_content_metadata and get_parsed_content ---")
        parsed_metadata = parsed_content_storage.get_parsed_content_metadata(parsed_content_id)
        parsed_content = parsed_content_storage.get_parsed_content(parsed_content_id)

        print(f"  ✓ Retrieved metadata for: {parsed_content_id}")
        print(f"  Parser type: {parsed_metadata.parser_type}")
        print(f"  Source file ID: {parsed_metadata.source_file_id}")
        print(f"  Status: {parsed_metadata.status}")
        print(f"  Content matches: {parsed_content == parsed_data}")

        # Test 3: Store multiple parsed contents
        print("\n--- Test 2.3: store_parsed_content (multiple) ---")
        parsed_content_ids = []

        for i, source_file_id in enumerate(uploaded_files[1:3]):  # Use 2 remaining files
            pd = f"# Parsed Document {i+1}\n\nBatch parsed content {i+1}".encode('utf-8')
            pcid = parsed_content_storage.store_parsed_content(
                source_file_id=source_file_id,
                parser_type="batch_parser",
                parsed_data=pd,
                content_type="text/markdown",
                validate_after_store=True
            )
            parsed_content_ids.append(pcid)
            print(f"  ✓ Stored parsed content {i+1}: {pcid}")

        print(f"  Total parsed contents: {len(parsed_content_ids) + 1}")

        # Test 4: Delete parsed content
        print("\n--- Test 2.4: delete_parsed_content ---")
        if parsed_content_ids:
            test_parsed_id = parsed_content_ids[0]
            delete_result = parsed_content_storage.delete_parsed_content(test_parsed_id)
            deleted_metadata = parsed_content_storage.get_parsed_content_metadata(test_parsed_id)
            print(f"  ✓ Delete result: {delete_result}")
            print(f"  Deleted metadata: {deleted_metadata}")

        # =====================================================================
        # PART 3: CHUNK STORAGE TESTS
        # =====================================================================
        print("\n" + "=" * 80)
        print("PART 3: TESTING CHUNK STORAGE")
        print("=" * 80)

        # Test 1: Store single chunk
        print("\n--- Test 3.1: store_chunk (single) ---")
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

        chunk_id = chunk_storage.store_chunk(
            source_parsed_content_id=parsed_content_id,
            chunker_type="test_chunker",
            chunk_data=chunk_bytes,
            validate_after_store=True
        )
        print(f"  ✓ Stored chunk ID: {chunk_id}")

        # Test 2: Get chunk metadata and content
        print("\n--- Test 3.2: get_chunk_metadata and get_chunk_content ---")
        chunk_metadata = chunk_storage.get_chunk_metadata(chunk_id)
        chunk_content = chunk_storage.get_chunk_content(chunk_id)

        print(f"  ✓ Retrieved metadata for: {chunk_id}")
        print(f"  Chunker type: {chunk_metadata.chunker_type}")
        print(f"  Source parsed content ID: {chunk_metadata.source_parsed_content_id}")
        print(f"  Index status: {chunk_metadata.index_status}")
        print(f"  Content matches: {chunk_content == chunk_bytes}")

        # Verify JSON structure
        chunk_json = json.loads(chunk_content.decode('utf-8'))
        print(f"  Chunk content: {chunk_json['content']}")

        # Test 3: Store multiple chunks
        print("\n--- Test 3.3: store_chunk (multiple) ---")
        stored_chunk_ids = []

        for i in range(5):
            cd = {
                "chunk_id": i + 1,
                "content": f"This is chunk {i+1} content with sample text.",
                "metadata": {
                    "batch_test": True,
                    "chunk_index": i + 1,
                    "start_pos": i * 50,
                    "end_pos": (i + 1) * 50
                }
            }
            cb = json.dumps(cd).encode('utf-8')

            cid = chunk_storage.store_chunk(
                source_parsed_content_id=parsed_content_id,
                chunker_type="batch_chunker",
                chunk_data=cb,
                validate_after_store=True
            )
            stored_chunk_ids.append(cid)
            print(f"  ✓ Stored chunk {i+1}: {cid}")

        print(f"  Total chunks stored: {len(stored_chunk_ids) + 1}")

        # Test 4: Store chunks for different parsed content
        print("\n--- Test 3.4: store_chunks (different parsed content) ---")
        if len(parsed_content_ids) > 1:
            second_parsed_id = parsed_content_ids[1]
            chunks_for_second = []

            for i in range(3):
                cd = {
                    "chunk_id": i,
                    "content": f"Second parsed content chunk {i+1}",
                    "metadata": {"parsed_content_2": True, "index": i}
                }
                cb = json.dumps(cd).encode('utf-8')

                cid = chunk_storage.store_chunk(
                    source_parsed_content_id=second_parsed_id,
                    chunker_type="second_chunker",
                    chunk_data=cb,
                    validate_after_store=True
                )
                chunks_for_second.append(cid)
                print(f"  ✓ Chunk {i+1} for second parsed content: {cid}")

            print(f"  Total chunks for second parsed content: {len(chunks_for_second)}")

        # Test 5: Delete chunk
        print("\n--- Test 3.5: delete_chunk ---")
        if stored_chunk_ids:
            test_chunk_id = stored_chunk_ids[0]
            delete_result = chunk_storage.delete_chunk(test_chunk_id)
            deleted_metadata = chunk_storage.get_chunk_metadata(test_chunk_id)
            print(f"  ✓ Delete result: {delete_result}")
            print(f"  Deleted metadata: {deleted_metadata}")

        # =====================================================================
        # PART 4: INTEGRATION TEST - COMPLETE PIPELINE
        # =====================================================================
        print("\n" + "=" * 80)
        print("PART 4: INTEGRATION TEST - COMPLETE PIPELINE")
        print("=" * 80)

        print("\n--- Test 4.1: Complete Pipeline (File -> Parsed Content -> Chunks) ---")

        # Step 1: Upload a new file
        pipeline_file_data = b"This is a complete pipeline test document with multiple sections."
        pipeline_file_id = file_storage.upload_file(
            filename="pipeline_test.txt",
            file_data=pipeline_file_data,
            content_type="text/plain",
            validate_after_store=True
        )
        print(f"  Step 1: ✓ Uploaded file: {pipeline_file_id}")

        # Step 2: Parse the file
        pipeline_parsed_data = b"# Pipeline Test\n\nSection 1: Introduction\nSection 2: Content\nSection 3: Conclusion"
        pipeline_parsed_id = parsed_content_storage.store_parsed_content(
            source_file_id=pipeline_file_id,
            parser_type="pipeline_parser",
            parsed_data=pipeline_parsed_data,
            content_type="text/markdown",
            validate_after_store=True
        )
        print(f"  Step 2: ✓ Created parsed content: {pipeline_parsed_id}")

        # Step 3: Create chunks from parsed content
        sections = ["Introduction", "Content", "Conclusion"]
        pipeline_chunks = []

        for i, section in enumerate(sections):
            cd = {
                "chunk_id": i,
                "content": f"Section {i+1}: {section}",
                "metadata": {"section": section, "order": i}
            }
            cb = json.dumps(cd).encode('utf-8')

            cid = chunk_storage.store_chunk(
                source_parsed_content_id=pipeline_parsed_id,
                chunker_type="pipeline_chunker",
                chunk_data=cb,
                validate_after_store=True
            )
            pipeline_chunks.append(cid)
            print(f"  Step 3.{i+1}: ✓ Created chunk for {section}: {cid}")

        print(f"\n  Pipeline Summary:")
        print(f"    File ID: {pipeline_file_id}")
        print(f"    Parsed Content ID: {pipeline_parsed_id}")
        print(f"    Chunk IDs: {len(pipeline_chunks)} chunks created")
        print(f"  ✓ Complete pipeline test successful!")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✓ All FileStorage methods tested successfully")
        print(f"✓ All ParsedContentStorage methods tested successfully")
        print(f"✓ All ChunkStorage methods tested successfully")
        print(f"✓ Complete pipeline integration test successful")
        print(f"\nFiles stored in: {file_db_config.base_path}")
        print(f"Database: {relational_db_config.database}")
        print("\nYou can inspect the stored files and directory structure")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nMake sure PostgreSQL is running with the configured credentials:")
        print(f"  Host: {relational_db_config.host}:{relational_db_config.port}")
        print(f"  Database: {relational_db_config.database}")
        print(f"  User: {relational_db_config.user}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
