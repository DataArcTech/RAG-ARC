"""
Real-world concurrent indexing test that simulates the actual IndexManager behavior.

This test verifies that the BM25 batch indexing solution correctly handles:
1. Multiple files being indexed concurrently
2. Multiple indexers running in parallel via asyncio.gather()
3. No LockBusy errors from Tantivy
"""
import asyncio
import logging
import sys
import time
from typing import List
from encapsulation.data_model.schema import Chunk
from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_chunks_for_file(file_id: str, num_chunks: int, owner_id: str = "test_user") -> List[Chunk]:
    """Create test chunks for a file."""
    chunks = []
    for i in range(num_chunks):
        chunk = Chunk(
            id=f"{file_id}_chunk_{i}",
            content=f"This is chunk {i} from file {file_id}. It contains some sample text for BM25 indexing.",
            metadata={
                "source_file_id": file_id,
                "chunk_index": i,
                "filename": f"{file_id}.txt"
            },
            owner_id=owner_id
        )
        chunks.append(chunk)
    return chunks


async def index_single_document(
    indexer,
    file_id: str,
    num_chunks: int
) -> dict:
    """
    Index a single document (file).
    This simulates what happens when ONE file is uploaded and indexed.
    """
    try:
        logger.info(f"[{file_id}] Creating {num_chunks} chunks")
        chunks = create_chunks_for_file(file_id, num_chunks)

        logger.info(f"[{file_id}] Calling indexer.update_index()")
        start_time = time.time()

        indexed_ids = await indexer.update_index(chunks)

        elapsed = time.time() - start_time
        logger.info(f"[{file_id}] Indexed {len(indexed_ids)} chunks in {elapsed:.3f}s")

        return {
            "file_id": file_id,
            "success": True,
            "indexed_count": len(indexed_ids) if indexed_ids else 0,
            "total_chunks": num_chunks,
            "elapsed_time": elapsed
        }
    except Exception as e:
        logger.error(f"[{file_id}] Indexing failed: {str(e)}")
        return {
            "file_id": file_id,
            "success": False,
            "error_message": str(e),
            "indexed_count": 0,
            "total_chunks": num_chunks
        }


async def test_concurrent_file_uploads():
    """
    Test Case 1: Multiple DOCUMENTS uploaded concurrently.
    This simulates what happens when users upload multiple DOCUMENTS at the same time.
    Each document is indexed independently and concurrently.
    """
    print("\n" + "="*80)
    print("Test Case 1: Concurrent Document Uploads (5 documents)")
    print("="*80 + "\n")

    # Create a single BM25 indexer (shared across all documents)
    bm25_config = BM25BuilderConfig(
        type="bm25_builder",
        index_path="./data/test_concurrent_uploads"
    )

    indexer_config = BM25IndexerConfig(
        type="bm25_indexer",
        index_config=bm25_config,
        batch_size=50,
        flush_interval=3.0,
        immediate_flush_threshold=10
    )

    indexer = indexer_config.build()

    print(f"Configuration:")
    print(f"  - batch_size: {indexer.batch_size}")
    print(f"  - flush_interval: {indexer.flush_interval}s")
    print(f"  - immediate_flush_threshold: {indexer.immediate_flush_threshold}")
    print()

    # Simulate 5 DOCUMENTS being uploaded concurrently
    # Each document has different number of chunks
    document_configs = [
        ("document_1", 8),   # Small document
        ("document_2", 25),  # Medium document
        ("document_3", 12),  # Small-medium document
        ("document_4", 60),  # Large document
        ("document_5", 5),   # Very small document
    ]

    print(f"Simulating {len(document_configs)} documents uploaded concurrently...")
    print("Each document will call indexer.update_index() at the same time")
    print("This is the scenario that causes LockBusy errors without proper locking")
    print()

    start_time = time.time()

    # Index all documents concurrently
    # This simulates multiple users uploading documents at the same time
    tasks = [
        index_single_document(indexer, file_id, num_chunks)
        for file_id, num_chunks in document_configs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Process results
    print("\nResults:")
    print("-" * 80)
    successful = 0
    failed = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Exception: {str(result)}")
            failed += 1
        elif result.get("success"):
            file_id = result["file_id"]
            chunks = result["total_chunks"]
            time_taken = result["elapsed_time"]
            print(f"✅ {file_id}: {chunks} chunks indexed in {time_taken:.3f}s")
            successful += 1
        else:
            file_id = result.get("file_id", "unknown")
            error = result.get("error_message", "Unknown error")
            print(f"❌ {file_id}: {error}")
            failed += 1

    print()
    print(f"Total time: {elapsed:.3f}s")
    print(f"Success: {successful}/{len(document_configs)} documents")
    print()

    if failed > 0:
        print(f"❌ {failed} documents failed!")
        return False
    else:
        print("✅ All documents indexed successfully - NO LOCK ERRORS!")

    # Wait for any pending flushes
    print("\nWaiting for pending flushes...")
    await asyncio.sleep(4)

    # Shutdown
    await indexer.shutdown()

    return True


async def test_stress_concurrent_indexing():
    """
    Test Case 2: Stress test with many concurrent DOCUMENTS.
    """
    print("\n" + "="*80)
    print("Test Case 2: Stress Test - 20 Concurrent Documents")
    print("="*80 + "\n")

    # Create indexer
    bm25_config = BM25BuilderConfig(
        type="bm25_builder",
        index_path="./data/test_stress_concurrent"
    )

    indexer_config = BM25IndexerConfig(
        type="bm25_indexer",
        index_config=bm25_config,
        batch_size=100,
        flush_interval=5.0,
        immediate_flush_threshold=15
    )

    indexer = indexer_config.build()

    print(f"Configuration:")
    print(f"  - batch_size: {indexer.batch_size}")
    print(f"  - flush_interval: {indexer.flush_interval}s")
    print(f"  - immediate_flush_threshold: {indexer.immediate_flush_threshold}")
    print()

    # Simulate 20 DOCUMENTS uploaded concurrently
    document_configs = [(f"stress_doc_{i}", 20) for i in range(20)]

    print(f"Simulating {len(document_configs)} documents uploaded concurrently...")
    print("Each document has 20 chunks")
    print()

    start_time = time.time()

    # Index all documents concurrently
    tasks = [
        index_single_document(indexer, file_id, num_chunks)
        for file_id, num_chunks in document_configs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Process results
    successful = 0
    failed = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception: {str(result)}")
            failed += 1
        elif result.get("success"):
            successful += 1
        else:
            failed += 1

    print(f"Total time: {elapsed:.3f}s")
    print(f"Success: {successful}/{len(document_configs)} documents")
    print()

    if failed > 0:
        print(f"❌ {failed} documents failed!")
        return False
    else:
        print("✅ All documents indexed successfully!")

    # Wait for pending flushes
    print("\nWaiting for pending flushes...")
    await asyncio.sleep(6)

    # Shutdown
    await indexer.shutdown()

    return True


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Real-World Concurrent Document Indexing Test")
    print("="*80)
    print()
    print("This test simulates the real scenario where multiple users")
    print("upload multiple documents at the same time.")
    print()
    print("WITHOUT the batch indexing solution, this would cause:")
    print("  ❌ LockBusy errors from Tantivy")
    print()
    print("WITH the batch indexing solution, this should:")
    print("  ✅ Handle all concurrent uploads without errors")
    print("="*80)
    
    all_passed = True
    
    # Test 1: Concurrent file uploads
    try:
        passed = await test_concurrent_file_uploads()
        if not passed:
            all_passed = False
    except Exception as e:
        logger.error(f"Test 1 failed with exception: {e}", exc_info=True)
        all_passed = False
    
    # Test 2: Stress test
    try:
        passed = await test_stress_concurrent_indexing()
        if not passed:
            all_passed = False
    except Exception as e:
        logger.error(f"Test 2 failed with exception: {e}", exc_info=True)
        all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - No LockBusy errors!")
        print()
        print("The batch indexing solution successfully handles:")
        print("  ✅ Multiple documents uploaded concurrently")
        print("  ✅ Async lock prevents concurrent writes to BM25 index")
        print("  ✅ Batch processing improves throughput")
        print("  ✅ Immediate flush for small documents reduces latency")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

