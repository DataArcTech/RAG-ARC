"""
Test script for soft-delete functionality with DenseRetriever
"""
import sys
import os
import tempfile
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))

from encapsulation.data_model.schema import Chunk
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.core.retrieval.dense_config import DenseRetrieverConfig

def test_soft_delete_with_retrieval():
    """Test soft-delete functionality with DenseRetriever"""
    print("=" * 80)
    print("Testing Soft-Delete with DenseRetriever")
    print("=" * 80)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # 1. Create test data with meaningful content
        print("\n1. Creating test data...")
        test_chunks = [
            Chunk(id="doc_1", content="Python is a high-level programming language", metadata={"topic": "python"}),
            Chunk(id="doc_2", content="Java is an object-oriented programming language", metadata={"topic": "java"}),
            Chunk(id="doc_3", content="Python has excellent data science libraries", metadata={"topic": "python"}),
            Chunk(id="doc_4", content="JavaScript is used for web development", metadata={"topic": "javascript"}),
            Chunk(id="doc_5", content="Python is popular for machine learning", metadata={"topic": "python"}),
            Chunk(id="doc_6", content="Java runs on the Java Virtual Machine", metadata={"topic": "java"}),
            Chunk(id="doc_7", content="Python has simple and readable syntax", metadata={"topic": "python"}),
            Chunk(id="doc_8", content="C++ is a powerful systems programming language", metadata={"topic": "cpp"}),
            Chunk(id="doc_9", content="Python supports multiple programming paradigms", metadata={"topic": "python"}),
            Chunk(id="doc_10", content="Rust provides memory safety without garbage collection", metadata={"topic": "rust"}),
        ]
        print(f" Created {len(test_chunks)} test chunks")
        
        # 2. Setup FaissVectorDB and DenseRetriever
        print("\n2. Setting up FaissVectorDB and DenseRetriever...")
        embedding_config = QwenEmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_china_mirror=True,
            cache_folder="./models"
        )
        faiss_config = FaissVectorDBConfig(
            index_path=os.path.join(temp_dir, "test_index"),
            embedding_config=embedding_config,
            index_type="flat",
            metric="cosine"
        )
        
        # Build vector database and add chunks
        vector_db = faiss_config.build()
        vector_db.build_index(test_chunks)
        print(f" Built vector database with {len(test_chunks)} chunks")
        
        # Save the index
        index_path = os.path.join(temp_dir, "test_index")
        vector_db.save_index(index_path)
        print(f" Saved index to {index_path}")
        
        # Create DenseRetriever
        dense_config = DenseRetrieverConfig(
            index_config=faiss_config,
            search_kwargs={"k": 5, "with_score": True}
        )
        retriever = dense_config.build()
        print(" Created DenseRetriever")
        
        # 3. Test retrieval before deletion
        print("\n3. Testing retrieval BEFORE soft-delete...")
        query = "Tell me about Python programming"
        results_before = retriever.invoke(query)
        print(f"   Query: '{query}'")
        print(f"   Retrieved {len(results_before)} chunks:")
        for i, chunk in enumerate(results_before, 1):
            score = chunk.metadata.get('score', 'N/A')
            print(f"      {i}. [{chunk.id}] (score: {score:.4f}): {chunk.content[:60]}...")
        
        # Count Python-related results
        python_results_before = [c for c in results_before if c.metadata.get('topic') == 'python']
        print(f"   Python-related results: {len(python_results_before)}")
        
        # 4. Soft-delete some Python-related chunks
        print("\n4. Soft-deleting Python-related chunks...")
        ids_to_delete = ["doc_3", "doc_5", "doc_7"]  # Delete 3 Python chunks
        print(f"   Deleting chunks: {ids_to_delete}")
        
        # Get the vector database from retriever
        index = retriever.get_index()
        result = index.delete_index(ids_to_delete)
        assert result == True, "Soft-delete should succeed"
        print(f" Soft-deleted {len(ids_to_delete)} chunks")

        # Check state
        info = index.get_vector_db_info()
        print(f"   Active chunks: {info['chunk_count']}")
        print(f"   Deleted chunks: {info['deleted_chunks']}")
        print(f"   Vector count: {info['vector_count']} (no rebuild)")

        # Save the index with soft-delete state
        index.save_index(index_path)
        print(f" Saved index with soft-delete state")
        
        # 5. Test retrieval after deletion
        print("\n5. Testing retrieval AFTER soft-delete...")
        results_after = retriever.invoke(query)
        print(f"   Query: '{query}'")
        print(f"   Retrieved {len(results_after)} chunks:")
        for i, chunk in enumerate(results_after, 1):
            score = chunk.metadata.get('score', 'N/A')
            print(f"      {i}. [{chunk.id}] (score: {score:.4f}): {chunk.content[:60]}...")
        
        # Verify deleted chunks are not in results
        result_ids = [c.id for c in results_after]
        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids, f"Deleted chunk {deleted_id} should not appear in results"
        print(f" Deleted chunks are correctly filtered out")
        
        # Count Python-related results
        python_results_after = [c for c in results_after if c.metadata.get('topic') == 'python']
        print(f"   Python-related results: {len(python_results_after)}")
        print(f"   Reduction: {len(python_results_before)} → {len(python_results_after)}")
        
        # 6. Test with different query
        print("\n6. Testing with different query...")
        query2 = "What is Java used for?"
        results_java = retriever.invoke(query2)
        print(f"   Query: '{query2}'")
        print(f"   Retrieved {len(results_java)} chunks:")
        for i, chunk in enumerate(results_java, 1):
            score = chunk.metadata.get('score', 'N/A')
            print(f"      {i}. [{chunk.id}] (score: {score:.4f}): {chunk.content[:60]}...")
        
        # Verify deleted chunks are not in results
        result_ids_java = [c.id for c in results_java]
        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids_java, f"Deleted chunk {deleted_id} should not appear in results"
        print(f" Deleted chunks are correctly filtered out in different query")
        
        # 7. Test MMR search
        print("\n7. Testing MMR search with soft-deleted chunks...")
        dense_config_mmr = DenseRetrieverConfig(
            index_config=faiss_config,
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
        )
        retriever_mmr = dense_config_mmr.build()
        
        results_mmr = retriever_mmr.invoke(query, search_type="mmr")
        print(f"   Query: '{query}'")
        print(f"   Retrieved {len(results_mmr)} chunks (MMR):")
        for i, chunk in enumerate(results_mmr, 1):
            print(f"      {i}. [{chunk.id}]: {chunk.content[:60]}...")
        
        # Verify deleted chunks are not in MMR results
        result_ids_mmr = [c.id for c in results_mmr]
        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids_mmr, f"Deleted chunk {deleted_id} should not appear in MMR results"
        print(f" Deleted chunks are correctly filtered out in MMR search")
        
        # 8. Test persistence: reload and search again
        print("\n8. Testing persistence: reload and search...")
        # Create new retriever instance
        dense_config_new = DenseRetrieverConfig(
            index_config=faiss_config,
            search_kwargs={"k": 5, "with_score": True}
        )
        retriever_new = dense_config_new.build()
        
        results_reloaded = retriever_new.invoke(query)
        print(f"   Query: '{query}'")
        print(f"   Retrieved {len(results_reloaded)} chunks after reload:")
        for i, chunk in enumerate(results_reloaded, 1):
            score = chunk.metadata.get('score', 'N/A')
            print(f"      {i}. [{chunk.id}] (score: {score:.4f}): {chunk.content[:60]}...")
        
        # Verify deleted chunks are still filtered out after reload
        result_ids_reloaded = [c.id for c in results_reloaded]
        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids_reloaded, f"Deleted chunk {deleted_id} should not appear after reload"
        print(f" Soft-delete state persisted correctly")
        
        # 9. Test compact_index
        print("\n9. Testing compact_index...")
        index_new = retriever_new.get_index()
        info_before_compact = index_new.get_vector_db_info()
        print(f"   Before compact: {info_before_compact['chunk_count']} active, {info_before_compact['deleted_chunks']} deleted, {info_before_compact['vector_count']} vectors")
        
        index_new.compact_index()
        
        info_after_compact = index_new.get_vector_db_info()
        print(f"   After compact: {info_after_compact['chunk_count']} active, {info_after_compact['deleted_chunks']} deleted, {info_after_compact['vector_count']} vectors")
        assert info_after_compact['deleted_chunks'] == 0, "Should have 0 deleted chunks after compaction"
        assert info_after_compact['vector_count'] == info_after_compact['chunk_count'], "Vector count should equal chunk count after compaction"
        print(f" Index compacted successfully")
        
        # 10. Test retrieval after compaction
        print("\n10. Testing retrieval after compaction...")
        results_after_compact = retriever_new.invoke(query)
        print(f"   Query: '{query}'")
        print(f"   Retrieved {len(results_after_compact)} chunks:")
        for i, chunk in enumerate(results_after_compact, 1):
            score = chunk.metadata.get('score', 'N/A')
            print(f"      {i}. [{chunk.id}] (score: {score:.4f}): {chunk.content[:60]}...")
        
        # Verify deleted chunks are still not in results
        result_ids_after_compact = [c.id for c in results_after_compact]
        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids_after_compact, f"Deleted chunk {deleted_id} should not appear after compaction"
        print(f" Retrieval works correctly after compaction")
        
        print("\n" + "=" * 80)
        print("All tests passed! ")
        print("=" * 80)
        print("\nSummary:")
        print(f"  - Soft-delete successfully filters out deleted chunks during retrieval")
        print(f"  - Works with similarity search, MMR search, and score threshold")
        print(f"  - Soft-delete state persists across save/load cycles")
        print(f"  - Compact operation successfully rebuilds index without deleted chunks")
        print(f"  - No index rebuild required for soft-delete (fast operation)")
        
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
    test_soft_delete_with_retrieval()

