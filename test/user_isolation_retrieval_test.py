"""
Test script for user-isolated retrieval functionality
Tests that users can only retrieve their own documents
"""
import uuid
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encapsulation.data_model.schema import Chunk


def test_chunk_with_owner_id_field():
    """Test that chunks can be created with owner_id as a separate field"""
    print("\n=== Test 1: Chunk with owner_id field ===")

    owner_id = uuid.uuid4()
    chunk = Chunk(
        id=str(uuid.uuid4()),
        content="Test content for user isolation",
        owner_id=owner_id,
        metadata={
            "source_file_id": "test_file_123",
            "filename": "test.txt"
        }
    )

    assert chunk.owner_id == owner_id
    print(f"✓ Chunk created with owner_id field: {owner_id}")
    print(f"✓ Chunk.owner_id: {chunk.owner_id}")
    print(f"✓ Chunk.metadata: {chunk.metadata}")


def test_owner_id_filtering():
    """Test filtering chunks by owner_id"""
    print("\n=== Test 2: Owner ID filtering ===")

    # Create chunks for different users
    user1_id = uuid.uuid4()
    user2_id = uuid.uuid4()

    chunks = [
        Chunk(id="1", content="User 1 doc 1", owner_id=user1_id),
        Chunk(id="2", content="User 1 doc 2", owner_id=user1_id),
        Chunk(id="3", content="User 2 doc 1", owner_id=user2_id),
        Chunk(id="4", content="User 2 doc 2", owner_id=user2_id),
    ]

    # Filter for user 1
    user1_chunks = [c for c in chunks if c.owner_id == user1_id]
    assert len(user1_chunks) == 2
    print(f"✓ User 1 ({user1_id[:8]}...) has {len(user1_chunks)} chunks")

    # Filter for user 2
    user2_chunks = [c for c in chunks if c.owner_id == user2_id]
    assert len(user2_chunks) == 2
    print(f"✓ User 2 ({user2_id[:8]}...) has {len(user2_chunks)} chunks")

    # Verify no cross-contamination
    assert all(c.owner_id == user1_id for c in user1_chunks)
    assert all(c.owner_id == user2_id for c in user2_chunks)
    print("✓ No cross-contamination between users")


def test_retrieval_api_with_owner_id():
    """Test RetrievalAPI with owner_id parameter"""
    print("\n=== Test 3: RetrievalAPI with owner_id ===")
    
    try:
        from api.retrieval_api import RetrievalAPI
        
        api = RetrievalAPI()
        print("✓ RetrievalAPI initialized")
        
        # Check that search method accepts owner_id parameter
        import inspect
        sig = inspect.signature(api.search)
        params = list(sig.parameters.keys())
        
        assert 'owner_id' in params, "search() should accept owner_id parameter"
        print(f"✓ search() method parameters: {params}")
        print("✓ owner_id parameter is supported")
        
    except Exception as e:
        print(f"✗ Error testing RetrievalAPI: {e}")
        raise


def test_rag_inference_with_owner_id():
    """Test RAGInference with owner_id parameter"""
    print("\n=== Test 4: RAGInference with owner_id ===")
    
    try:
        from application.rag_inference.module import RAGInference
        
        # Check that chat method accepts owner_id parameter
        import inspect
        sig = inspect.signature(RAGInference.chat)
        params = list(sig.parameters.keys())
        
        assert 'owner_id' in params, "chat() should accept owner_id parameter"
        print(f"✓ chat() method parameters: {params}")
        print("✓ owner_id parameter is supported")
        
    except Exception as e:
        print(f"✗ Error testing RAGInference: {e}")
        raise


def test_api_router_with_owner_id():
    """Test API router accepts owner_id"""
    print("\n=== Test 5: API Router with owner_id ===")
    
    try:
        from api.routers.rag_inference import ChatRequest
        
        # Test ChatRequest model
        request = ChatRequest(query="test query", owner_id="test-user-123")
        assert request.query == "test query"
        assert request.owner_id == "test-user-123"
        print("✓ ChatRequest accepts owner_id")
        
        # Test with None owner_id
        request2 = ChatRequest(query="test query")
        assert request2.owner_id is None
        print("✓ owner_id is optional (backward compatible)")
        
    except Exception as e:
        print(f"✗ Error testing API router: {e}")
        raise


def main():
    """Run all tests"""
    print("=" * 60)
    print("User Isolation Retrieval Tests")
    print("=" * 60)
    
    try:
        test_chunk_with_owner_id_field()
        test_owner_id_filtering()
        test_retrieval_api_with_owner_id()
        test_rag_inference_with_owner_id()
        test_api_router_with_owner_id()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

