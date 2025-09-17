"""
Simple test to verify LocalBlobStore implementation matches BlobStore interface
"""
import tempfile
import shutil
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass

from encapsulation.database.file_db.local import LocalDB


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    base_path: str
    cleanup_empty_dirs: bool = False




def test_local_blob_store():
    """Test LocalBlobStore implementation"""
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Testing in: {temp_dir}")
    
    try:
        # Create config and instance
        config = MockConfig(base_path=temp_dir)
        store = LocalDB(config)
        
        # Test data
        test_key = "test/file.txt"
        test_data = b"Hello, World!"
        test_content_type = "text/plain"
        
        # Test store
        print("Testing store()...")
        stored_key, was_overwritten = store.store(test_key, test_data, test_content_type)
        assert stored_key == test_key
        assert was_overwritten is False
        print("✓ store() works")
        
        # Test exists
        print("Testing exists()...")
        assert store.exists(test_key)
        assert not store.exists("nonexistent/key")
        print("✓ exists() works")
        
        # Test retrieve
        print("Testing retrieve()...")
        retrieved_data = store.retrieve(test_key)
        assert retrieved_data == test_data
        print("✓ retrieve() works")
        
        # Test retrieve_stream
        print("Testing retrieve_stream()...")
        with store.retrieve_stream(test_key) as stream:
            stream_data = stream.read()
            assert stream_data == test_data
        print("✓ retrieve_stream() works")
        
        # Test store_stream
        print("Testing store_stream()...")
        stream_key = "test/stream_file.txt"
        stream_data = b"Stream data"
        input_stream = BytesIO(stream_data)
        stored_stream_key, was_overwritten = store.store_stream(stream_key, input_stream)
        assert stored_stream_key == stream_key
        assert was_overwritten is False
        assert store.retrieve(stream_key) == stream_data
        print("✓ store_stream() works")
        
        # Test list_keys
        print("Testing list_keys()...")
        keys = store.list_keys()
        assert test_key in keys
        assert stream_key in keys
        
        # Test with prefix
        test_keys = store.list_keys(prefix="test/")
        assert test_key in test_keys
        assert stream_key in test_keys
        print("✓ list_keys() works")
        
        # Test generate_presigned_url
        print("Testing generate_presigned_url()...")
        url = store.generate_presigned_url(test_key)
        assert url.startswith("file://")
        print("✓ generate_presigned_url() works")
        
        # Test delete
        print("Testing delete()...")
        assert store.delete(test_key)
        assert not store.exists(test_key)
        assert not store.delete("nonexistent/key")  # Should return False
        print("✓ delete() works")
        
        # Test error cases
        print("Testing error cases...")
        try:
            store.retrieve("nonexistent/key")
            assert False, "Should have raised KeyError"
        except KeyError:
            print("✓ retrieve() raises KeyError for missing key")
        
        try:
            store.retrieve_stream("nonexistent/key")
            assert False, "Should have raised KeyError"
        except KeyError:
            print("✓ retrieve_stream() raises KeyError for missing key")
        
        try:
            store.generate_presigned_url("nonexistent/key")
            assert False, "Should have raised KeyError"
        except KeyError:
            print("✓ generate_presigned_url() raises KeyError for missing key")
        
        print("\n✅ All tests passed! LocalBlobStore implementation is correct.")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up: {temp_dir}")


if __name__ == "__main__":
    test_local_blob_store()