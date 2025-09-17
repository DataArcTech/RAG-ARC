"""
Test for MinIOBlobStore - testing blob storage operations
"""

import tempfile
import os
import pytest
from io import BytesIO
from typing import Literal
import logging

from framework.config import AbstractConfig
from encapsulation.database.file_db.minio import MinIODB


class MinIOConfig(AbstractConfig):
    """Configuration for MinIO blob store testing"""
    type: Literal["minio_blob_store"] = "minio_blob_store"
    endpoint: str = "localhost:9000"
    username: str = "ROOTNAME"
    password: str = "CHANGEME123"
    bucket_name: str = "test-bucket"
    secure: bool = False
    region: str = "us-east-1"
    
    def build(self) -> MinIODB:
        return MinIODB(self)


def create_test_config() -> MinIOConfig:
    """Create MinIO configuration for testing"""
    return MinIOConfig()


@pytest.fixture
def blob_store():
    """Create MinIO blob store instance for testing"""
    config = create_test_config()
    return config.build()


@pytest.fixture
def sample_data():
    """Sample test data"""
    return {
        "text_data": b"Hello, this is test content for blob storage!",
        "binary_data": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",
        "json_data": b'{"test": "data", "number": 42}',
    }


class TestMinIOBlobStore:
    """Test cases for MinIO blob store"""
    
    def test_store_and_retrieve_text(self, blob_store, sample_data):
        """Test storing and retrieving text data"""
        print("\n--- Testing text storage ---")
        key = "test/text_file.txt"
        data = sample_data["text_data"]
        
        # Store data
        stored_key, was_overwritten = blob_store.store(key, data, content_type="text/plain")
        assert stored_key == key
        assert was_overwritten is False
        print(f"✓ Stored text data with key: {stored_key}, overwritten: {was_overwritten}")
        
        # Check if exists
        exists = blob_store.exists(key)
        assert exists is True
        print(f"✓ Confirmed blob exists: {exists}")
        
        # Retrieve data
        retrieved_data = blob_store.retrieve(key)
        assert retrieved_data == data
        print(f"✓ Retrieved data matches original: {len(retrieved_data)} bytes")
        
        # Clean up
        deleted = blob_store.delete(key)
        assert deleted is True
        print(f"✓ Deleted blob: {deleted}")
    
    def test_store_and_retrieve_stream(self, blob_store, sample_data):
        """Test storing and retrieving data using streams"""
        print("\n--- Testing stream storage ---")
        key = "test/stream_file.bin"
        data = sample_data["binary_data"]
        stream = BytesIO(data)
        
        # Store stream
        stored_key, was_overwritten = blob_store.store_stream(key, stream, content_type="application/octet-stream")
        assert stored_key == key
        assert was_overwritten is False
        print(f"✓ Stored stream data with key: {stored_key}, overwritten: {was_overwritten}")
        
        # Retrieve as stream
        retrieved_stream = blob_store.retrieve_stream(key)
        retrieved_data = retrieved_stream.read()
        retrieved_stream.close()
        
        assert retrieved_data == data
        print(f"✓ Retrieved stream data matches original: {len(retrieved_data)} bytes")
        
        # Clean up
        blob_store.delete(key)
    
    def test_list_blobs(self, blob_store, sample_data):
        """Test listing blobs with prefix filtering"""
        print("\n--- Testing blob listing ---")
        
        # Store multiple blobs with same prefix
        prefix = "test/list/"
        keys = [
            f"{prefix}file1.txt",
            f"{prefix}file2.json", 
            f"{prefix}subdir/file3.bin"
        ]
        
        for i, key in enumerate(keys):
            stored_key, was_overwritten = blob_store.store(key, sample_data["text_data"], content_type="text/plain")
            assert stored_key == key
            assert was_overwritten is False
        
        print(f"✓ Stored {len(keys)} test blobs")
        
        # Wait a moment for eventual consistency
        import time
        time.sleep(1)
        
        # Verify all keys exist before listing
        for key in keys:
            exists = blob_store.exists(key)
            print(f"Key {key} exists: {exists}")
            assert exists, f"Key {key} should exist after storage"
        
        # List blobs with prefix
        listed_keys = blob_store.list_keys(prefix=prefix)
        # Check that all our test keys are in the listed keys
        for key in keys:
            assert key in listed_keys, f"Expected key {key} not found in listed keys"
        print(f"✓ Listed {len(listed_keys)} blobs with prefix '{prefix}', found all {len(keys)} test keys")
        
        # List with limit
        limited_keys = blob_store.list_keys(prefix=prefix, limit=2)
        assert len(limited_keys) == 2
        print(f"✓ Limited listing returned {len(limited_keys)} blobs")
        
        # Clean up
        for key in keys:
            blob_store.delete(key)
    
    def test_presigned_url(self, blob_store, sample_data):
        """Test generating presigned URLs"""
        print("\n--- Testing presigned URLs ---")
        key = "test/presigned_file.txt"
        data = sample_data["text_data"]
        
        # Store data first
        stored_key, was_overwritten = blob_store.store(key, data)
        assert stored_key == key
        assert was_overwritten is False
        
        # Generate presigned URL for GET
        get_url = blob_store.generate_presigned_url(key, expiration_seconds=3600, method="GET")
        assert get_url.startswith("http")
        assert key in get_url
        print(f"✓ Generated GET presigned URL: {get_url[:50]}...")
        
        # Generate presigned URL for PUT
        put_url = blob_store.generate_presigned_url(key, expiration_seconds=1800, method="PUT")
        assert put_url.startswith("http")
        assert key in put_url
        print(f"✓ Generated PUT presigned URL: {put_url[:50]}...")
        
        # Clean up
        blob_store.delete(key)
    
    
    
    

    def test_error_handling(self, blob_store):
        """Test error handling for non-existent blobs"""
        print("\n--- Testing error handling ---")
        non_existent_key = "test/non_existent_file.txt"
        
        # Check non-existent blob
        exists = blob_store.exists(non_existent_key)
        assert exists is False
        print(f"✓ Non-existent blob correctly reported as not existing")
        
        # Try to retrieve non-existent blob
        try:
            blob_store.retrieve(non_existent_key)
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert non_existent_key in str(e)
            print(f"✓ KeyError raised for non-existent blob: {e}")
        
        # Try to delete non-existent blob
        deleted = blob_store.delete(non_existent_key)
        assert deleted is False
        print(f"✓ Delete of non-existent blob returned False")


def main():
    """Main test function"""
    print("Testing MinIO Blob Store...")
    print("Note: This test requires MinIO server running on localhost:9000")
    print("Start MinIO with: docker run -p 9000:9000 -p 9001:9001 --name minio \\")
    print("  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin123 \\")
    print("  quay.io/minio/minio server /data --console-address ':9001'")
    
    try:
        # Create blob store instance
        config = create_test_config()
        blob_store = config.build()
        
        # Sample data
        sample_data = {
            "text_data": b"Hello, this is test content for blob storage!",
            "binary_data": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",
            "json_data": b'{"test": "data", "number": 42}',
        }
        
        # Run tests
        test_instance = TestMinIOBlobStore()
        
        test_instance.test_store_and_retrieve_text(blob_store, sample_data)
        test_instance.test_store_and_retrieve_stream(blob_store, sample_data)
        test_instance.test_list_blobs(blob_store, sample_data)
        test_instance.test_presigned_url(blob_store, sample_data)
        test_instance.test_error_handling(blob_store)
        
        print("\n🎉 All MinIO blob store tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure MinIO server is running and accessible")


if __name__ == "__main__":
    main()