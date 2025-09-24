"""
Simple test to verify MinIODB implementation matches BlobStore interface
"""

from config.encapsulation.database.file_db.minio_config import MinIOConfig


def test_minio_blob_store():
    """Test MinIODB implementation"""
    # Create config and instance using proper framework pattern
    config = MinIOConfig()

    try:
        store = config.build()
        print("MinIODB initialized successfully")
    except Exception as e:
        print(f"Failed to connect to MinIO server at {config.endpoint}: {e}")
        print("Make sure MinIO is running with the configured credentials:")
        print(f"  Endpoint: {config.endpoint}")
        print(f"  Username: {config.username}")
        print(f"  Password: {config.password}")
        print("  You can start MinIO with:")
        print("  docker run -p 9000:9000 -p 9001:9001 --name minio \\")
        print("    -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \\")
        print("    quay.io/minio/minio server /data --console-address ':9001'")
        return

    # Test data
    test_key = "test/file.txt"
    test_data = b"Hello, World!"
    test_content_type = "text/plain"

    # Test store
    print("Testing store()...")
    stored_key, was_overwritten = store.store(test_key, test_data, test_content_type)
    assert stored_key == test_key
    assert was_overwritten is False
    print("store() works")

    # Test exists
    print("Testing exists()...")
    assert store.exists(test_key)
    assert not store.exists("nonexistent/key")
    print("exists() works")

    # Test retrieve
    print("Testing retrieve()...")
    retrieved_data = store.retrieve(test_key)
    assert retrieved_data == test_data
    print("retrieve() works")

    # Test list_keys
    print("Testing list_keys()...")
    keys = store.list_keys()
    assert test_key in keys

    # Test with prefix
    test_keys = store.list_keys(prefix="test/")
    assert test_key in test_keys
    print("list_keys() works")

    # Test generate_presigned_url
    print("Testing generate_presigned_url()...")
    url = store.generate_presigned_url(test_key)
    assert url.startswith("http")  # MinIO returns HTTP/HTTPS URLs
    print("generate_presigned_url() works")

    # Test delete
    print("Testing delete()...")
    assert store.delete(test_key)
    assert not store.exists(test_key)
    assert not store.delete("nonexistent/key")  # Should return False
    print("delete() works")

    # Test error cases
    print("Testing error cases...")
    try:
        store.retrieve("nonexistent/key")
        assert False, "Should have raised KeyError"
    except KeyError:
        print("retrieve() raises KeyError for missing key")

    try:
        store.generate_presigned_url("nonexistent/key")
        assert False, "Should have raised an exception"
    except Exception:
        print("generate_presigned_url() raises exception for missing key")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_minio_blob_store()