"""
Simple test to verify LocalBlobStore implementation matches BlobStore interface
"""

from config.encapsulation.database.file_db.local_config import LocalDBConfig


def test_local_blob_store():
    """Test LocalBlobStore implementation"""
    # Create config and instance using proper framework pattern
    config = LocalDBConfig()
    store = config.build()

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
    assert url.startswith("file://")
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
        assert False, "Should have raised KeyError"
    except KeyError:
        print("generate_presigned_url() raises KeyError for missing key")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_local_blob_store()