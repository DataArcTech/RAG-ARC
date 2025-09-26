import os
from typing import (
    Any,
    Optional,
    List,
    Tuple,
)
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from .base import FileDB
from framework.singleton_decorator import singleton


logger = logging.getLogger(__name__)


@singleton
class LocalDB(FileDB):
    """
    Local filesystem blob storage implementation for high-performance file operations.

    This class provides a complete blob storage solution using the local filesystem,
    supporting hierarchical key organization, collision handling, and extended attributes
    for metadata storage where supported by the filesystem.

    Key features:
    - Hierarchical directory structure based on blob keys
    - Collision handling with overwrite/error/version modes
    - Extended attribute support for content-type metadata
    - Path traversal protection for security
    - Optional empty directory cleanup
    - File URL generation for direct access
    - Comprehensive error handling and logging

    Storage organization:
    - Base directory: Configured via RAG_FILE_STORAGE_PATH environment variable
    - Hierarchical structure: Mirrors blob key structure in directories
    - Version naming: file_v2.ext, file_v3.ext for versioned keys
    - Path safety: Removes '..' and leading '/' from keys

    Collision handling modes:
        - "overwrite": Replace existing file (default)
        - "error": Raise KeyError if file exists
        - "version": Create versioned filename (file_v2.ext)

    Performance considerations:
    - Direct filesystem I/O for maximum speed
    - Parent directory creation as needed
    - Optional cleanup of empty directories
    - Extended attributes may not be supported on all filesystems
    - Large file operations are memory efficient with streaming

    Environment variables:
        RAG_FILE_STORAGE_PATH (str): Root directory for blob storage (default: ./data/files)

    Typical usage:
        >>> config = LocalConfig()  # No base_path needed
        >>> storage = LocalDB(config)
        >>> key, overwritten = storage.store("path/file.txt", data)
        >>> content = storage.retrieve("path/file.txt")
        >>> url = storage.generate_presigned_url("path/file.txt")

    Security considerations:
    - Path traversal protection prevents '../' attacks
    - File permissions inherit from system umask
    - Extended attributes may expose metadata to filesystem users
    - Direct filesystem access bypasses application-level permissions

    Attributes:
        config: Configuration object (base_path no longer used)
    """
    
    def _get_base_path(self) -> Path:
        """Get base storage directory path from environment variable"""
        base_path_str = os.getenv('LOCAL_FILE_STORAGE_PATH', './data/files')
        base_path = Path(base_path_str)
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path
    
    def _get_full_path(self, key: str) -> Path:
        """Convert blob key to full filesystem path"""
        base_path = self._get_base_path()
        # Ensure key doesn't contain path traversal attempts
        safe_key = key.replace('..', '').lstrip('/')
        return base_path / safe_key
    
    def store(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, bool]:
        """Store blob data with given key"""
        try:
            # Since FileStore always generates unique keys, no collision handling needed
            was_overwritten = False
            storage_key = key
            
            key_exists = self.exists(key)
            if key_exists:
                was_overwritten = True
                logger.info(f"Overwriting existing blob with key: '{key}'")
            
            file_path = self._get_full_path(storage_key)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Store content type as extended attribute if supported
            if content_type and hasattr(os, 'setxattr'):
                try:
                    os.setxattr(str(file_path), b'user.content_type', content_type.encode())
                except OSError:
                    # Extended attributes not supported on this filesystem
                    pass
            
            logger.debug(f"Stored blob with key: {storage_key}")
            return storage_key, was_overwritten
            
        except OSError as e:
            logger.error(f"Error storing blob {key}: {e}")
            raise
    
    def retrieve(self, key: str, **kwargs: Any) -> bytes:
        """Retrieve blob data by key"""
        try:
            file_path = self._get_full_path(key)
            
            if not file_path.exists():
                raise KeyError(f"Blob with key '{key}' not found")
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            logger.debug(f"Retrieved blob with key: {key}")
            return data
            
        except OSError as e:
            logger.error(f"Error retrieving blob {key}: {e}")
            raise
    
    def delete(self, key: str, **kwargs: Any) -> bool:
        """Delete blob by key"""
        try:
            file_path = self._get_full_path(key)
            
            if not file_path.exists():
                logger.warning(f"Attempted to delete non-existent blob: {key}")
                return False
            
            file_path.unlink()
            
            # Clean up empty parent directories if configured to do so
            if getattr(self.config, 'cleanup_empty_dirs', False):
                try:
                    parent = file_path.parent
                    base_path = self._get_base_path()
                    while parent != base_path and parent.exists():
                        try:
                            parent.rmdir()  # Only removes if empty
                            parent = parent.parent
                        except OSError:
                            break  # Directory not empty or other error
                except OSError:
                    pass  # Ignore errors during cleanup
            
            logger.debug(f"Deleted blob with key: {key}")
            return True
            
        except OSError as e:
            logger.error(f"Error deleting blob {key}: {e}")
            raise
    
    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if blob exists"""
        try:
            file_path = self._get_full_path(key)
            exists = file_path.exists() and file_path.is_file()
            return exists
            
        except OSError as e:
            logger.error(f"Error checking blob existence {key}: {e}")
            raise
    
    def list_keys(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """List blob keys with optional prefix filter"""
        try:
            base_path = self._get_base_path()
            keys = []
            
            if prefix:
                # Use prefix to limit search scope
                prefix_path = base_path / prefix.lstrip('/')
                if prefix_path.is_dir():
                    search_path = prefix_path
                    search_pattern = "**/*"
                else:
                    # Prefix might be a partial filename
                    search_path = prefix_path.parent
                    search_pattern = f"{prefix_path.name}*/**/*"
            else:
                search_path = base_path
                search_pattern = "**/*"
            
            for file_path in search_path.rglob(search_pattern.split('/')[-1] if '/' in search_pattern else search_pattern):
                if file_path.is_file():
                    # Convert back to key by making path relative to base_path
                    try:
                        relative_path = file_path.relative_to(base_path)
                        key = str(relative_path).replace(os.sep, '/')
                        
                        # Apply prefix filter if specified
                        if prefix and not key.startswith(prefix):
                            continue
                        
                        keys.append(key)
                        
                        if limit and len(keys) >= limit:
                            break
                            
                    except ValueError:
                        # File is not under base_path, skip it
                        continue
            
            # Sort keys for consistent ordering
            keys.sort()
            
            logger.debug(f"Listed {len(keys)} blob keys with prefix: {prefix}")
            return keys
            
        except OSError as e:
            logger.error(f"Error listing blobs with prefix {prefix}: {e}")
            raise
    
    def generate_presigned_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        **kwargs: Any,
    ) -> str:
        """Generate presigned URL for blob access
        
        Note: Local filesystem doesn't support true presigned URLs with expiration.
        This returns a file:// URL for local access.
        """
        try:
            file_path = self._get_full_path(key)

            if not file_path.exists():
                raise KeyError(f"Blob with key '{key}' not found")

            # Convert to absolute path before generating URI
            absolute_path = file_path.resolve()
            file_url = absolute_path.as_uri()

            logger.debug(f"Generated file URL for key: {key}")
            logger.warning(f"Local filesystem doesn't support expiration. URL will not expire after {expiration_seconds} seconds")
            
            return file_url
            
        except OSError as e:
            logger.error(f"Error generating URL for {key}: {e}")
            raise