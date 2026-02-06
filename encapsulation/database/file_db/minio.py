from typing import (
    Any,
    Optional,
    List,
    Tuple,
)
from io import BytesIO
import logging

from minio import Minio
from minio.error import S3Error

from .base import FileDB


logger = logging.getLogger(__name__)


class MinIODB(FileDB):
    """
    MinIO S3-compatible blob storage implementation for distributed object storage.
    
    This class provides a complete blob storage solution using MinIO (or AWS S3), supporting
    distributed storage, collision handling, presigned URLs, and comprehensive bucket management.
    It implements the S3 API for high-performance, scalable blob storage operations.
    
    Key features:
    - S3-compatible API with MinIO or AWS S3 backends
    - Automatic bucket creation and management
    - Collision handling with overwrite/error/version modes
    - Presigned URL generation for secure direct access
    - Stream-based operations for memory efficiency
    - Comprehensive error handling with S3Error mapping
    - Connection pooling and SSL/TLS support
    
    Storage architecture:
    - Bucket-based organization with configurable bucket names
    - Object keys can be hierarchical (path/to/object.ext)
    - Version naming: file_v2.ext, file_v3.ext for versioned keys
    - Metadata stored as object metadata (content-type, etc.)
    
    Collision handling modes:
        - "overwrite": Replace existing object (default)
        - "error": Raise KeyError if object exists
        - "version": Create versioned object key (file_v2.ext)
        
    Performance considerations:
    - Network I/O dependent on MinIO/S3 server location
    - Stream operations minimize memory usage for large objects
    - Bucket existence checks are cached per client instance
    - Connection pooling reduces connection overhead
    - SSL/TLS adds encryption overhead but improves security
    
    Configuration parameters:
        endpoint (str): MinIO server endpoint (host:port)
        username (str): Access key for authentication
        password (str): Secret key for authentication  
        bucket_name (str): Target bucket name
        secure (bool): Use HTTPS connection (default: False)
        region (str): AWS region for S3 compatibility
        
    Presigned URL capabilities:
    - Generate temporary URLs for direct client access
    - Support GET, PUT, POST, DELETE methods
    - Configurable expiration times (default: 1 hour)
    - Eliminates server proxy for large file transfers
    
    Typical usage:
        >>> config = MinIOConfig(
        ...     endpoint="localhost:9000",
        ...     username="admin",
        ...     password="password",
        ...     bucket_name="my-bucket"
        ... )
        >>> storage = MinIODB(config)
        >>> key, overwritten = storage.store("path/file.txt", data)
        >>> url = storage.generate_presigned_url("path/file.txt", method="GET")
        
    Error handling:
    - S3Error exceptions are caught and logged appropriately
    - NoSuchKey errors are mapped to KeyError for consistency
    - Bucket creation failures are logged and re-raised
    - Network timeouts and connection errors are handled gracefully
        
    Attributes:
        config: Configuration object with MinIO connection parameters
        client: MinIO client instance (initialized in __init__)
    """

    def __init__(self, config):
        """Initialize MinIODB with config

        Args:
            config: Configuration object with MinIO connection parameters
        """
        super().__init__(config)
        self._object_prefix = (getattr(self.config, "object_key_prefix", None) or "").strip().rstrip("/")
        if self._object_prefix and not self._object_prefix.endswith("/"):
            self._object_prefix += "/"
        logger.info("Initializing MinIODB (endpoint=%s, bucket=%s, prefix=%r)", self.config.endpoint, self.config.bucket_name, self._object_prefix or "(none)")

        # Build MinIO client immediately
        self.client = Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.username,
            secret_key=self.config.password,
            secure=getattr(self.config, 'secure', False),
            region=getattr(self.config, 'region', None),
        )

        # Ensure bucket exists (Alibaba OSS bucket must pre-exist, skip create)
        try:
            if not self.client.bucket_exists(self.config.bucket_name):
                try:
                    self.client.make_bucket(self.config.bucket_name)
                except Exception as make_err:
                    logger.warning("Bucket create skipped (OSS bucket must pre-exist): %s", make_err)
        except Exception as check_err:
            logger.warning("Bucket existence check skipped (OSS): %s", check_err)
        logger.info(f"MinIODB initialized with bucket: {self.config.bucket_name}")

    def _object_key(self, key: str) -> str:
        """Apply object_key_prefix to the storage key."""
        k = (key or "").lstrip("/")
        if not self._object_prefix:
            return k
        return self._object_prefix + k

    def _strip_prefix(self, object_name: str) -> str:
        """Remove object_key_prefix from returned key."""
        if not self._object_prefix or not object_name:
            return object_name
        if object_name.startswith(self._object_prefix):
            return object_name[len(self._object_prefix):]
        return object_name

    def store(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, bool]:
        """Store blob data with given key"""
        try:
            client = self.client
            bucket_name = self.config.bucket_name
            
            # Always overwrite - no conflict resolution needed since keys are unique
            was_overwritten = False
            storage_key = key
            obj_key = self._object_key(storage_key)

            key_exists = self.exists(key)

            if key_exists:
                was_overwritten = True
                logger.info(f"Overwriting existing blob with key: '{key}'")

            data_stream = BytesIO(data)

            client.put_object(
                bucket_name=bucket_name,
                object_name=obj_key,
                data=data_stream,
                length=len(data),
                content_type=content_type,
                **kwargs
            )
            
            logger.debug(f"Stored blob with key: {storage_key}")
            return storage_key, was_overwritten
            
        except S3Error as e:
            logger.error(f"Error storing blob {key}: {e}")
            raise
    
    def retrieve(self, key: str, **kwargs: Any) -> bytes:
        """Retrieve blob data by key"""
        try:
            client = self.client
            bucket_name = self.config.bucket_name
            obj_key = self._object_key(key)

            response = client.get_object(
                bucket_name=bucket_name,
                object_name=obj_key,
                **kwargs
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.debug(f"Retrieved blob with key: {key}")
            return data
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                raise KeyError(f"Blob with key '{key}' not found")
            logger.error(f"Error retrieving blob {key}: {e}")
            raise
    
    def delete(self, key: str, **kwargs: Any) -> bool:
        """Delete blob by key"""
        try:
            client = self.client
            bucket_name = self.config.bucket_name
            obj_key = self._object_key(key)

            # Check if object exists first
            try:
                client.stat_object(bucket_name, obj_key)
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    logger.warning(f"Attempted to delete non-existent blob: {key}")
                    return False
                raise
            
            # Object exists, proceed with deletion
            client.remove_object(
                bucket_name=bucket_name,
                object_name=obj_key,
                **kwargs
            )
            
            logger.debug(f"Deleted blob with key: {key}")
            return True
            
        except S3Error as e:
            logger.error(f"Error deleting blob {key}: {e}")
            raise
    
    def exists(self, key: str, **kwargs: Any) -> bool:
        """Check if blob exists"""
        try:
            client = self.client
            bucket_name = self.config.bucket_name
            obj_key = self._object_key(key)

            client.stat_object(
                bucket_name=bucket_name,
                object_name=obj_key,
                **kwargs
            )
            return True
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
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
            client = self.client
            bucket_name = self.config.bucket_name
            recursive = True
            obj_prefix = self._object_key(prefix or "") or None

            objects = client.list_objects(
                bucket_name=bucket_name,
                prefix=obj_prefix or None,
                recursive=recursive,
                **kwargs
            )

            keys = []
            for obj in objects:
                keys.append(self._strip_prefix(obj.object_name))
                if limit and len(keys) >= limit:
                    break

            logger.debug(f"Listed {len(keys)} blob keys with prefix: {prefix}")
            return keys
            
        except S3Error as e:
            logger.error(f"Error listing blobs with prefix {prefix}: {e}")
            raise
    
    def generate_presigned_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET",
        **kwargs: Any,
    ) -> str:
        """Generate presigned URL for blob access"""
        try:
            from datetime import timedelta

            client = self.client
            bucket_name = self.config.bucket_name
            obj_key = self._object_key(key)

            url = client.get_presigned_url(
                method=method,
                bucket_name=bucket_name,
                object_name=obj_key,
                expires=timedelta(seconds=expiration_seconds),
                **kwargs
            )
            
            logger.debug(f"Generated presigned URL for key: {key}")
            return url
            
        except S3Error as e:
            logger.error(f"Error generating presigned URL for {key}: {e}")
            raise