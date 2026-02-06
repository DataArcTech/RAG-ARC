from typing import Any, Optional, List, Dict
import logging

from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ...data_model.orm_models import (
    ChunkMetadata, ChunkIndexStatus,
)

logger = logging.getLogger(__name__)


class _PostgreSQLChunksMixin:
    def store_chunk_metadata_batch(
        self,
        chunk_metadatas: List[ChunkMetadata],
        **kwargs: Any,
    ) -> List[str]:
        """Store chunk metadata in bulk (single transaction).

        Why:
        - The indexing pipeline can generate hundreds/thousands of chunks per file.
        - Committing per-row is extremely slow; a single transaction reduces overhead dramatically.
        """
        if not chunk_metadatas:
            return []

        try:
            with self.SessionMaker() as session:
                # Prefer SQLAlchemy bulk path to reduce ORM overhead.
                session.bulk_save_objects(chunk_metadatas, return_defaults=False)
                session.commit()
                return [m.chunk_id for m in chunk_metadatas]
        except IntegrityError as e:
            if "already exists" in str(e) or "duplicate key" in str(e):
                raise ValueError("One or more chunks already exist (duplicate chunk_id)") from e
            raise
        except SQLAlchemyError as e:
            logger.error("Database error storing chunk metadata batch: %s", e)
            raise

    def store_chunk_metadata(
        self,
        chunk_metadata: ChunkMetadata,
        **kwargs: Any,
    ) -> str:
        """Store chunk metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(chunk_metadata)
                session.commit()
                logger.info(f"Stored chunk metadata: {chunk_metadata.chunk_id}")
                return chunk_metadata.chunk_id

        except IntegrityError as e:
            if "already exists" in str(e) or "duplicate key" in str(e):
                raise ValueError(f"Chunk with ID {chunk_metadata.chunk_id} already exists")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error storing chunk metadata: {e}")
            raise

    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional[ChunkMetadata]:
        """Get chunk metadata by chunk_id using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                chunk_metadata = session.query(ChunkMetadata).filter_by(chunk_id=chunk_id).first()

                if chunk_metadata:
                    return chunk_metadata
                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error getting chunk metadata for {chunk_id}: {e}")
            raise

    def update_chunk_metadata(
        self,
        chunk_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update chunk metadata fields using SQLAlchemy ORM"""
        if not updates:
            return False

        try:
            with self.SessionMaker() as session:
                # Update the record (SQLAlchemy handles enum conversion automatically)
                rows_updated = session.query(ChunkMetadata).filter_by(chunk_id=chunk_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated chunk metadata: {chunk_id}")
                    return True
                else:
                    logger.warning(f"No chunk found with ID: {chunk_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating chunk metadata {chunk_id}: {e}")
            raise

    def delete_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> bool:
        """Delete chunk metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(ChunkMetadata).filter_by(chunk_id=chunk_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted chunk metadata: {chunk_id}")
                    return True
                else:
                    logger.warning(f"No chunk found with ID: {chunk_id}")
                    return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting chunk metadata {chunk_id}: {e}")
            raise

    def list_chunk_metadata(
        self,
        source_parsed_content_id: Optional[str] = None,
        index_status: Optional[ChunkIndexStatus] = None,
        chunker_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ChunkMetadata]:
        """List chunk metadata with optional filtering using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(ChunkMetadata)

                # Add filters
                if source_parsed_content_id:
                    query = query.filter(ChunkMetadata.source_parsed_content_id == source_parsed_content_id)
                if index_status:
                    query = query.filter(ChunkMetadata.index_status == index_status.value)
                if chunker_type:
                    query = query.filter(ChunkMetadata.chunker_type == chunker_type)

                # Add ordering
                query = query.order_by(ChunkMetadata.created_at.desc())

                # Add pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                chunk_metadata_list = query.all()
                logger.debug(f"Retrieved {len(chunk_metadata_list)} chunk metadata records")

                return chunk_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing chunk metadata: {e}")
            raise

    # ==================== USER MANAGEMENT ====================

