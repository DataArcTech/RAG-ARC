from typing import Any, Optional, List, Dict
import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ...data_model.orm_models import (
    ParsedContentMetadata, ParsedContentStatus,
)

logger = logging.getLogger(__name__)


class _PostgreSQLParsedContentMixin:
    def store_parsed_content_metadata(
        self,
        parsed_metadata: ParsedContentMetadata,
        **kwargs: Any,
    ) -> str:
        """Store parsed content metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                session.add(parsed_metadata)
                session.commit()
                logger.debug(f"Stored parsed content metadata: {parsed_metadata.parsed_content_id}")
                return parsed_metadata.parsed_content_id

        except IntegrityError:
            logger.error(f"Parsed content metadata with ID '{parsed_metadata.parsed_content_id}' already exists")
            raise ValueError(f"Parsed content metadata with ID '{parsed_metadata.parsed_content_id}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Database error storing parsed content metadata: {e}")
            raise
    
    def get_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> Optional[ParsedContentMetadata]:
        """Retrieve parsed content metadata by ID using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                parsed_metadata = session.query(ParsedContentMetadata).filter_by(parsed_content_id=parsed_content_id).first()

                if parsed_metadata:
                    logger.debug(f"Retrieved parsed content metadata: {parsed_content_id}")
                    return parsed_metadata

                return None

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving parsed content metadata: {e}")
            raise
    
    def update_parsed_content_metadata(
        self,
        parsed_content_id: str,
        updates: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update parsed content metadata using SQLAlchemy ORM"""
        if not updates:
            return True

        try:
            with self.SessionMaker() as session:
                # Add updated_at timestamp
                updates['updated_at'] = datetime.now(tz=datetime.now().astimezone().tzinfo)

                # Update the record (SQLAlchemy handles enum conversion automatically)
                rows_updated = session.query(ParsedContentMetadata).filter_by(parsed_content_id=parsed_content_id).update(updates)
                session.commit()

                if rows_updated > 0:
                    logger.debug(f"Updated parsed content metadata: {parsed_content_id}")
                    return True

                logger.warning(f"No parsed content metadata found to update: {parsed_content_id}")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error updating parsed content metadata: {e}")
            raise
    
    def delete_parsed_content_metadata(self, parsed_content_id: str, **kwargs: Any) -> bool:
        """Delete parsed content metadata using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                rows_deleted = session.query(ParsedContentMetadata).filter_by(parsed_content_id=parsed_content_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.debug(f"Deleted parsed content metadata: {parsed_content_id}")
                    return True

                logger.warning(f"No parsed content metadata found to delete: {parsed_content_id}")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting parsed content metadata: {e}")
            raise
    
    def update_parsed_content_status(
        self,
        parsed_content_id: str,
        new_status: ParsedContentStatus,
        **kwargs: Any,
    ) -> bool:
        """Update parsed content processing status"""
        return self.update_parsed_content_metadata(
            parsed_content_id,
            {'status': new_status},
            **kwargs
        )
    
    def list_parsed_content_metadata(
        self,
        source_file_id: Optional[str] = None,
        status: Optional[ParsedContentStatus] = None,
        parser_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ParsedContentMetadata]:
        """List parsed content metadata with optional filtering using SQLAlchemy ORM"""
        try:
            with self.SessionMaker() as session:
                query = session.query(ParsedContentMetadata)

                # Add filters
                if source_file_id:
                    query = query.filter(ParsedContentMetadata.source_file_id == source_file_id)
                if status:
                    query = query.filter(ParsedContentMetadata.status == status.value)
                if parser_type:
                    query = query.filter(ParsedContentMetadata.parser_type == parser_type)

                # Add ordering
                query = query.order_by(ParsedContentMetadata.created_at.desc())

                # Add pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                parsed_metadata_list = query.all()
                logger.debug(f"Retrieved {len(parsed_metadata_list)} parsed content metadata records")

                return parsed_metadata_list

        except SQLAlchemyError as e:
            logger.error(f"Database error listing parsed content metadata: {e}")
            raise

    # ==================== CHUNK METADATA METHODS ====================


