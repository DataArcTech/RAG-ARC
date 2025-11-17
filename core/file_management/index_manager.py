import logging
import json
import asyncio
from typing import List, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from encapsulation.data_model.orm_models import ChunkIndexStatus

from framework.module import AbstractModule
from encapsulation.data_model.schema import Chunk


if TYPE_CHECKING:
    from config.core.file_management.index_manager_config import IndexManagerConfig

logger = logging.getLogger(__name__)


class IndexManager(AbstractModule):
    """
    This class orchestrates the complete indexing pipeline:
    1. Retrieves file content using file_id from FileStorage
    2. Parses the file using StandardParser
    3. Chunks the parsed content using configured chunker
    4. Indexes the chunks using configured indexers
    5. Stores parsed content and chunks back to storage modules
    """

    def __init__(self, config: "IndexManagerConfig"):
        super().__init__(config)

        # Build storage instances
        self.file_storage = config.file_storage_config.build()
        self.parsed_content_storage = config.parsed_content_storage_config.build()
        self.chunk_storage = config.chunk_storage_config.build()

        # Build parser
        self.parser = self.config.parser_config.build()
        logger.info(f"Initialized parser: {type(self.parser).__name__}")

        # Build chunker
        self.chunker = self.config.chunker_config.build()
        logger.info(f"Initialized chunker: {self.chunker.get_chunker_info()['strategy']}")

        # Build indexers
        self.indexers = []
        for indexer_config in self.config.indexer_configs:
            indexer = indexer_config.build()
            self.indexers.append(indexer)
            logger.info(f"Initialized indexer: {type(indexer).__name__}")

        logger.info(f"IndexManager initialized with {len(self.indexers)} indexers")


    async def index_file(self, file_id: str) -> Dict[str, Any]:
        """
        Async method for indexing a file by file_id.
        This is the main entry point for external usage.

        Args:
            file_id: The ID of the file to index

        Returns:
            Dict containing indexing results
        """
        # Validate file_id
        if file_id is None or not isinstance(file_id, str) or not file_id.strip():
            error_msg = "file_id must be a non-empty string"
            logger.error(error_msg)
            return {
                "success": False,
                "file_id": file_id,
                "error_message": error_msg
            }


        return await self.process_file(file_id)

    async def process_file(
        self,
        file_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process a file through the complete indexing pipeline.

        Args:
            file_id: ID of the file to process
            file_storage: FileStorage instance to retrieve file content
            parsed_content_storage: ParsedContentStorage instance to store parsed content
            chunk_storage: ChunkStorage instance to store chunks
            **kwargs: Additional arguments passed to parser, chunker, and indexers

        Returns:
            Dictionary containing processing results:
            - success: bool - Whether the entire pipeline succeeded
            - file_id: str - Input file ID
            - parsed_content_id: str - ID of stored parsed content (if successful)
            - chunk_ids: List[str] - IDs of stored chunks (if successful)
            - indexing_results: Dict - Results from each indexer
            - error_message: str - Error message if failed
            - metadata: Dict - Processing metadata
        """
        result = {
            "success": False,
            "file_id": file_id,
            "parsed_content_id": None,
            "chunk_ids": [],
            "indexing_results": {},
            "error_message": None,
            "metadata": {
                "parser_type": None,
                "chunker_type": None,
                "num_chunks": 0,
                "indexers_used": []
            }
        }

        try:
            logger.info(f"Starting indexing pipeline for file_id: {file_id}")

            # Step 1: Get file content from FileStorage
            logger.info(f"Step 1: Retrieving file content for {file_id}")
            file_content = self.file_storage.get_file_content(file_id)
            if file_content is None:
                raise ValueError(f"File content not found for file_id: {file_id}")

            # Get file metadata for filename
            file_metadata = self.file_storage.get_file_metadata(file_id)
            if file_metadata is None:
                raise ValueError(f"File metadata not found for file_id: {file_id}")

            filename = file_metadata.filename
            if not filename:
                logger.warning(f"File metadata has empty filename for file_id: {file_id}")
                filename = f"unknown_file_{file_id}"

            logger.info(f"Retrieved file: {filename} ({len(file_content)} bytes)")

            # Step 2: Parse the file
            logger.info(f"Step 2: Parsing file {filename}")
            parse_results = await self.parser.parse_file(
                file_data=file_content,
                filename=filename,
                **kwargs
            )

            if not parse_results:
                raise ValueError(f"Parser returned no results for file: {filename}")

            if not isinstance(parse_results, list):
                parse_results = [parse_results]

            logger.info(f"Parser returned {len(parse_results)} results")

            # Process all parse results and concatenate them
            if len(parse_results) == 1:
                # Single result - process as before
                parse_result = parse_results[0]
                parsed_text = self._extract_text_from_parse_result(parse_result)
            else:
                # Multiple results - concatenate them in order
                logger.info(f"Processing {len(parse_results)} parse results and concatenating them")
                concatenated_texts = []

                for i, parse_result in enumerate(parse_results):
                    if parse_result is None:
                        logger.warning(f"Parse result {i+1} is None, skipping")
                        continue

                    text_content = self._extract_text_from_parse_result(parse_result)
                    if text_content:
                        logger.info(f"Extracted {len(text_content)} characters from result {i+1}")
                        concatenated_texts.append(text_content)
                    else:
                        logger.warning(f"No text content found in parse result {i+1}")

                if not concatenated_texts:
                    raise ValueError("No valid text content extracted from any parse results")

                # Join all texts with double newlines to separate different sections
                parsed_text = "\n\n".join(concatenated_texts)
                logger.info(f"Concatenated {len(concatenated_texts)} results into {len(parsed_text)} characters")

                # Use the first parse result for metadata purposes
                parse_result = parse_results[0]

            if not parsed_text:
                raise ValueError("No text content extracted from parsed result")

            logger.info(f"Extracted {len(parsed_text)} characters of text content")

            # Step 3: Store parsed content
            logger.info(f"Step 3: Storing parsed content")
            parser_type = getattr(self.parser, 'parser', None)
            if parser_type is not None:
                parser_type_name = type(parser_type).__name__
            else:
                parser_type_name = "auto_selected"

            result["metadata"]["parser_type"] = parser_type_name

            # Convert parsed text to bytes for storage
            parsed_data = parsed_text.encode('utf-8')

            parsed_content_id = self.parsed_content_storage.store_parsed_content(
                source_file_id=file_id,
                parser_type=parser_type_name,
                parsed_data=parsed_data,
                content_type="text/markdown",
                **kwargs
            )

            if not parsed_content_id:
                raise ValueError("Failed to store parsed content")

            result["parsed_content_id"] = parsed_content_id
            logger.info(f"Stored parsed content with ID: {parsed_content_id}")

            # Step 4: Chunk the parsed text
            logger.info(f"Step 4: Chunking parsed text")
            chunker_info = self.chunker.get_chunker_info()
            chunker_strategy = chunker_info.get("strategy", type(self.chunker).__name__)
            result["metadata"]["chunker_type"] = chunker_strategy

            # Prepare metadata for chunking
            # Note: owner_id is added to metadata so it can be extracted when creating Chunk objects
            chunk_metadata = {
                "source_file_id": file_id,
                "parsed_content_id": parsed_content_id,
                "filename": filename,
                "parser_type": parser_type_name,
                "owner_id": str(file_metadata.owner_id)  # Will be extracted as Chunk.owner_id field
            }

            chunks = self.chunker.chunk_text(
                text=parsed_text,
                metadata=chunk_metadata,
                **kwargs
            )

            if not chunks:
                raise ValueError("Chunker returned no chunks")

            logger.info(f"Created {len(chunks)} chunks")
            result["metadata"]["num_chunks"] = len(chunks)

            # Step 5: Store chunks
            logger.info(f"Step 5: Storing chunks")
            chunk_ids = []

            for i, chunk in enumerate(chunks):
                if chunk is None:
                    logger.warning(f"Chunk {i+1}/{len(chunks)} is None, skipping")
                    continue

                # Convert chunk to JSON bytes for storage
                try:
                    chunk_data = json.dumps(chunk, ensure_ascii=False).encode('utf-8')
                    chunk_id = self.chunk_storage.store_chunk(
                        source_parsed_content_id=parsed_content_id,
                        chunker_type=chunker_strategy,
                        chunk_data=chunk_data,
                        chunk_index=i,  # Pass the chunk index
                        **kwargs
                    )

                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        logger.debug(f"Stored chunk {i+1}/{len(chunks)} with ID: {chunk_id}")
                    else:
                        logger.warning(f"Failed to store chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Failed to store chunk {i+1}/{len(chunks)}: {str(e)}")
                    continue

            if not chunk_ids:
                raise ValueError("Failed to store any chunks")

            result["chunk_ids"] = chunk_ids
            logger.info(f"Stored {len(chunk_ids)}/{len(chunks)} chunks successfully")

            # Update parsed content status to CHUNKED
            self._update_parsed_content_status_to_chunked(parsed_content_id, **kwargs)

            # Step 6: Index the chunks (if indexers are configured)
            if self.indexers:
                logger.info(f"Step 6: Indexing chunks with {len(self.indexers)} indexers")
                indexing_results = await self._index_chunks(chunks, chunk_ids)
                result["indexing_results"] = indexing_results
                result["metadata"]["indexers_used"] = list(indexing_results.keys())

                # Step 7: Update chunk metadata status for successfully indexed chunks
                chunks_updated = self._update_indexed_chunks_status(chunk_ids, indexing_results, **kwargs)

                # Step 8: Update file metadata status if indexing succeeded
                if chunks_updated:
                    self._update_file_status_to_indexed(file_id, **kwargs)
            else:
                logger.info("Step 6: No indexers configured, skipping indexing")

            # Success!
            result["success"] = True
            logger.info(f"Successfully completed indexing pipeline for file_id: {file_id}")

        except Exception as e:
            error_msg = f"Indexing pipeline failed for file_id {file_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg

            # Update file status to FAILED
            try:
                from encapsulation.data_model.orm_models import FileStatus
                self.file_storage.metadata_store.update_file_status(
                    file_id,
                    FileStatus.FAILED,
                    **kwargs
                )
                logger.info(f"Updated file {file_id} status to FAILED due to indexing error")
            except Exception as status_error:
                logger.error(f"Failed to update file status to FAILED for {file_id}: {status_error}")

        return result

    def _delete_chunks_from_indexers(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """
        Delete chunks from all configured indexers using ThreadPoolExecutor.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Dictionary with deletion results for each indexer
        """
        deletion_results = {}

        # Use ThreadPoolExecutor for concurrent deletion
        with ThreadPoolExecutor(max_workers=len(self.indexers)) as executor:
            future_to_indexer = {}

            for i, indexer in enumerate(self.indexers):
                indexer_name = f"{type(indexer).__name__}_{i}"
                future = executor.submit(self._delete_with_single_indexer_sync, indexer, indexer_name, chunk_ids)
                future_to_indexer[future] = (indexer, indexer_name)

            # Collect results
            for future in as_completed(future_to_indexer):
                indexer, indexer_name = future_to_indexer[future]

                try:
                    result = future.result()
                    deletion_results[indexer_name] = result
                    if result.get("success"):
                        logger.info(f"Successfully deleted {len(chunk_ids)} chunks from {indexer_name}")
                except Exception as e:
                    logger.error(f"Deletion failed with {indexer_name}: {str(e)}")
                    deletion_results[indexer_name] = {
                        "success": False,
                        "error_message": str(e),
                        "deleted_count": 0,
                        "total_chunks": len(chunk_ids)
                    }

        return deletion_results

    def _delete_with_single_indexer_sync(
        self,
        indexer,
        indexer_name: str,
        chunk_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Delete chunks with a single indexer (synchronous).

        This is a helper method that allows concurrent execution of multiple indexers.

        Args:
            indexer: The indexer instance
            indexer_name: Name of the indexer for logging
            chunk_ids: List of chunk IDs to delete

        Returns:
            Dictionary with deletion result for this indexer
        """
        try:
            logger.info(f"Deleting {len(chunk_ids)} chunks from {indexer_name}")
            success = indexer.delete_chunks(chunk_ids)

            if success:
                logger.info(f"Successfully deleted {len(chunk_ids)} chunks from {indexer_name}")
            else:
                logger.warning(f"Deletion returned False for {indexer_name}")

            return {
                "success": bool(success),
                "deleted_count": len(chunk_ids) if success else 0,
                "total_chunks": len(chunk_ids)
            }
        except Exception as e:
            logger.error(f"Deletion failed with {indexer_name}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error_message": str(e),
                "deleted_count": 0,
                "total_chunks": len(chunk_ids)
            }

    def _extract_text_from_parse_result(self, parse_result: Dict[str, Any]) -> str:
        """
        Extract text content from parser result.

        Args:
            parse_result: Dictionary containing parser output

        Returns:
            Extracted text content as string
        """
        if not parse_result:
            return ""

        # Try to find text content in various possible keys
        text_keys = ['content', 'text', 'markdown', 'md_content', 'extracted_text']

        for key in text_keys:
            if key in parse_result and parse_result[key]:
                content = parse_result[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, bytes):
                    try:
                        return content.decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Failed to decode bytes content from key '{key}': {e}")

        # If we have output_paths, try to read from markdown file
        if 'output_paths' in parse_result:
            output_paths = parse_result['output_paths']
            if isinstance(output_paths, dict) and 'markdown' in output_paths:
                markdown_path = output_paths['markdown']
                if markdown_path:
                    try:
                        with open(markdown_path, 'r', encoding='utf-8') as f:
                            return f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read markdown file {markdown_path}: {e}")

        # If we have md_content_path, try to read from it
        if 'md_content_path' in parse_result:
            md_path = parse_result['md_content_path']
            if md_path:
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read markdown file {md_path}: {e}")

        # Fallback: convert the entire result to string
        logger.warning("Could not find text content in standard keys, using string representation")
        return str(parse_result)

    async def _index_chunks(self, chunks: List[Dict[str, Any]], chunk_ids: List[str]) -> Dict[str, Any]:
        """
        Index chunks using configured indexers concurrently.

        Args:
            chunks: List of chunk dictionaries
            chunk_ids: List of corresponding chunk IDs

        Returns:
            Dictionary with indexing results for each indexer
        """
        indexing_results = {}

        # Convert chunks to Chunk objects for indexing
        chunk_objects = []
        for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            if not chunk or not chunk_id:
                logger.warning(f"Chunk or chunk_id at index {i} is invalid, skipping")
                continue

            try:
                if 'content' not in chunk:
                    logger.warning(f"Chunk at index {i} missing 'content' field, skipping")
                    continue

                # Merge source_metadata into the main metadata
                merged_metadata = chunk.get('metadata', {}).copy() if chunk.get('metadata') else {}
                source_metadata = chunk.get('source_metadata', {})
                if source_metadata:
                    merged_metadata.update(source_metadata)

                # Extract owner_id from metadata or source_metadata
                owner_id = merged_metadata.get('owner_id', '')
                if not owner_id and source_metadata:
                    owner_id = source_metadata.get('owner_id', '')

                chunk_obj = Chunk(
                    id=chunk_id,
                    content=chunk['content'],
                    owner_id=owner_id,
                    metadata=merged_metadata
                )
                chunk_objects.append(chunk_obj)
            except Exception as e:
                logger.error(f"Failed to create Chunk for chunk {i}: {e}")
                continue

        if not chunk_objects:
            logger.warning("No valid chunks created for indexing")
            return indexing_results

        # Run all indexers concurrently
        tasks = []
        for i, indexer in enumerate(self.indexers):
            indexer_name = f"{type(indexer).__name__}_{i}"
            tasks.append(self._index_with_single_indexer(indexer, indexer_name, chunk_objects))
        
        # Execute all indexers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            indexer_name = f"{type(self.indexers[i]).__name__}_{i}"

            if isinstance(result, Exception):
                logger.error(f"Indexing failed with {indexer_name}: {str(result)}")
                indexing_results[indexer_name] = {
                    "success": False,
                    "error_message": str(result),
                    "indexed_count": 0,
                    "total_chunks": len(chunk_objects)
                }
            else:
                indexing_results[indexer_name] = result
                if result.get("success"):
                    logger.info(f"Successfully indexed {result.get('indexed_count', 0)} chunks with {indexer_name}")

        return indexing_results

    async def _index_with_single_indexer(
        self,
        indexer,
        indexer_name: str,
        chunk_objects: List[Chunk]
    ) -> Dict[str, Any]:
        """
        Index chunks with a single indexer.

        This is a helper method that allows concurrent execution of multiple indexers.

        Args:
            indexer: The indexer instance
            indexer_name: Name of the indexer for logging
            chunk_objects: List of Chunk objects to index

        Returns:
            Dictionary with indexing result for this indexer
        """
        try:
            logger.info(f"Indexing {len(chunk_objects)} chunks with {indexer_name}")
            indexed_ids = await indexer.update_index(chunk_objects)

            return {
                "success": True,
                "indexed_count": len(indexed_ids) if indexed_ids else 0,
                "total_chunks": len(chunk_objects),
                "indexed_ids": indexed_ids or []
            }
        except Exception as e:
            logger.error(f"Indexing failed with {indexer_name}: {str(e)}")
            return {
                "success": False,
                "error_message": str(e),
                "indexed_count": 0,
                "total_chunks": len(chunk_objects)
            }

    def _update_indexed_chunks_status(
        self,
        chunk_ids: List[str],
        indexing_results: Dict[str, Any],
        **kwargs: Any
    ) -> bool:
        """
        Update chunk metadata status to INDEXED for successfully indexed chunks.

        Args:
            chunk_ids: List of chunk IDs that were indexed
            indexing_results: Results from indexers
            **kwargs: Additional arguments

        Returns:
            bool: True if any chunks were successfully updated, False otherwise
        """
        # Check if any indexer succeeded
        any_success = any(
            result.get("success", False)
            for result in indexing_results.values()
        )

        if not any_success:
            logger.warning("No indexer succeeded, skipping chunk status update")
            return False

        # Collect successfully indexed chunk IDs from all indexers
        successfully_indexed_ids = set()
        for indexer_name, result in indexing_results.items():
            if result.get("success", False):
                indexed_ids = result.get("indexed_ids", [])
                if indexed_ids:
                    successfully_indexed_ids.update(indexed_ids)

        if not successfully_indexed_ids:
            logger.warning("No chunks were successfully indexed")
            return False

        # Update status for each successfully indexed chunk
        now = datetime.now(tz=datetime.now().astimezone().tzinfo)
        updated_count = 0
        failed_count = 0

        for chunk_id in successfully_indexed_ids:
            try:
                success = self.chunk_storage.metadata_store.update_chunk_metadata(
                    chunk_id,
                    {
                        "index_status": ChunkIndexStatus.INDEXED,
                        "indexed_at": now
                    },
                    **kwargs
                )
                if success:
                    updated_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to update status for chunk {chunk_id}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error updating status for chunk {chunk_id}: {e}")

        logger.info(f"Updated chunk status: {updated_count} succeeded, {failed_count} failed")
        return updated_count > 0

    def _update_file_status_to_indexed(
        self,
        file_id: str,
        **kwargs: Any
    ) -> None:
        """
        Update file metadata status to INDEXED after successful indexing.

        Args:
            file_id: File ID to update
            **kwargs: Additional arguments
        """
        from encapsulation.data_model.orm_models import FileStatus

        try:
            success = self.file_storage.metadata_store.update_file_metadata(
                file_id,
                {"status": FileStatus.INDEXED},
                **kwargs
            )
            if success:
                logger.info(f"Updated file {file_id} status to INDEXED")
            else:
                logger.warning(f"Failed to update file {file_id} status to INDEXED")
        except Exception as e:
            logger.error(f"Error updating file {file_id} status: {e}")

    def _update_parsed_content_status_to_chunked(
        self,
        parsed_content_id: str,
        **kwargs: Any
    ) -> None:
        """
        Update parsed content metadata status to CHUNKED after successful chunking.

        Args:
            parsed_content_id: Parsed content ID to update
            **kwargs: Additional arguments
        """
        from encapsulation.data_model.orm_models import ParsedContentStatus

        try:
            success = self.parsed_content_storage.metadata_store.update_parsed_content_metadata(
                parsed_content_id,
                {"status": ParsedContentStatus.CHUNKED},
                **kwargs
            )
            if success:
                logger.info(f"Updated parsed content {parsed_content_id} status to CHUNKED")
            else:
                logger.warning(f"Failed to update parsed content {parsed_content_id} status to CHUNKED")
        except Exception as e:
            logger.error(f"Error updating parsed content {parsed_content_id} status: {e}")

    def delete_file(self, file_id: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Synchronous method for deleting a file and all its associated data.
        This is the main entry point for external usage.

        Args:
            file_id: The ID of the file to delete
            **kwargs: Additional arguments

        Returns:
            Dict containing deletion results
        """
        # Validate file_id
        if file_id is None or not isinstance(file_id, str) or not file_id.strip():
            error_msg = "file_id must be a non-empty string"
            logger.error(error_msg)
            return {
                "success": False,
                "file_id": file_id,
                "error_message": error_msg
            }

        return self.delete_file_data(file_id, **kwargs)

    def delete_file_data(
        self,
        file_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Delete a file and all its associated data (parsed content, chunks and index entries).

        This method:
        1. Finds all parsed content associated with the file
        2. Finds all chunks associated with each parsed content
        3. Deletes chunks from all configured indexers (FIRST - not covered by CASCADE)
        4. Deletes chunk blobs (SECOND - not covered by CASCADE)
        5. Deletes parsed content blobs (THIRD - not covered by CASCADE)

        Note:
        - Chunk metadata and parsed content metadata are automatically deleted via CASCADE
          when parsed_content_metadata is deleted (which happens when file_metadata is deleted)
        - However, indexer data and blob files must be manually deleted as they are not
          managed by the database

        Args:
            file_id: ID of the file to delete
            **kwargs: Additional arguments

        Returns:
            Dictionary containing deletion results:
            - success: bool - Whether the entire deletion succeeded
            - file_id: str - Input file ID
            - deleted_parsed_content_ids: List[str] - IDs of deleted parsed content
            - deleted_chunk_ids: List[str] - IDs of deleted chunks
            - indexer_deletion_results: Dict - Results from each indexer
            - error_message: str - Error message if failed
        """
        result = {
            "success": False,
            "file_id": file_id,
            "deleted_parsed_content_ids": [],
            "deleted_chunk_ids": [],
            "indexer_deletion_results": {},
            "error_message": None
        }

        try:
            logger.info(f"Starting deletion pipeline for file_id: {file_id}")

            # Step 1: Find all parsed content for this file
            logger.info(f"Step 1: Finding parsed content for file {file_id}")
            parsed_content_list = self.parsed_content_storage.metadata_store.list_parsed_content_metadata(
                source_file_id=file_id,
                **kwargs
            )

            if not parsed_content_list:
                logger.warning(f"No parsed content found for file_id: {file_id}")
                result["success"] = True
                return result

            logger.info(f"Found {len(parsed_content_list)} parsed content entries")

            # Step 2: For each parsed content, find all chunks
            all_chunk_ids = []
            for parsed_content in parsed_content_list:
                if not parsed_content or not hasattr(parsed_content, 'parsed_content_id'):
                    logger.warning("Invalid parsed_content, skipping")
                    continue

                parsed_content_id = parsed_content.parsed_content_id
                if not parsed_content_id:
                    logger.warning("parsed_content has empty parsed_content_id, skipping")
                    continue

                # Find all chunks for this parsed content
                logger.info(f"Step 2: Finding chunks for parsed_content_id: {parsed_content_id}")
                chunk_list = self.chunk_storage.metadata_store.list_chunk_metadata(
                    source_parsed_content_id=parsed_content_id,
                    **kwargs
                )

                if chunk_list:
                    chunk_ids = [chunk.chunk_id for chunk in chunk_list if chunk and hasattr(chunk, 'chunk_id') and chunk.chunk_id]
                    all_chunk_ids.extend(chunk_ids)
                    logger.info(f"Found {len(chunk_ids)} chunks for parsed_content {parsed_content_id}")

            # Step 3: Delete chunks from all indexers FIRST
            # REQUIRED: Indexer data (FAISS, BM25) is NOT managed by database CASCADE
            if all_chunk_ids and self.indexers:
                logger.info(f"Step 3: Deleting {len(all_chunk_ids)} chunks from {len(self.indexers)} indexers")
                deletion_results = self._delete_chunks_from_indexers(all_chunk_ids)
                result["indexer_deletion_results"] = deletion_results

                # Check if indexer deletion was successful
                all_indexers_successful = all(
                    res.get("success", False) for res in deletion_results.values()
                )
                if not all_indexers_successful:
                    error_msg = "Failed to delete chunks from some indexers"
                    logger.warning(error_msg)
                    # Log which indexers failed
                    failed_indexers = [
                        name for name, res in deletion_results.items()
                        if not res.get("success", False)
                    ]
                    logger.warning(f"Failed indexers: {failed_indexers}")
                    # Store error but continue with metadata deletion
                    result["error_message"] = error_msg
                else:
                    logger.info("Successfully deleted chunks from all indexers")
            else:
                logger.info("Step 3: No chunks to delete from indexers")

            # Step 4: Delete chunk blobs (SECOND, after indexer deletion succeeds)
            # REQUIRED: Blob files are NOT managed by database CASCADE
            # NOTE: Chunk metadata will be automatically deleted via CASCADE when parsed_content_metadata is deleted
            if all_chunk_ids:
                logger.info(f"Step 4: Deleting {len(all_chunk_ids)} chunk blobs")
                for chunk_id in all_chunk_ids:
                    if not chunk_id:
                        continue

                    try:
                        # Delete chunk blob (REQUIRED - not covered by CASCADE)
                        chunk_metadata = self.chunk_storage.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
                        if chunk_metadata and hasattr(chunk_metadata, 'blob_key') and chunk_metadata.blob_key:
                            self.chunk_storage.blob_store.delete(chunk_metadata.blob_key, **kwargs)
                            logger.debug(f"Deleted chunk blob: {chunk_metadata.blob_key}")

                        # Delete chunk metadata (will also be deleted by CASCADE, but doing it explicitly for clarity)
                        self.chunk_storage.metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                        logger.debug(f"Deleted chunk metadata: {chunk_id}")
                        result["deleted_chunk_ids"].append(chunk_id)
                    except Exception as e:
                        logger.error(f"Failed to delete chunk {chunk_id}: {e}")

                logger.info(f"Successfully deleted {len(result['deleted_chunk_ids'])}/{len(all_chunk_ids)} chunk blobs")

            # Step 5: Delete parsed content blobs (THIRD, after chunks are deleted)
            # REQUIRED: Blob files are NOT managed by database CASCADE
            # NOTE: Parsed content metadata will be automatically deleted via CASCADE when file_metadata is deleted
            logger.info(f"Step 5: Deleting {len(parsed_content_list)} parsed content blobs")
            for parsed_content in parsed_content_list:
                if not parsed_content or not hasattr(parsed_content, 'parsed_content_id'):
                    continue

                parsed_content_id = parsed_content.parsed_content_id
                if not parsed_content_id:
                    continue

                try:
                    # Delete parsed content blob (REQUIRED - not covered by CASCADE)
                    if hasattr(parsed_content, 'blob_key') and parsed_content.blob_key:
                        self.parsed_content_storage.blob_store.delete(parsed_content.blob_key, **kwargs)
                        logger.debug(f"Deleted parsed content blob: {parsed_content.blob_key}")

                    # Delete parsed content metadata (will also be deleted by CASCADE, but doing it explicitly for clarity)
                    self.parsed_content_storage.metadata_store.delete_parsed_content_metadata(parsed_content_id, **kwargs)
                    logger.debug(f"Deleted parsed content metadata: {parsed_content_id}")
                    result["deleted_parsed_content_ids"].append(parsed_content_id)
                except Exception as e:
                    logger.error(f"Failed to delete parsed content {parsed_content_id}: {e}")

            logger.info(f"Successfully deleted {len(result['deleted_parsed_content_ids'])}/{len(parsed_content_list)} parsed content blobs")

            # Success!
            result["success"] = True
            logger.info(f"Successfully completed deletion pipeline for file_id: {file_id}")

        except Exception as e:
            error_msg = f"Deletion pipeline failed for file_id {file_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg

        return result
