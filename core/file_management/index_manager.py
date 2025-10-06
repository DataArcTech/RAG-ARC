import logging
import json
import asyncio
from typing import List, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def __init__(self, config: "IndexManagerConfig", file_storage=None, parsed_content_storage=None, chunk_storage=None):
        super().__init__(config)

        # Optional storage instances for async index_file method
        self.file_storage = file_storage
        self.parsed_content_storage = parsed_content_storage
        self.chunk_storage = chunk_storage

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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_file,
            file_id,
            self.file_storage,
            self.parsed_content_storage,
            self.chunk_storage
        )

    def process_file(
        self,
        file_id: str,
        file_storage,
        parsed_content_storage,
        chunk_storage,
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
            file_content = file_storage.get_file_content(file_id)
            if file_content is None:
                raise ValueError(f"File content not found for file_id: {file_id}")

            # Get file metadata for filename
            file_metadata = file_storage.get_file_metadata(file_id)
            if file_metadata is None:
                raise ValueError(f"File metadata not found for file_id: {file_id}")

            filename = file_metadata.filename
            logger.info(f"Retrieved file: {filename} ({len(file_content)} bytes)")

            # Step 2: Parse the file
            logger.info(f"Step 2: Parsing file {filename}")
            parse_results = self.parser.parse_file(
                file_data=file_content,
                filename=filename,
                **kwargs
            )

            if not parse_results:
                raise ValueError(f"Parser returned no results for file: {filename}")

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
                    text_content = self._extract_text_from_parse_result(parse_result)
                    if text_content:
                        logger.info(f"Extracted {len(text_content)} characters from result {i+1}")
                        concatenated_texts.append(text_content)
                    else:
                        logger.warning(f"No text content found in parse result {i+1}")

                # Join all texts with double newlines to separate different sections
                parsed_text = "\n\n".join(concatenated_texts)
                logger.info(f"Concatenated {len(concatenated_texts)} results into {len(parsed_text)} characters")

                # Use the first parse result for metadata purposes
                parse_result = parse_results[0]

            if not parsed_text:
                raise ValueError(f"No text content extracted from parsed result")

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

            parsed_content_id = parsed_content_storage.store_parsed_content(
                source_file_id=file_id,
                parser_type=parser_type_name,
                parsed_data=parsed_data,
                content_type="text/markdown",
                **kwargs
            )

            result["parsed_content_id"] = parsed_content_id
            logger.info(f"Stored parsed content with ID: {parsed_content_id}")

            # Step 4: Chunk the parsed text
            logger.info(f"Step 4: Chunking parsed text")
            chunker_info = self.chunker.get_chunker_info()
            result["metadata"]["chunker_type"] = chunker_info["strategy"]

            # Prepare metadata for chunking
            chunk_metadata = {
                "source_file_id": file_id,
                "parsed_content_id": parsed_content_id,
                "filename": filename,
                "parser_type": parser_type_name
            }

            chunks = self.chunker.chunk_text(
                text=parsed_text,
                metadata=chunk_metadata,
                **kwargs
            )

            if not chunks:
                raise ValueError(f"Chunker returned no chunks for parsed text")

            logger.info(f"Created {len(chunks)} chunks")
            result["metadata"]["num_chunks"] = len(chunks)

            # Step 5: Store chunks
            logger.info(f"Step 5: Storing chunks")
            chunk_ids = []

            for i, chunk in enumerate(chunks):
                # Convert chunk to JSON bytes for storage
                chunk_data = json.dumps(chunk, ensure_ascii=False).encode('utf-8')

                chunk_id = chunk_storage.store_chunk(
                    source_parsed_content_id=parsed_content_id,
                    chunker_type=chunker_info["strategy"],
                    chunk_data=chunk_data,
                    **kwargs
                )

                chunk_ids.append(chunk_id)
                logger.debug(f"Stored chunk {i+1}/{len(chunks)} with ID: {chunk_id}")

            result["chunk_ids"] = chunk_ids
            logger.info(f"Stored all {len(chunk_ids)} chunks")

            # Step 6: Index the chunks (if indexers are configured)
            if self.indexers:
                logger.info(f"Step 6: Indexing chunks with {len(self.indexers)} indexers")
                indexing_results = self._index_chunks(chunks, chunk_ids)
                result["indexing_results"] = indexing_results
                result["metadata"]["indexers_used"] = list(indexing_results.keys())
            else:
                logger.info("Step 6: No indexers configured, skipping indexing")

            # Success!
            result["success"] = True
            logger.info(f"Successfully completed indexing pipeline for file_id: {file_id}")

        except Exception as e:
            error_msg = f"Indexing pipeline failed for file_id {file_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg

        return result

    def _extract_text_from_parse_result(self, parse_result: Dict[str, Any]) -> str:
        """
        Extract text content from parser result.

        Args:
            parse_result: Dictionary containing parser output

        Returns:
            Extracted text content as string
        """
        # The exact format depends on the parser used
        # This is a generic implementation that handles common formats

        # Try to find text content in various possible keys
        text_keys = ['content', 'text', 'markdown', 'md_content', 'extracted_text']

        for key in text_keys:
            if key in parse_result and parse_result[key]:
                content = parse_result[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, bytes):
                    return content.decode('utf-8')

        # If we have output_paths, try to read from markdown file
        if 'output_paths' in parse_result and 'markdown' in parse_result['output_paths']:
            markdown_path = parse_result['output_paths']['markdown']
            try:
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read markdown file {markdown_path}: {e}")

        # If we have md_content_path, try to read from it
        if 'md_content_path' in parse_result:
            md_path = parse_result['md_content_path']
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read markdown file {md_path}: {e}")

        # Fallback: convert the entire result to string
        logger.warning("Could not find text content in standard keys, using string representation")
        return str(parse_result)

    def _index_chunks(self, chunks: List[Dict[str, Any]], chunk_ids: List[str]) -> Dict[str, Any]:
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
            # Create Chunk object
            # The exact format depends on the Chunk schema
            try:
                # Merge source_metadata into the main metadata
                merged_metadata = chunk.get('metadata', {}).copy()
                source_metadata = chunk.get('source_metadata', {})
                if source_metadata:
                    merged_metadata.update(source_metadata)

                chunk_obj = Chunk(
                    id=chunk_id,
                    content=chunk['content'],
                    metadata=merged_metadata
                )
                chunk_objects.append(chunk_obj)
            except Exception as e:
                logger.error(f"Failed to create Chunk for chunk {i}: {e}")
                continue

        if not chunk_objects:
            logger.error("No valid chunks created for indexing")
            return indexing_results

        # Run all indexers concurrently in a single event loop
        async def run_all_indexers():
            """Run all indexers concurrently"""
            tasks = []
            for i, indexer in enumerate(self.indexers):
                indexer_name = f"{type(indexer).__name__}_{i}"
                tasks.append(self._index_with_single_indexer(indexer, indexer_name, chunk_objects))

            return await asyncio.gather(*tasks, return_exceptions=True)

        # Execute all indexers concurrently
        results = asyncio.run(run_all_indexers())

        # Process results
        for i, result in enumerate(results):
            indexer_name = f"{type(self.indexers[i]).__name__}_{i}"

            if isinstance(result, Exception):
                error_msg = f"Indexing failed with {indexer_name}: {str(result)}"
                logger.error(error_msg, exc_info=True)
                indexing_results[indexer_name] = {
                    "success": False,
                    "error_message": error_msg,
                    "indexed_count": 0,
                    "total_chunks": len(chunk_objects)
                }
            else:
                indexing_results[indexer_name] = result
                if result["success"]:
                    logger.info(f"Successfully indexed {result['indexed_count']} chunks with {indexer_name}")

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

            # Call the indexer's async update_index method
            indexed_ids = await indexer.update_index(chunk_objects)

            return {
                "success": True,
                "indexed_count": len(indexed_ids) if indexed_ids else 0,
                "total_chunks": len(chunk_objects),
                "indexed_ids": indexed_ids or []
            }

        except Exception as e:
            error_msg = f"Indexing failed with {indexer_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error_message": error_msg,
                "indexed_count": 0,
                "total_chunks": len(chunk_objects)
            }

    def process_multiple_files(
        self,
        file_ids: List[str],
        file_storage,
        max_workers: int = 3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process multiple files through the indexing pipeline in parallel.

        Args:
            file_ids: List of file IDs to process
            file_storage: FileStorage instance
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments passed to process_file

        Returns:
            Dictionary containing batch processing results
        """
        logger.info(f"Starting batch processing for {len(file_ids)} files")

        results = {
            "total_files": len(file_ids),
            "successful_files": 0,
            "failed_files": 0,
            "results": {},
            "summary": {
                "total_chunks_created": 0,
                "total_indexers_used": set(),
                "processing_errors": []
            }
        }

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file_id = {
                executor.submit(self.process_file, file_id, file_storage, **kwargs): file_id
                for file_id in file_ids
            }

            # Collect results
            for future in as_completed(future_to_file_id):
                file_id = future_to_file_id[future]

                try:
                    result = future.result()
                    results["results"][file_id] = result

                    if result["success"]:
                        results["successful_files"] += 1
                        results["summary"]["total_chunks_created"] += result["metadata"]["num_chunks"]
                        results["summary"]["total_indexers_used"].update(result["metadata"]["indexers_used"])
                    else:
                        results["failed_files"] += 1
                        results["summary"]["processing_errors"].append({
                            "file_id": file_id,
                            "error": result["error_message"]
                        })

                except Exception as e:
                    error_msg = f"Unexpected error processing file {file_id}: {str(e)}"
                    logger.error(error_msg, exc_info=True)

                    results["results"][file_id] = {
                        "success": False,
                        "file_id": file_id,
                        "error_message": error_msg
                    }
                    results["failed_files"] += 1
                    results["summary"]["processing_errors"].append({
                        "file_id": file_id,
                        "error": error_msg
                    })

        # Convert set to list for JSON serialization
        results["summary"]["total_indexers_used"] = list(results["summary"]["total_indexers_used"])

        success_rate = (results["successful_files"] / results["total_files"]) * 100 if results["total_files"] > 0 else 0

        logger.info(f"Batch processing completed: {results['successful_files']}/{results['total_files']} successful ({success_rate:.1f}%)")

        return results
