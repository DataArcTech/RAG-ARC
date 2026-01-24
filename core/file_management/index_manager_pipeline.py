import asyncio
import json
import logging
from typing import Any, Dict, List

from encapsulation.data_model.schema import Chunk
from framework.thread_pool import get_thread_pool

logger = logging.getLogger(__name__)


class _IndexManagerPipelineMixin:
    async def process_file(self, file_id: str, **kwargs: Any) -> Dict[str, Any]:
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
                "indexers_used": [],
            },
        }

        progress_cb = kwargs.pop("progress", None)

        def _emit(stage: str, percent: int | None = None, payload: Dict[str, Any] | None = None) -> None:
            if progress_cb is None or not callable(progress_cb):
                return
            try:
                progress_cb(str(stage), percent, payload or {})
            except Exception:
                return

        try:
            logger.info(f"Starting indexing pipeline for file_id: {file_id}")
            _emit("start", 1, {"file_id": file_id})

            # Step 1: Get file content from FileStorage (use thread pool to avoid blocking)
            logger.info(f"Step 1: Retrieving file content for {file_id}")
            file_content = await get_thread_pool().run_blocking(
                self.file_storage.get_file_content,
                file_id,
            )
            if file_content is None:
                raise ValueError(f"File content not found for file_id: {file_id}")

            # Get file metadata for filename (use thread pool to avoid blocking)
            file_metadata = await get_thread_pool().run_blocking(
                self.file_storage.get_file_metadata,
                file_id,
            )
            if file_metadata is None:
                raise ValueError(f"File metadata not found for file_id: {file_id}")

            filename = file_metadata.filename
            if not filename:
                logger.warning(f"File metadata has empty filename for file_id: {file_id}")
                filename = f"unknown_file_{file_id}"

            logger.info(f"Retrieved file: {filename} ({len(file_content)} bytes)")
            _emit("retrieved", 5, {"file_id": file_id, "filename": filename, "bytes": len(file_content)})

            # Step 2: Parse the file
            logger.info(f"Step 2: Parsing file {filename}")
            parser_kwargs = dict(kwargs)
            parser_kwargs.setdefault("source_file_id", file_id)
            parse_results = await self.parser.parse_file(
                file_data=file_content,
                filename=filename,
                **parser_kwargs,
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
            _emit("parsed", 25, {"file_id": file_id, "filename": filename, "chars": len(parsed_text)})

            pageindex_context = None
            pageindex_info: Dict[str, Any] = {}
            if getattr(self, "pageindex_service", None) is not None:
                try:
                    from config import pageindex as pageindex_cfg

                    if pageindex_cfg.pageindex_enabled():
                        md_path = None
                        output_dir = None
                        if isinstance(parse_result, dict):
                            md_path = parse_result.get("md_content_path")
                            output_paths = parse_result.get("output_paths")
                            if isinstance(output_paths, dict):
                                md_path = md_path or output_paths.get("markdown")
                            meta = parse_result.get("metadata")
                            if isinstance(meta, dict):
                                output_dir = meta.get("output_dir")

                        pageindex_context = self.pageindex_service.build_context(
                            file_id=file_id,
                            filename=filename,
                            markdown=parsed_text,
                            md_path=str(md_path) if md_path else None,
                            output_dir=str(output_dir) if output_dir else None,
                        )
                        pageindex_info = {
                            "sections": len(pageindex_context.tree.nodes),
                            "level_conflict_ratio": pageindex_context.tree.level_conflict_ratio,
                            "uniform_level_flattened": pageindex_context.tree.uniform_level_flattened,
                        }
                except Exception as exc:
                    logger.warning("PageIndex tree build failed for %s: %s", file_id, exc)
                    pageindex_info = {"error": str(exc)}

            # Step 3: Store parsed content (use thread pool to avoid blocking)
            logger.info(f"Step 3: Storing parsed content")
            parser_type_name = "auto_selected"
            if isinstance(parse_result, dict):
                meta = parse_result.get("metadata")
                if isinstance(meta, dict):
                    token = str(
                        meta.get("parser_label")
                        or meta.get("parser_type")
                        or meta.get("parser_name")
                        or ""
                    ).strip()
                    if token:
                        parser_type_name = token

            result["metadata"]["parser_type"] = parser_type_name

            # Convert parsed text to bytes for storage
            # Use errors='replace' to handle surrogate pairs and invalid Unicode characters
            parsed_data = parsed_text.encode("utf-8", errors="replace")

            parsed_content_id = await get_thread_pool().run_blocking(
                self.parsed_content_storage.store_parsed_content,
                source_file_id=file_id,
                parser_type=parser_type_name,
                parsed_data=parsed_data,
                content_type="text/markdown",
                **kwargs,
            )

            if not parsed_content_id:
                raise ValueError("Failed to store parsed content")

            result["parsed_content_id"] = parsed_content_id
            logger.info(f"Stored parsed content with ID: {parsed_content_id}")
            _emit("parsed_stored", 35, {"file_id": file_id, "parsed_content_id": parsed_content_id})

            # Update file status to PARSED after successfully persisting parsed content.
            await get_thread_pool().run_blocking(
                self._update_file_status_to_parsed,
                file_id,
                **kwargs,
            )

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
                "owner_id": str(file_metadata.owner_id),  # Will be extracted as Chunk.owner_id field
            }

            chunks = self.chunker.chunk_text(
                text=parsed_text,
                metadata=chunk_metadata,
                **kwargs,
            )

            if not chunks:
                raise ValueError("Chunker returned no chunks")

            # Augment index_text early so *all* indexers (dense/BM25/graph) benefit,
            # and so stored chunk JSON remains consistent with indexed text.
            try:
                from core.file_management.index_text_augmentation import augment_chunk_dict_index_text

                for chunk in chunks:
                    if isinstance(chunk, dict):
                        augment_chunk_dict_index_text(chunk)
            except Exception as exc:
                logger.warning("Failed to augment chunk index_text from filename: %s", exc)

            if pageindex_context is not None:
                try:
                    enrichment = self.pageindex_service.enrich_chunks(
                        context=pageindex_context,
                        chunks=chunks,
                    )
                    pageindex_info["chunk_enrichment"] = enrichment
                except Exception as exc:
                    logger.warning("PageIndex chunk enrichment failed for %s: %s", file_id, exc)
                    pageindex_info["chunk_enrichment_error"] = str(exc)

            logger.info(f"Created {len(chunks)} chunks")
            _emit("chunked", 55, {"file_id": file_id, "num_chunks": len(chunks)})
            result["metadata"]["num_chunks"] = len(chunks)

            # Step 5: Store chunks (use thread pool for each chunk to avoid blocking)
            logger.info(f"Step 5: Storing chunks")
            chunk_ids = []
            stored_chunks: List[Dict[str, Any]] = []

            for i, chunk in enumerate(chunks):
                if chunk is None:
                    logger.warning(f"Chunk {i+1}/{len(chunks)} is None, skipping")
                    continue

                # Convert chunk to JSON bytes for storage (use thread pool to avoid blocking)
                try:
                    # Use errors='replace' to handle surrogate pairs and invalid Unicode characters
                    chunk_data = json.dumps(chunk, ensure_ascii=False).encode("utf-8", errors="replace")
                    chunk_id = await get_thread_pool().run_blocking(
                        self.chunk_storage.store_chunk,
                        source_parsed_content_id=parsed_content_id,
                        chunker_type=chunker_strategy,
                        chunk_data=chunk_data,
                        chunk_index=i,  # Pass the chunk index
                        **kwargs,
                    )

                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        stored_chunks.append(chunk)
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
            _emit("chunks_stored", 65, {"file_id": file_id, "num_chunks": len(chunk_ids)})

            # Update parsed content status to CHUNKED (use thread pool to avoid blocking)
            await get_thread_pool().run_blocking(
                self._update_parsed_content_status_to_chunked,
                parsed_content_id,
                **kwargs,
            )

            # Update file status to CHUNKED after successfully persisting chunks.
            await get_thread_pool().run_blocking(
                self._update_file_status_to_chunked,
                file_id,
                **kwargs,
            )

            # Backfill anchor_chunk_id for hierarchical chunking strategies (e.g., semantic_unit).
            try:
                self._backfill_anchor_chunk_ids(stored_chunks, chunk_ids)
                self._persist_backfilled_anchor_chunk_ids(self.chunk_storage, stored_chunks, chunk_ids)
            except Exception as backfill_error:
                logger.warning(f"Failed to backfill anchor_chunk_id metadata: {backfill_error}")

            if pageindex_context is not None:
                try:
                    summary_info = await self.pageindex_service.summarize_sections(
                        context=pageindex_context,
                        chunks=stored_chunks,
                    )
                    pageindex_info["summaries"] = summary_info
                    doc_desc = await self.pageindex_service.build_doc_description(context=pageindex_context)
                    if doc_desc is not None:
                        pageindex_info["doc_description"] = {"enabled": True}
                except Exception as exc:
                    logger.warning("PageIndex summaries failed for %s: %s", file_id, exc)
                    pageindex_info["summary_error"] = str(exc)

                try:
                    pageindex_indexing = await self.pageindex_service.build_indexes(
                        context=pageindex_context,
                        owner_id=str(file_metadata.owner_id) if file_metadata else None,
                        base_indexers=self.indexers,
                    )
                    if pageindex_indexing:
                        pageindex_info["indexing"] = pageindex_indexing
                except Exception as exc:
                    logger.warning("PageIndex indexing failed for %s: %s", file_id, exc)
                    pageindex_info["indexing_error"] = str(exc)

            if pageindex_info:
                result["metadata"]["pageindex"] = pageindex_info

            # Step 6: Index the chunks (if indexers are configured)
            if self.indexers:
                logger.info(f"Step 6: Indexing chunks with {len(self.indexers)} indexers")
                _emit("indexing", 75, {"file_id": file_id, "indexers": len(self.indexers)})
                indexing_results = await self._index_chunks(stored_chunks, chunk_ids)
                result["indexing_results"] = indexing_results
                result["metadata"]["indexers_used"] = list(indexing_results.keys())

                successful_indexers = [name for name, res in indexing_results.items() if res and res.get("success", False)]
                failed_indexers = [name for name, res in indexing_results.items() if not (res and res.get("success", False))]
                result["metadata"]["indexing_summary"] = {
                    "total_indexers": len(self.indexers),
                    "successful_indexers": successful_indexers,
                    "failed_indexers": failed_indexers,
                }

                # Step 7: Update chunk metadata status for successfully indexed chunks (use thread pool to avoid blocking)
                chunks_updated = await get_thread_pool().run_blocking(
                    self._update_indexed_chunks_status,
                    chunk_ids,
                    indexing_results,
                    **kwargs,
                )

                # Step 8: Update file metadata status based on indexer outcomes (use thread pool to avoid blocking)
                if not successful_indexers:
                    from encapsulation.data_model.orm_models import FileStatus

                    result["success"] = False
                    result["error_message"] = "all indexers failed"
                    try:
                        metadata = None
                        try:
                            metadata = self.file_storage.get_file_metadata(file_id)
                        except Exception:
                            metadata = None
                        if metadata is not None and getattr(metadata, "status", None) == FileStatus.DELETED:
                            logger.info("Skip updating file %s to FAILED because status is DELETED", file_id)
                        else:
                            await get_thread_pool().run_blocking(
                                self.file_storage.metadata_store.update_file_status,
                                file_id,
                                FileStatus.FAILED,
                                **kwargs,
                            )
                    except Exception as status_error:
                        logger.error("Failed to update file status to FAILED for %s: %s", file_id, status_error)
                    _emit(
                        "index_failed",
                        100,
                        {"file_id": file_id, "success": False, "indexers": result["metadata"]["indexing_summary"]},
                    )
                    return result

                if len(successful_indexers) == len(self.indexers):
                    await get_thread_pool().run_blocking(
                        self._update_file_status_to_indexed,
                        file_id,
                        **kwargs,
                    )
                    _emit(
                        "indexed",
                        95,
                        {"file_id": file_id, "success": True, "status": "INDEXED", "chunks_updated": chunks_updated},
                    )
                else:
                    await get_thread_pool().run_blocking(
                        self._update_file_status_to_partially_indexed,
                        file_id,
                        **kwargs,
                    )
                    _emit(
                        "indexed_partial",
                        95,
                        {
                            "file_id": file_id,
                            "success": True,
                            "status": "PARTIAL_INDEXED",
                            "chunks_updated": chunks_updated,
                            "indexers": result["metadata"]["indexing_summary"],
                        },
                    )
            else:
                logger.info("Step 6: No indexers configured, skipping indexing")

            # Success!
            result["success"] = True
            logger.info(f"Successfully completed indexing pipeline for file_id: {file_id}")
            _emit("done", 100, {"file_id": file_id, "success": True})

        except Exception as e:
            error_msg = f"Indexing pipeline failed for file_id {file_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            _emit("failed", 100, {"file_id": file_id, "success": False, "error_message": str(e)})

            # Update file status to FAILED (use thread pool to avoid blocking)
            try:
                from encapsulation.data_model.orm_models import FileStatus

                # Deletion can race with indexing; never override DELETED with FAILED.
                metadata = None
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                except Exception:
                    metadata = None
                if metadata is not None and getattr(metadata, "status", None) == FileStatus.DELETED:
                    logger.info("Skip updating file %s to FAILED because status is DELETED", file_id)
                else:
                    await get_thread_pool().run_blocking(
                        self.file_storage.metadata_store.update_file_status,
                        file_id,
                        FileStatus.FAILED,
                        **kwargs,
                    )
                    logger.info(f"Updated file {file_id} status to FAILED due to indexing error")
            except Exception as status_error:
                logger.error(f"Failed to update file status to FAILED for {file_id}: {status_error}")

        return result

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
        text_keys = ["content", "text", "markdown", "md_content", "extracted_text"]

        for key in text_keys:
            if key in parse_result and parse_result[key]:
                content = parse_result[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, bytes):
                    try:
                        return content.decode("utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to decode bytes content from key '{key}': {e}")

        # If we have output_paths, try to read from markdown file
        if "output_paths" in parse_result:
            output_paths = parse_result["output_paths"]
            if isinstance(output_paths, dict) and "markdown" in output_paths:
                markdown_path = output_paths["markdown"]
                if markdown_path:
                    try:
                        with open(markdown_path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read markdown file {markdown_path}: {e}")

        # If we have md_content_path, try to read from it
        if "md_content_path" in parse_result:
            md_path = parse_result["md_content_path"]
            if md_path:
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read markdown file {md_path}: {e}")

        # Fallback: convert the entire result to string
        logger.warning("Could not find text content in standard keys, using string representation")
        return str(parse_result)

    @staticmethod
    def _backfill_anchor_chunk_ids(chunks: List[Dict[str, Any]], chunk_ids: List[str]) -> None:
        """
        Backfill per-slice `anchor_chunk_id` once ChunkStorage has allocated real Chunk IDs.

        Chunkers that produce parent/child relationships should emit:
        - metadata.semantic_unit_id
        - metadata.chunk_role in {"anchor", "slice"}
        - slices may set anchor_chunk_id=None as a placeholder

        This method mutates `chunks` in-place (metadata only) to ensure indexers persist
        `anchor_chunk_id` in the searchable metadata.
        """
        if not chunks or not chunk_ids:
            return
        if len(chunks) != len(chunk_ids):
            return

        unit_to_anchor_id: Dict[str, str] = {}
        for chunk, chunk_id in zip(chunks, chunk_ids):
            meta = chunk.get("metadata") or {}
            if meta.get("chunk_role") != "anchor":
                continue
            semantic_unit_id = str(meta.get("semantic_unit_id") or "").strip()
            if not semantic_unit_id:
                continue
            unit_to_anchor_id[semantic_unit_id] = chunk_id

        if not unit_to_anchor_id:
            return

        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            if meta.get("chunk_role") != "slice":
                continue
            semantic_unit_id = str(meta.get("semantic_unit_id") or "").strip()
            if not semantic_unit_id:
                continue
            anchor_id = unit_to_anchor_id.get(semantic_unit_id)
            if anchor_id:
                meta["anchor_chunk_id"] = anchor_id
                chunk["metadata"] = meta

    @staticmethod
    def _persist_backfilled_anchor_chunk_ids(chunk_storage: Any, chunks: List[Dict[str, Any]], chunk_ids: List[str]) -> int:
        """
        Persist backfilled `anchor_chunk_id` into blob storage for slice chunks.

        Indexers operate on the in-memory dicts, but later retrieval-by-id flows
        can depend on the stored chunk JSON blobs. Without rewriting slice blobs
        after backfill, their metadata remains stale (anchor_chunk_id=null).
        """
        if not chunk_storage or not hasattr(chunk_storage, "overwrite_chunk_json"):
            return 0
        if not chunks or not chunk_ids or len(chunks) != len(chunk_ids):
            return 0

        updated = 0
        for chunk, chunk_id in zip(chunks, chunk_ids):
            meta = chunk.get("metadata") or {}
            if meta.get("chunk_role") != "slice":
                continue
            anchor_id = str(meta.get("anchor_chunk_id") or "").strip()
            if not anchor_id:
                continue
            if chunk_storage.overwrite_chunk_json(chunk_id, chunk):
                updated += 1
        return updated

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
                if "content" not in chunk:
                    logger.warning(f"Chunk at index {i} missing 'content' field, skipping")
                    continue

                # Merge source_metadata into the main metadata
                merged_metadata = chunk.get("metadata", {}).copy() if chunk.get("metadata") else {}
                source_metadata = chunk.get("source_metadata", {})
                if source_metadata:
                    merged_metadata.update(source_metadata)

                # Extract owner_id from metadata or source_metadata
                owner_id = merged_metadata.get("owner_id", "")
                if not owner_id and source_metadata:
                    owner_id = source_metadata.get("owner_id", "")

                chunk_obj = Chunk(
                    id=chunk_id,
                    content=chunk["content"],
                    owner_id=owner_id,
                    metadata=merged_metadata,
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
                    "total_chunks": len(chunk_objects),
                }
            else:
                indexing_results[indexer_name] = result
                if result.get("success"):
                    logger.info(f"Successfully indexed {result.get('indexed_count', 0)} chunks with {indexer_name}")

        return indexing_results

    async def _index_with_single_indexer(self, indexer, indexer_name: str, chunk_objects: List[Chunk]) -> Dict[str, Any]:
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

            indexed_ids = indexed_ids or []
            indexed_count = len(indexed_ids)
            if indexed_count <= 0:
                return {
                    "success": False,
                    "error_message": "indexer returned no indexed ids",
                    "indexed_count": 0,
                    "total_chunks": len(chunk_objects),
                    "indexed_ids": [],
                }

            return {
                "success": True,
                "indexed_count": indexed_count,
                "total_chunks": len(chunk_objects),
                "indexed_ids": indexed_ids,
            }
        except Exception as e:
            logger.error(f"Indexing failed with {indexer_name}: {str(e)}")
            return {
                "success": False,
                "error_message": str(e),
                "indexed_count": 0,
                "total_chunks": len(chunk_objects),
            }
