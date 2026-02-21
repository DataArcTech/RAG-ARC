import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from encapsulation.data_model.schema import Chunk
from framework.thread_pool import get_thread_pool
from core.file_management.parser_artifact_paths import (
    extract_md_path_and_output_dir,
    relativize_under_base,
    resolve_under_base,
)
from core.file_management.pageindex.strict_page_chunking import (
    attach_page_chunking_diagnostics,
    build_page_markdown_from_content_list,
    load_page_markdown,
    strict_page_chunk,
)

logger = logging.getLogger(__name__)


class _IndexManagerPipelineMixin:
    @staticmethod
    def _resolve_io_manager() -> Any | None:
        """Best-effort access to IOManager.

        NOTE: do not resolve `io://...` paths in core logic. IOManager owns backend mapping
        (LocalDB now; MinIO later).
        """

        try:
            from framework.register import Register

            return Register().get_object("io_manager")
        except Exception:
            return None

    def _chunk_with_optional_strict_page(
        self,
        *,
        text: str,
        chunk_metadata: Dict[str, Any],
        pageindex_context: Any,
        md_path: str | None,
        output_dir: str | None,
        pageindex_info: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        from config import pageindex as pageindex_cfg

        mode = pageindex_cfg.strict_page_chunking_mode()
        if mode == "off":
            return self.chunker.chunk_text(text=text, metadata=chunk_metadata, **kwargs)

        page_texts: Dict[int, str] = {}
        diagnostics: Dict[str, Any] = {
            "mode": mode,
            "source": None,
        }

        if pageindex_context is not None:
            tree = getattr(pageindex_context, "tree", None)
            content_list = getattr(tree, "content_list", None) if tree is not None else None
            if isinstance(content_list, list) and content_list:
                page_texts = build_page_markdown_from_content_list(content_list)
                if page_texts:
                    diagnostics["source"] = "pageindex_content_list"

        if not page_texts:
            loaded_page_texts, load_diag = load_page_markdown(
                md_path=md_path,
                output_dir=output_dir,
                io_manager=self._resolve_io_manager(),
            )
            diagnostics.update(load_diag)
            if loaded_page_texts:
                page_texts = loaded_page_texts
                diagnostics["source"] = diagnostics.get("source") or "content_list"

        if not page_texts:
            diagnostics["reason"] = diagnostics.get("reason") or "page_text_missing"
            if mode == "require":
                pageindex_info.update(attach_page_chunking_diagnostics(pageindex_info, diagnostics))
                raise ValueError("Strict page chunking requires MinerU content_list page text, but none was found")
            logger.warning(
                "Strict page chunking requested (mode=%s) but no page text was found for file_id=%s; falling back to default chunking",
                mode,
                chunk_metadata.get("source_file_id"),
            )
            pageindex_info.update(attach_page_chunking_diagnostics(pageindex_info, diagnostics))
            return self.chunker.chunk_text(text=text, metadata=chunk_metadata, **kwargs)

        chunks = strict_page_chunk(
            chunker=self.chunker,
            chunk_metadata=chunk_metadata,
            page_texts=page_texts,
            **kwargs,
        )
        diagnostics["pages_used"] = len(page_texts)
        diagnostics["strict_chunk_count"] = len(chunks)
        diagnostics["reason"] = "ok" if chunks else "strict_chunker_empty"

        if not chunks:
            if mode == "require":
                pageindex_info.update(attach_page_chunking_diagnostics(pageindex_info, diagnostics))
                raise ValueError("Strict page chunking produced no chunks")
            logger.warning(
                "Strict page chunking produced no chunks for file_id=%s; falling back to default chunking",
                chunk_metadata.get("source_file_id"),
            )
            pageindex_info.update(attach_page_chunking_diagnostics(pageindex_info, diagnostics))
            return self.chunker.chunk_text(text=text, metadata=chunk_metadata, **kwargs)

        pageindex_info.update(attach_page_chunking_diagnostics(pageindex_info, diagnostics))
        return chunks

    async def process_file_from_parsed_markdown(
        self,
        *,
        file_id: str,
        parsed_markdown: str,
        filename: str | None = None,
        parser_type: str | None = None,
        md_path: str | None = None,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Index a file starting from an already-parsed markdown string.

        This is meant for performance experiments and ingestion flows that
        reuse a precomputed parser output (e.g., MinerU markdown) and want to
        measure/optimize the "parsed -> askable" latency.

        Notes:
        - File metadata must already exist (we still need owner_id + filename).
        - This intentionally skips Step 1 (read file bytes) and Step 2 (parse).
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
                "timings_s": {},
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

        if file_id is None or not isinstance(file_id, str) or not file_id.strip():
            result["error_message"] = "file_id must be a non-empty string"
            return result
        if parsed_markdown is None or not isinstance(parsed_markdown, str) or not parsed_markdown.strip():
            result["error_message"] = "parsed_markdown must be a non-empty string"
            return result

        started = time.perf_counter()
        stage_t0 = started
        try:
            logger.info("Starting parsed-markdown indexing pipeline for file_id=%s", file_id)
            _emit("start", 1, {"file_id": file_id, "mode": "parsed_markdown"})

            # Load file metadata for owner_id + filename.
            file_metadata = await get_thread_pool().run_blocking(
                self.file_storage.get_file_metadata,
                file_id,
            )
            if file_metadata is None:
                raise ValueError(f"File metadata not found for file_id: {file_id}")

            effective_filename = (filename or getattr(file_metadata, "filename", None) or "").strip()
            if not effective_filename:
                effective_filename = f"unknown_file_{file_id}"

            result["metadata"]["timings_s"]["load_file_metadata"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            from config.core.file_management.chunk_ingest_defaults import PARSED_MARKDOWN_PARSER_TYPE_DEFAULT

            parser_type_name = (parser_type or PARSED_MARKDOWN_PARSER_TYPE_DEFAULT).strip() or PARSED_MARKDOWN_PARSER_TYPE_DEFAULT
            result["metadata"]["parser_type"] = parser_type_name

            base_output_dir = getattr(getattr(self, "parser", None), "base_output_dir", None)
            md_path_rel = relativize_under_base(md_path, base_dir=str(base_output_dir) if base_output_dir else None)
            output_dir_rel = relativize_under_base(output_dir, base_dir=str(base_output_dir) if base_output_dir else None)
            md_path_abs = resolve_under_base(md_path_rel, base_dir=str(base_output_dir) if base_output_dir else None)
            output_dir_abs = resolve_under_base(output_dir_rel, base_dir=str(base_output_dir) if base_output_dir else None)

            parsed_text = parsed_markdown
            _emit("parsed_ready", 15, {"file_id": file_id, "filename": effective_filename, "chars": len(parsed_text)})

            # Optional: build PageIndex context + enrich chunks (keeps DeepSearch/tree tools available).
            pageindex_context = None
            pageindex_info: Dict[str, Any] = {}
            if getattr(self, "pageindex_service", None) is not None:
                try:
                    from config import pageindex as pageindex_cfg

                    if pageindex_cfg.pageindex_enabled():
                        io_manager = self._resolve_io_manager()
                        pageindex_context = self.pageindex_service.build_context(
                            file_id=file_id,
                            filename=effective_filename,
                            markdown=parsed_text,
                            md_path=md_path_abs,
                            output_dir=output_dir_abs,
                            io_manager=io_manager,
                        )
                        pageindex_info = {
                            "sections": len(pageindex_context.tree.nodes),
                            "level_conflict_ratio": pageindex_context.tree.level_conflict_ratio,
                            "uniform_level_flattened": pageindex_context.tree.uniform_level_flattened,
                        }
                except Exception as exc:
                    logger.warning("PageIndex tree build failed for %s: %s", file_id, exc)
                    pageindex_info = {"error": str(exc)}

            result["metadata"]["timings_s"]["pageindex_tree"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Persist parsed markdown (same as Step 3 in process_file).
            parsed_data = parsed_text.encode("utf-8", errors="replace")
            parsed_content_id = await get_thread_pool().run_blocking(
                self.parsed_content_storage.store_parsed_content,
                source_file_id=file_id,
                parser_type=parser_type_name,
                parsed_data=parsed_data,
                content_type="text/markdown",
                md_path=md_path_rel,
                output_dir=output_dir_rel,
                **kwargs,
            )
            if not parsed_content_id:
                raise ValueError("Failed to store parsed content")
            result["parsed_content_id"] = parsed_content_id

            _emit("parsed_stored", 30, {"file_id": file_id, "parsed_content_id": parsed_content_id})

            # Update file status to PARSED after successfully persisting parsed content.
            await get_thread_pool().run_blocking(
                self._update_file_status_to_parsed,
                file_id,
                **kwargs,
            )

            result["metadata"]["timings_s"]["store_parsed_content"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Chunk (Step 4).
            chunker_info = self.chunker.get_chunker_info()
            chunker_strategy = chunker_info.get("strategy", type(self.chunker).__name__)
            result["metadata"]["chunker_type"] = chunker_strategy

            chunk_metadata = {
                "source_file_id": file_id,
                "parsed_content_id": parsed_content_id,
                "filename": effective_filename,
                "parser_type": parser_type_name,
                "owner_id": str(getattr(file_metadata, "owner_id", "")),
            }

            chunks = self._chunk_with_optional_strict_page(
                text=parsed_text,
                chunk_metadata=chunk_metadata,
                pageindex_context=pageindex_context,
                md_path=md_path_abs,
                output_dir=output_dir_abs,
                pageindex_info=pageindex_info,
                **kwargs,
            )
            if not chunks:
                raise ValueError("Chunker returned no chunks")

            try:
                from core.file_management.index_text_augmentation import augment_chunk_dict_index_text

                for chunk in chunks:
                    if isinstance(chunk, dict):
                        augment_chunk_dict_index_text(chunk)
            except Exception as exc:
                logger.warning("Failed to augment chunk index_text from filename: %s", exc)

            if pageindex_context is not None:
                try:
                    enrichment = self.pageindex_service.enrich_chunks(context=pageindex_context, chunks=chunks)
                    pageindex_info["chunk_enrichment"] = enrichment
                except Exception as exc:
                    logger.warning("PageIndex chunk enrichment failed for %s: %s", file_id, exc)
                    pageindex_info["chunk_enrichment_error"] = str(exc)
            if pageindex_info:
                result["metadata"]["pageindex"] = pageindex_info

            result["metadata"]["num_chunks"] = len(chunks)
            _emit("chunked", 50, {"file_id": file_id, "num_chunks": len(chunks)})

            result["metadata"]["timings_s"]["chunk"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Store chunks (Step 5) - prefer batch store.
            chunk_ids: list[str] = []
            stored_chunks: List[Dict[str, Any]] = []
            store_batch = getattr(self.chunk_storage, "store_chunks_batch", None)
            if callable(store_batch):
                from config.core.file_management.chunk_ingest_defaults import (
                    CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT,
                )

                # `process_file_from_parsed_content_id` may be called with `owner_id=...` in kwargs.
                # `store_chunks_batch` already receives `owner_id` explicitly from file_metadata, so
                # we must avoid passing duplicate kwargs which would raise a TypeError.
                store_kwargs = dict(kwargs)
                store_kwargs.pop("owner_id", None)

                stored_refs: list[tuple[int, str]] = await get_thread_pool().run_blocking(
                    store_batch,
                    source_parsed_content_id=parsed_content_id,
                    chunker_type=chunker_strategy,
                    chunks=chunks,
                    owner_id=getattr(file_metadata, "owner_id", None),
                    validate_after_store=CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT,
                    **store_kwargs,
                )
                for chunk_index, chunk_id in stored_refs:
                    if not chunk_id:
                        continue
                    chunk_ids.append(chunk_id)
                    try:
                        stored_chunks.append(chunks[int(chunk_index)])
                    except Exception:
                        continue
            else:
                store_kwargs = dict(kwargs)
                store_kwargs.pop("owner_id", None)
                for i, chunk in enumerate(chunks):
                    if chunk is None:
                        continue
                    chunk_data = json.dumps(chunk, ensure_ascii=False).encode("utf-8", errors="replace")
                    chunk_id = await get_thread_pool().run_blocking(
                        self.chunk_storage.store_chunk,
                        source_parsed_content_id=parsed_content_id,
                        chunker_type=chunker_strategy,
                        chunk_data=chunk_data,
                        chunk_index=i,
                        owner_id=getattr(file_metadata, "owner_id", None),
                        validate_after_store=True,
                        **store_kwargs,
                    )
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        stored_chunks.append(chunk)

            if not chunk_ids:
                raise ValueError("No chunks stored successfully")

            # Backfill semantic_unit anchor_chunk_id for graph/dense retrieval.
            try:
                self._backfill_anchor_chunk_ids(stored_chunks, chunk_ids)
                overwrite = getattr(self.chunk_storage, "overwrite_chunk_json", None)
                if callable(overwrite):
                    await get_thread_pool().run_blocking(
                        self._persist_backfilled_anchor_chunk_ids,
                        self.chunk_storage,
                        stored_chunks,
                        chunk_ids,
                    )
            except Exception as exc:
                logger.warning("Anchor chunk id backfill failed: %s", exc)

            result["chunk_ids"] = chunk_ids
            _emit("chunks_stored", 65, {"file_id": file_id, "num_chunks": len(chunk_ids)})

            result["metadata"]["timings_s"]["store_chunks"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Index (Step 6) + statuses (Steps 7-8).
            indexing_results = await self._index_chunks(stored_chunks, chunk_ids)
            result["indexing_results"] = indexing_results
            result["metadata"]["indexers_used"] = list(indexing_results.keys())
            # Surface per-indexer timing breakdown for performance experiments.
            indexer_timings: Dict[str, Any] = {}
            for name, payload in (indexing_results or {}).items():
                if not isinstance(payload, dict):
                    continue
                meta = payload.get("metadata") or {}
                if isinstance(meta, dict):
                    elapsed = meta.get("elapsed_s")
                    if elapsed is not None:
                        indexer_timings[name] = float(elapsed)
            if indexer_timings:
                result["metadata"]["timings_s"]["indexers"] = indexer_timings

            result["metadata"]["timings_s"]["index"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            chunks_updated = await get_thread_pool().run_blocking(
                self._update_indexed_chunks_status,
                chunk_ids,
                indexing_results,
                **kwargs,
            )
            result["metadata"]["timings_s"]["update_chunk_status"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # File status decision mirrors process_file.
            successful_indexers = [k for k, v in (indexing_results or {}).items() if v.get("success")]
            if not self.indexers:
                successful_indexers = []
            if not self.indexers:
                # If no indexers are configured, mark the file as indexed (askable) after chunk storage.
                await get_thread_pool().run_blocking(self._update_file_status_to_indexed, file_id, **kwargs)
                result["success"] = True
                _emit("indexed", 95, {"file_id": file_id, "success": True, "status": "INDEXED", "chunks_updated": chunks_updated})
            elif successful_indexers and len(successful_indexers) == len(self.indexers):
                await get_thread_pool().run_blocking(self._update_file_status_to_indexed, file_id, **kwargs)
                result["success"] = True
                _emit("indexed", 95, {"file_id": file_id, "success": True, "status": "INDEXED", "chunks_updated": chunks_updated})
            else:
                # Partial or total failure.
                result["success"] = False
                result["error_message"] = "all indexers failed" if not successful_indexers else "partial indexing: not all indexers succeeded"
                from encapsulation.data_model.orm_models import FileStatus

                metadata = None
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                except Exception:
                    metadata = None
                if metadata is None or getattr(metadata, "status", None) != FileStatus.DELETED:
                    target_status = FileStatus.PARTIAL_INDEXED if successful_indexers else FileStatus.FAILED
                    await get_thread_pool().run_blocking(
                        self.file_storage.metadata_store.update_file_status,
                        file_id,
                        target_status,
                        **kwargs,
                    )
                _emit("index_failed", 100, {"file_id": file_id, "success": False, "indexers": indexing_results})
                return result

            result["metadata"]["timings_s"]["update_file_status"] = time.perf_counter() - stage_t0
            result["metadata"]["timings_s"]["total"] = time.perf_counter() - started
            _emit("done", 100, {"file_id": file_id, "success": True})
            return result

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Parsed-markdown indexing pipeline failed for file_id {file_id}: {exc}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            result["metadata"]["timings_s"]["total"] = time.perf_counter() - started
            _emit("failed", 100, {"file_id": file_id, "success": False, "error_message": str(exc)})
            try:
                from encapsulation.data_model.orm_models import FileStatus

                metadata = None
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                except Exception:
                    metadata = None
                if metadata is None or getattr(metadata, "status", None) != FileStatus.DELETED:
                    await get_thread_pool().run_blocking(
                        self.file_storage.metadata_store.update_file_status,
                        file_id,
                        FileStatus.FAILED,
                        **kwargs,
                    )
            except Exception:
                pass
            return result

    async def parse_file_only(self, file_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Parse + persist parsed content, but do NOT chunk/index.

        This enables a two-step workflow:
        1) parse-only (status -> PARSED, parsed_content_metadata stored)
        2) ingest/index later (reuse the parsed output; skip parse stage)
        """
        result: Dict[str, Any] = {
            "success": False,
            "file_id": file_id,
            "parsed_content_id": None,
            "error_message": None,
            "metadata": {
                "parser_type": None,
                "md_path": None,
                "output_dir": None,
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
            if file_id is None or not isinstance(file_id, str) or not file_id.strip():
                result["error_message"] = "file_id must be a non-empty string"
                return result

            _emit("start", 1, {"file_id": file_id, "mode": "parse_only"})

            file_content = await get_thread_pool().run_blocking(self.file_storage.get_file_content, file_id)
            if file_content is None:
                raise ValueError(f"File content not found for file_id: {file_id}")

            file_metadata = await get_thread_pool().run_blocking(self.file_storage.get_file_metadata, file_id)
            if file_metadata is None:
                raise ValueError(f"File metadata not found for file_id: {file_id}")

            filename = str(getattr(file_metadata, "filename", "") or "").strip() or f"unknown_file_{file_id}"
            _emit("retrieved", 5, {"file_id": file_id, "filename": filename, "bytes": len(file_content)})

            parser_kwargs = dict(kwargs)
            parser_kwargs.setdefault("source_file_id", file_id)
            parse_results = await self.parser.parse_file(file_data=file_content, filename=filename, **parser_kwargs)
            if not parse_results:
                raise ValueError(f"Parser returned no results for file: {filename}")
            if not isinstance(parse_results, list):
                parse_results = [parse_results]

            if len(parse_results) == 1:
                parse_result = parse_results[0]
                parsed_text = self._extract_text_from_parse_result(parse_result)
            else:
                concatenated: list[str] = []
                for parse_result in parse_results:
                    if parse_result is None:
                        continue
                    text_content = self._extract_text_from_parse_result(parse_result)
                    if text_content:
                        concatenated.append(text_content)
                if not concatenated:
                    raise ValueError("No valid text content extracted from any parse results")
                parsed_text = "\n\n".join(concatenated)
                parse_result = parse_results[0]

            if not isinstance(parsed_text, str) or not parsed_text.strip():
                raise ValueError("No text content extracted from parsed result")
            _emit("parsed", 25, {"file_id": file_id, "chars": len(parsed_text)})

            base_output_dir = getattr(getattr(self, "parser", None), "base_output_dir", None)
            md_path_abs, output_dir_abs = extract_md_path_and_output_dir(parse_result)
            md_path_rel = relativize_under_base(md_path_abs, base_dir=str(base_output_dir) if base_output_dir else None)
            output_dir_rel = relativize_under_base(output_dir_abs, base_dir=str(base_output_dir) if base_output_dir else None)

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

            parsed_data = parsed_text.encode("utf-8", errors="replace")
            parsed_content_id = await get_thread_pool().run_blocking(
                self.parsed_content_storage.store_parsed_content,
                source_file_id=file_id,
                parser_type=parser_type_name,
                parsed_data=parsed_data,
                content_type="text/markdown",
                md_path=md_path_rel,
                output_dir=output_dir_rel,
                **kwargs,
            )
            if not parsed_content_id:
                raise ValueError("Failed to store parsed content")

            result["parsed_content_id"] = parsed_content_id
            result["metadata"]["parser_type"] = parser_type_name
            result["metadata"]["md_path"] = md_path_rel
            result["metadata"]["output_dir"] = output_dir_rel

            _emit("parsed_stored", 60, {"file_id": file_id, "parsed_content_id": parsed_content_id})
            await get_thread_pool().run_blocking(self._update_file_status_to_parsed, file_id, **kwargs)
            _emit("done", 100, {"file_id": file_id, "success": True, "status": "PARSED"})

            result["success"] = True
            return result

        except Exception as exc:  # noqa: BLE001
            err = f"parse-only pipeline failed for file_id {file_id}: {exc}"
            logger.error(err, exc_info=True)
            result["error_message"] = err
            _emit("failed", 100, {"file_id": file_id, "success": False, "error_message": str(exc)})
            try:
                from encapsulation.data_model.orm_models import FileStatus

                metadata = None
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                except Exception:
                    metadata = None
                if metadata is None or getattr(metadata, "status", None) != FileStatus.DELETED:
                    await get_thread_pool().run_blocking(
                        self.file_storage.metadata_store.update_file_status,
                        file_id,
                        FileStatus.FAILED,
                        **kwargs,
                    )
            except Exception:
                pass
            return result

    async def process_file_from_parsed_content_id(
        self,
        *,
        file_id: str,
        parsed_content_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Index a file starting from an existing parsed_content_id (skip parse + skip re-storing parsed content)."""
        result: Dict[str, Any] = {
            "success": False,
            "file_id": file_id,
            "parsed_content_id": parsed_content_id,
            "chunk_ids": [],
            "indexing_results": {},
            "error_message": None,
            "metadata": {
                "parser_type": None,
                "chunker_type": None,
                "num_chunks": 0,
                "indexers_used": [],
                "timings_s": {},
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

        started = time.perf_counter()
        stage_t0 = started
        try:
            if file_id is None or not isinstance(file_id, str) or not file_id.strip():
                result["error_message"] = "file_id must be a non-empty string"
                return result
            if parsed_content_id is None or not isinstance(parsed_content_id, str) or not parsed_content_id.strip():
                result["error_message"] = "parsed_content_id must be a non-empty string"
                return result

            _emit("start", 1, {"file_id": file_id, "mode": "parsed_content_id", "parsed_content_id": parsed_content_id})

            file_metadata = await get_thread_pool().run_blocking(self.file_storage.get_file_metadata, file_id)
            if file_metadata is None:
                raise ValueError(f"File metadata not found for file_id: {file_id}")
            filename = str(getattr(file_metadata, "filename", "") or "").strip() or f"unknown_file_{file_id}"
            result["metadata"]["timings_s"]["load_file_metadata"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            parsed_meta = await get_thread_pool().run_blocking(
                self.parsed_content_storage.metadata_store.get_parsed_content_metadata,
                parsed_content_id,
                **kwargs,
            )
            if parsed_meta is None:
                raise ValueError(f"parsed_content_id not found: {parsed_content_id}")
            if str(getattr(parsed_meta, "source_file_id", "") or "") != str(file_id):
                raise ValueError("parsed_content_id does not belong to file_id")

            blob_key = str(getattr(parsed_meta, "blob_key", "") or "").strip()
            if not blob_key:
                raise ValueError("parsed content blob_key missing")

            parsed_bytes = await get_thread_pool().run_blocking(
                self.parsed_content_storage.blob_store.retrieve,
                blob_key,
                **kwargs,
            )
            parsed_text = parsed_bytes.decode("utf-8", errors="replace") if isinstance(parsed_bytes, (bytes, bytearray)) else str(parsed_bytes or "")
            if not parsed_text.strip():
                raise ValueError("parsed content is empty")

            parser_type_name = str(getattr(parsed_meta, "parser_type", "") or "").strip() or "parsed_content"
            result["metadata"]["parser_type"] = parser_type_name
            _emit("parsed_ready", 15, {"file_id": file_id, "filename": filename, "chars": len(parsed_text)})

            base_output_dir = getattr(getattr(self, "parser", None), "base_output_dir", None)
            md_path_rel = str(getattr(parsed_meta, "md_path", "") or "").strip() or None
            output_dir_rel = str(getattr(parsed_meta, "output_dir", "") or "").strip() or None
            md_path_abs = resolve_under_base(md_path_rel, base_dir=str(base_output_dir) if base_output_dir else None)
            output_dir_abs = resolve_under_base(output_dir_rel, base_dir=str(base_output_dir) if base_output_dir else None)

            pageindex_context = None
            pageindex_info: Dict[str, Any] = {}
            if getattr(self, "pageindex_service", None) is not None:
                try:
                    from config import pageindex as pageindex_cfg

                    if pageindex_cfg.pageindex_enabled():
                        io_manager = self._resolve_io_manager()
                        pageindex_context = self.pageindex_service.build_context(
                            file_id=file_id,
                            filename=filename,
                            markdown=parsed_text,
                            md_path=md_path_abs,
                            output_dir=output_dir_abs,
                            io_manager=io_manager,
                        )
                        pageindex_info = {
                            "sections": len(pageindex_context.tree.nodes),
                            "level_conflict_ratio": pageindex_context.tree.level_conflict_ratio,
                            "uniform_level_flattened": pageindex_context.tree.uniform_level_flattened,
                        }
                except Exception as exc:
                    logger.warning("PageIndex tree build failed for %s: %s", file_id, exc)
                    pageindex_info = {"error": str(exc)}

            result["metadata"]["timings_s"]["pageindex_tree"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Chunk (Step 4).
            chunker_info = self.chunker.get_chunker_info()
            chunker_strategy = chunker_info.get("strategy", type(self.chunker).__name__)
            result["metadata"]["chunker_type"] = chunker_strategy

            chunk_metadata = {
                "source_file_id": file_id,
                "parsed_content_id": parsed_content_id,
                "filename": filename,
                "parser_type": parser_type_name,
                "owner_id": str(getattr(file_metadata, "owner_id", "")),
            }

            chunks = self._chunk_with_optional_strict_page(
                text=parsed_text,
                chunk_metadata=chunk_metadata,
                pageindex_context=pageindex_context,
                md_path=md_path_abs,
                output_dir=output_dir_abs,
                pageindex_info=pageindex_info,
                **kwargs,
            )
            if not chunks:
                raise ValueError("Chunker returned no chunks")

            try:
                from core.file_management.index_text_augmentation import augment_chunk_dict_index_text

                for chunk in chunks:
                    if isinstance(chunk, dict):
                        augment_chunk_dict_index_text(chunk)
            except Exception as exc:
                logger.warning("Failed to augment chunk index_text from filename: %s", exc)

            if pageindex_context is not None:
                try:
                    enrichment = self.pageindex_service.enrich_chunks(context=pageindex_context, chunks=chunks)
                    pageindex_info["chunk_enrichment"] = enrichment
                except Exception as exc:
                    logger.warning("PageIndex chunk enrichment failed for %s: %s", file_id, exc)
                    pageindex_info["chunk_enrichment_error"] = str(exc)
            if pageindex_info:
                result["metadata"]["pageindex"] = pageindex_info

            result["metadata"]["num_chunks"] = len(chunks)
            _emit("chunked", 50, {"file_id": file_id, "num_chunks": len(chunks)})

            result["metadata"]["timings_s"]["chunk"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Store chunks (Step 5) - prefer batch store.
            chunk_ids: list[str] = []
            stored_chunks: List[Dict[str, Any]] = []
            store_batch = getattr(self.chunk_storage, "store_chunks_batch", None)
            if callable(store_batch):
                from config.core.file_management.chunk_ingest_defaults import (
                    CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT,
                )

                # `process_file_from_parsed_content_id` may be called with `owner_id=...` in kwargs.
                # `store_chunks_batch` already receives `owner_id` explicitly from file_metadata, so
                # we must avoid passing duplicate kwargs which would raise a TypeError.
                store_kwargs = dict(kwargs)
                store_kwargs.pop("owner_id", None)

                stored_refs: list[tuple[int, str]] = await get_thread_pool().run_blocking(
                    store_batch,
                    source_parsed_content_id=parsed_content_id,
                    chunker_type=chunker_strategy,
                    chunks=chunks,
                    owner_id=getattr(file_metadata, "owner_id", None),
                    validate_after_store=CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT,
                    **store_kwargs,
                )
                for chunk_index, chunk_id in stored_refs:
                    if not chunk_id:
                        continue
                    chunk_ids.append(chunk_id)
                    try:
                        stored_chunks.append(chunks[int(chunk_index)])
                    except Exception:
                        continue
            else:
                store_kwargs = dict(kwargs)
                store_kwargs.pop("owner_id", None)
                for i, chunk in enumerate(chunks):
                    if chunk is None:
                        continue
                    chunk_data = json.dumps(chunk, ensure_ascii=False).encode("utf-8", errors="replace")
                    chunk_id = await get_thread_pool().run_blocking(
                        self.chunk_storage.store_chunk,
                        source_parsed_content_id=parsed_content_id,
                        chunker_type=chunker_strategy,
                        chunk_data=chunk_data,
                        chunk_index=i,
                        owner_id=getattr(file_metadata, "owner_id", None),
                        validate_after_store=True,
                        **store_kwargs,
                    )
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        stored_chunks.append(chunk)

            if not chunk_ids:
                raise ValueError("No chunks stored successfully")

            # Backfill semantic_unit anchor_chunk_id for graph/dense retrieval.
            try:
                self._backfill_anchor_chunk_ids(stored_chunks, chunk_ids)
                overwrite = getattr(self.chunk_storage, "overwrite_chunk_json", None)
                if callable(overwrite):
                    await get_thread_pool().run_blocking(
                        self._persist_backfilled_anchor_chunk_ids,
                        self.chunk_storage,
                        stored_chunks,
                        chunk_ids,
                    )
            except Exception as exc:
                logger.warning("Anchor chunk id backfill failed: %s", exc)

            result["chunk_ids"] = chunk_ids
            _emit("chunks_stored", 65, {"file_id": file_id, "num_chunks": len(chunk_ids)})

            result["metadata"]["timings_s"]["store_chunks"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            await get_thread_pool().run_blocking(self._update_parsed_content_status_to_chunked, parsed_content_id, **kwargs)
            await get_thread_pool().run_blocking(self._update_file_status_to_chunked, file_id, **kwargs)

            if pageindex_context is not None:
                try:
                    summary_info = await self.pageindex_service.summarize_sections(context=pageindex_context, chunks=stored_chunks)
                    pageindex_info["summaries"] = summary_info
                except Exception as exc:
                    logger.warning("PageIndex summaries failed for %s: %s", file_id, exc)
                    pageindex_info["summary_error"] = str(exc)

                try:
                    pageindex_indexing = await self.pageindex_service.build_indexes(
                        context=pageindex_context,
                        owner_id=str(getattr(file_metadata, "owner_id", "")) if file_metadata else None,
                        base_indexers=self.indexers,
                    )
                    if pageindex_indexing:
                        pageindex_info["indexing"] = pageindex_indexing
                except Exception as exc:
                    logger.warning("PageIndex indexing failed for %s: %s", file_id, exc)
                    pageindex_info["indexing_error"] = str(exc)
                if pageindex_info:
                    result["metadata"]["pageindex"] = pageindex_info

            # Index (Step 6) + statuses (Steps 7-8).
            if self.indexers:
                _emit("indexing", 75, {"file_id": file_id, "indexers": len(self.indexers)})
                indexing_results = await self._index_chunks(stored_chunks, chunk_ids)
                result["indexing_results"] = indexing_results
                result["metadata"]["indexers_used"] = list(indexing_results.keys())
            else:
                indexing_results = {}
                result["indexing_results"] = {}
                result["metadata"]["indexers_used"] = []

            result["metadata"]["timings_s"]["index"] = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()

            # Update chunk status for indexed chunks
            chunks_updated = False
            if indexing_results:
                chunks_updated = await get_thread_pool().run_blocking(
                    self._update_indexed_chunks_status,
                    chunk_ids,
                    indexing_results,
                    **kwargs,
                )

            # File status decision mirrors process_file_from_parsed_markdown/process_file.
            successful_indexers = [k for k, v in (indexing_results or {}).items() if v.get("success")]
            if not self.indexers:
                successful_indexers = []
            if not self.indexers:
                await get_thread_pool().run_blocking(self._update_file_status_to_indexed, file_id, **kwargs)
                result["success"] = True
                _emit("indexed", 95, {"file_id": file_id, "success": True, "status": "INDEXED", "chunks_updated": chunks_updated})
            elif successful_indexers and len(successful_indexers) == len(self.indexers):
                await get_thread_pool().run_blocking(self._update_file_status_to_indexed, file_id, **kwargs)
                result["success"] = True
                _emit("indexed", 95, {"file_id": file_id, "success": True, "status": "INDEXED", "chunks_updated": chunks_updated})
            else:
                result["success"] = False
                result["error_message"] = "all indexers failed" if not successful_indexers else "partial indexing: not all indexers succeeded"
                from encapsulation.data_model.orm_models import FileStatus

                metadata = None
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                except Exception:
                    metadata = None
                if metadata is None or getattr(metadata, "status", None) != FileStatus.DELETED:
                    target_status = FileStatus.PARTIAL_INDEXED if successful_indexers else FileStatus.FAILED
                    await get_thread_pool().run_blocking(
                        self.file_storage.metadata_store.update_file_status,
                        file_id,
                        target_status,
                        **kwargs,
                    )
                _emit("index_failed", 100, {"file_id": file_id, "success": False, "indexers": indexing_results})
                return result

            result["metadata"]["timings_s"]["update_file_status"] = time.perf_counter() - stage_t0
            result["metadata"]["timings_s"]["total"] = time.perf_counter() - started
            _emit("done", 100, {"file_id": file_id, "success": True})
            return result

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Parsed-content indexing pipeline failed for file_id {file_id}: {exc}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            result["metadata"]["timings_s"]["total"] = time.perf_counter() - started
            _emit("failed", 100, {"file_id": file_id, "success": False, "error_message": str(exc)})
            try:
                from encapsulation.data_model.orm_models import FileStatus

                metadata = None
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                except Exception:
                    metadata = None
                if metadata is None or getattr(metadata, "status", None) != FileStatus.DELETED:
                    await get_thread_pool().run_blocking(
                        self.file_storage.metadata_store.update_file_status,
                        file_id,
                        FileStatus.FAILED,
                        **kwargs,
                    )
            except Exception:
                pass
            return result

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

            # Best-effort capture of local parser artifact paths (e.g., MinerU markdown + asset folder).
            base_output_dir = getattr(getattr(self, "parser", None), "base_output_dir", None)
            md_path_abs, output_dir_abs = extract_md_path_and_output_dir(parse_result)
            md_path_rel = relativize_under_base(md_path_abs, base_dir=str(base_output_dir) if base_output_dir else None)
            output_dir_rel = relativize_under_base(output_dir_abs, base_dir=str(base_output_dir) if base_output_dir else None)

            pageindex_context = None
            pageindex_info: Dict[str, Any] = {}
            if getattr(self, "pageindex_service", None) is not None:
                try:
                    from config import pageindex as pageindex_cfg

                    if pageindex_cfg.pageindex_enabled():
                        io_manager = self._resolve_io_manager()
                        pageindex_context = self.pageindex_service.build_context(
                            file_id=file_id,
                            filename=filename,
                            markdown=parsed_text,
                            md_path=resolve_under_base(md_path_rel, base_dir=str(base_output_dir) if base_output_dir else None),
                            output_dir=resolve_under_base(output_dir_rel, base_dir=str(base_output_dir) if base_output_dir else None),
                            io_manager=io_manager,
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
                md_path=md_path_rel,
                output_dir=output_dir_rel,
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

            chunks = self._chunk_with_optional_strict_page(
                text=parsed_text,
                chunk_metadata=chunk_metadata,
                pageindex_context=pageindex_context,
                md_path=md_path_abs,
                output_dir=output_dir_abs,
                pageindex_info=pageindex_info,
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

            store_batch = getattr(self.chunk_storage, "store_chunks_batch", None)
            if callable(store_batch):
                try:
                    from config.core.file_management.chunk_ingest_defaults import (
                        CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT,
                    )

                    stored_refs: list[tuple[int, str]] = await get_thread_pool().run_blocking(
                        store_batch,
                        source_parsed_content_id=parsed_content_id,
                        chunker_type=chunker_strategy,
                        chunks=chunks,
                        owner_id=getattr(file_metadata, "owner_id", None),
                        validate_after_store=CHUNK_BATCH_VALIDATE_AFTER_STORE_DEFAULT,
                        **kwargs,
                    )
                    for chunk_index, chunk_id in stored_refs:
                        if not chunk_id:
                            continue
                        chunk_ids.append(chunk_id)
                        try:
                            stored_chunks.append(chunks[int(chunk_index)])
                        except Exception:
                            continue
                except Exception as e:
                    logger.error("Batch chunk storage failed; falling back to per-chunk store: %s", e)
                    # Fall back to the safer per-chunk path (keeps behavior consistent even if batch is unsupported).
                    for i, chunk in enumerate(chunks):
                        if chunk is None:
                            logger.warning(f"Chunk {i+1}/{len(chunks)} is None, skipping")
                            continue
                        try:
                            chunk_data = json.dumps(chunk, ensure_ascii=False).encode("utf-8", errors="replace")
                            chunk_id = await get_thread_pool().run_blocking(
                                self.chunk_storage.store_chunk,
                                source_parsed_content_id=parsed_content_id,
                                chunker_type=chunker_strategy,
                                chunk_data=chunk_data,
                                chunk_index=i,
                                owner_id=getattr(file_metadata, "owner_id", None),
                                validate_after_store=True,
                                **kwargs,
                            )
                            if chunk_id:
                                chunk_ids.append(chunk_id)
                                stored_chunks.append(chunk)
                        except Exception as e2:
                            logger.error("Failed to store chunk %s/%s: %s", i + 1, len(chunks), e2)
                            continue
            else:
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
                            owner_id=getattr(file_metadata, "owner_id", None),
                            validate_after_store=True,
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
                            logger.info("Skip updating file %s to PARTIAL_INDEXED because status is DELETED", file_id)
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
                    # 部分 indexer 成功：标为 PARTIAL_INDEXED，保持可观测性，避免把可用的索引结果当作“全失败”。
                    from encapsulation.data_model.orm_models import FileStatus

                    result["success"] = False
                    result["error_message"] = "partial indexing: not all indexers succeeded"
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
                                FileStatus.PARTIAL_INDEXED,
                                **kwargs,
                            )
                    except Exception as status_error:
                        logger.error("Failed to update file status to PARTIAL_INDEXED for %s: %s", file_id, status_error)
                    _emit(
                        "index_failed",
                        100,
                        {"file_id": file_id, "success": False, "indexers": result["metadata"]["indexing_summary"]},
                    )
                    return result
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
                        from framework.virtual_paths import is_io_path

                        if is_io_path(markdown_path):
                            io_manager = self._resolve_io_manager()
                            if io_manager is None:
                                raise RuntimeError("io_manager unavailable for io:// markdown_path")
                            text = io_manager.get_text_path(str(markdown_path))
                            if text is None:
                                raise RuntimeError("IOManager returned empty content for io:// markdown_path")
                            return text

                        from pathlib import Path

                        local_path = Path(str(markdown_path)).expanduser().resolve()
                        with open(local_path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read markdown file {markdown_path}: {e}")

        # If we have md_content_path, try to read from it
        if "md_content_path" in parse_result:
            md_path = parse_result["md_content_path"]
            if md_path:
                try:
                    from framework.virtual_paths import is_io_path

                    if is_io_path(md_path):
                        io_manager = self._resolve_io_manager()
                        if io_manager is None:
                            raise RuntimeError("io_manager unavailable for io:// md_content_path")
                        text = io_manager.get_text_path(str(md_path))
                        if text is None:
                            raise RuntimeError("IOManager returned empty content for io:// md_content_path")
                        return text

                    from pathlib import Path

                    local_path = Path(str(md_path)).expanduser().resolve()
                    with open(local_path, "r", encoding="utf-8") as f:
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
            t0 = time.perf_counter()
            raw = await indexer.update_index(chunk_objects)
            elapsed_s = time.perf_counter() - t0

            indexed_ids: list[str] = []
            extra_meta: dict[str, Any] = {}
            if isinstance(raw, dict):
                candidate = raw.get("indexed_ids")
                if candidate is None:
                    candidate = raw.get("chunk_ids")
                if isinstance(candidate, list):
                    indexed_ids = [str(x).strip() for x in candidate if str(x or "").strip()]
                extra_meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            elif isinstance(raw, list):
                indexed_ids = [str(x).strip() for x in raw if str(x or "").strip()]
            elif raw is None:
                indexed_ids = []
            else:
                indexed_ids = [str(raw).strip()] if str(raw).strip() else []

            indexed_count = len(indexed_ids)
            if indexed_count <= 0:
                return {
                    "success": False,
                    "error_message": "indexer returned no indexed ids",
                    "indexed_count": 0,
                    "total_chunks": len(chunk_objects),
                    "indexed_ids": [],
                    "metadata": {"elapsed_s": elapsed_s},
                }

            return {
                "success": True,
                "indexed_count": indexed_count,
                "total_chunks": len(chunk_objects),
                "indexed_ids": indexed_ids,
                "metadata": {"elapsed_s": elapsed_s, **extra_meta},
            }
        except Exception as e:
            logger.error(f"Indexing failed with {indexer_name}: {str(e)}")
            return {
                "success": False,
                "error_message": str(e),
                "indexed_count": 0,
                "total_chunks": len(chunk_objects),
                "metadata": {"elapsed_s": 0.0},
            }
