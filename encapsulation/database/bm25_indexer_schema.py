import logging
import os
from typing import List, Optional

from encapsulation.data_model.schema import Chunk
from encapsulation.database.bm25_indexer_constants import EXCLUDED_METADATA_FIELDS
from encapsulation.database.bm25_indexer_tantivy import Filter, Index, SchemaBuilder, TextAnalyzerBuilder, Tokenizer

logger = logging.getLogger(__name__)


class _BM25IndexBuilderSchemaMixin:
    def _tantivy_meta_path(self) -> str:
        return os.path.join(self.config.index_path, "meta.json")

    def _extract_string_fields_from_chunks(self, chunks: List[Chunk]) -> set[str]:
        """Extract string fields from chunk metadata for dynamic schema creation

        Args:
            chunks: List of chunks to analyze

        Returns:
            Set of field names that contain string values
        """
        string_fields = set()

        for chunk in chunks:
            if not chunk.metadata:
                continue

            for key, value in chunk.metadata.items():
                # Only include string fields for filtering, exclude system/score fields
                if isinstance(value, str) and key not in EXCLUDED_METADATA_FIELDS:
                    string_fields.add(key)

        logger.debug(f"Extracted dynamic string fields: {string_fields}")
        return string_fields

    def _initialize_index(self, chunks: Optional[List[Chunk]] = None) -> None:
        """Initialize the Tantivy index with dynamic schema based on chunk metadata

        Args:
            chunks: Optional list of chunks to analyze for dynamic field creation
        """
        if self._index is not None:
            return

        # Build schema with core fields
        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw", fast=True)
        schema_builder.add_text_field("content", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=True)
        schema_builder.add_json_field("metadata", stored=True)

        # Add owner_id field for user isolation (stored for retrieval)
        schema_builder.add_text_field("owner_id", stored=True, tokenizer_name="raw", fast=True)
        logger.debug("Added owner_id field for user isolation")

        # Add dynamic fields based on chunk metadata
        if chunks:
            dynamic_fields = self._extract_string_fields_from_chunks(chunks)
            for field_name in dynamic_fields:
                if field_name == "owner_id":  # Skip owner_id as it's already added
                    continue
                schema_builder.add_text_field(field_name, tokenizer_name="raw", stored=False, fast=True)
                logger.debug(f"Added dynamic field: {field_name}")

        self._schema = schema_builder.build()

        # Load existing index or create new one
        if os.path.exists(self.config.index_path):
            with os.scandir(self.config.index_path) as entries:
                has_files = any(entries)
            if has_files:
                # Tantivy index directories must contain `meta.json`. If it's missing, we consider the directory
                # invalid (often only stale `.tantivy-*.lock` files after crashes) and recreate a fresh index.
                if not os.path.exists(self._tantivy_meta_path()):
                    logger.warning(
                        "BM25 index dir has files but is missing meta.json; recreating empty index at: %s",
                        self.config.index_path,
                    )
                    # Safety: only auto-clean if the directory only contains Tantivy lock files. Any other
                    # unexpected files might indicate a misconfiguration; require manual intervention then.
                    entries = list(os.scandir(self.config.index_path))
                    unexpected = [
                        e.name
                        for e in entries
                        if not (e.is_file(follow_symlinks=False) and e.name.startswith(".tantivy-") and e.name.endswith(".lock"))
                    ]
                    if unexpected:
                        raise RuntimeError(
                            "BM25 index dir is missing meta.json but contains unexpected entries; "
                            f"refusing to auto-clean: path={self.config.index_path} unexpected={unexpected[:10]}"
                        )

                    for entry in entries:
                        try:
                            os.remove(entry.path)
                        except FileNotFoundError:
                            continue

                    self._index = Index(self._schema, path=self.config.index_path)
                else:
                    logger.info(f"Loading existing index from: {self.config.index_path}")
                    try:
                        self._index = Index.open(self.config.index_path)
                        # Get schema from the loaded index
                        self._schema = self._index.schema
                    except Exception as e:
                        logger.error(f"Failed to load existing index at {self.config.index_path}: {str(e)}")
                        logger.error("Please check index intergerity or delete manually")
                        raise RuntimeError("Index corrupted or incompatible - manual intervention required")
            else:
                logger.info(f"Creating new index at existing empty directory: {self.config.index_path}")
                self._index = Index(self._schema, path=self.config.index_path)
        else:
            logger.info(f"Creating new index at: {self.config.index_path}")
            os.makedirs(self.config.index_path, exist_ok=True)
            self._index = Index(self._schema, path=self.config.index_path)

        # Always register tokenizers when index is loaded/created
        if not self._tokenizers_registered:
            self._register_tokenizers()

        logger.info("Tantivy index initialized successfully")

    def _register_tokenizers(self) -> None:
        """Register tokenizers to avoid duplicate registration"""
        if self._tokenizers_registered or self._index is None:
            return

        try:
            custom_analyzer = (
                TextAnalyzerBuilder(Tokenizer.whitespace())
                .filter(Filter.lowercase())
                .filter(Filter.custom_stopword(self.tokenizer_manager.get_stopwords()))
                .build()
            )
            self._index.register_tokenizer("custom", custom_analyzer)

            raw_analyzer = TextAnalyzerBuilder(Tokenizer.raw()).build()
            self._index.register_tokenizer("raw", raw_analyzer)

            self._tokenizers_registered = True
            logger.debug("Tokenizers registered successfully")

        except Exception as e:
            logger.error(f"Failed to register tokenizers: {e}")
            raise
