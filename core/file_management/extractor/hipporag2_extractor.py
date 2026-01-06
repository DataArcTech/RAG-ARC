"""
HippoRAG2 Graph Extractor - Optimized for minimal token usage with TSV format

This extractor follows HippoRAG2's approach:
1. Named Entity Recognition (NER) - extract entities with types in TSV format (entity\ttype)
2. Triple Extraction - construct RDF triples using extracted entities
3. TSV format output to minimize token usage
4. Support for optional entity type specification:
   - If entity_types is specified: only extract entities of those types
   - If entity_types is None: LLM determines entity types automatically
"""

import logging
import json
import os
import re
from pathlib import Path
from typing import List, TYPE_CHECKING, Tuple

from core.file_management.extractor.base import ExtractorBase
from core.file_management.extractor.metadata_keys import (
    BUSINESS_TIME_KEY,
    MINDMAP_ERROR_KEY,
    SDF_ERROR_KEY,
    SDF_HS_KEY,
    SDF_KEY,
    TEMPORAL_ERROR_KEY,
)
from core.prompts.hipporag2_extractor_prompt import (
    HIPPORAG2_NER_SYSTEM, HIPPORAG2_NER_SYSTEM_WITH_TYPES,
    HIPPORAG2_NER_ONE_SHOT_INPUT, HIPPORAG2_NER_ONE_SHOT_OUTPUT,
    HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES, HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES,
    HIPPORAG2_NER_PROMPT, HIPPORAG2_NER_PROMPT_WITH_TYPES,
    HIPPORAG2_TRIPLE_SYSTEM,
    HIPPORAG2_TRIPLE_SCHEMA_HINT,
    HIPPORAG2_TRIPLE_ONE_SHOT_INPUT,
    HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT,
    HIPPORAG2_TRIPLE_PROMPT,
    HIPPORAG2_NER_SYSTEM_ZH, HIPPORAG2_NER_SYSTEM_WITH_TYPES_ZH,
    HIPPORAG2_NER_ONE_SHOT_INPUT_ZH, HIPPORAG2_NER_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES_ZH, HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES_ZH,
    HIPPORAG2_NER_PROMPT_ZH, HIPPORAG2_NER_PROMPT_WITH_TYPES_ZH,
    HIPPORAG2_TRIPLE_SYSTEM_ZH,
    HIPPORAG2_TRIPLE_SCHEMA_HINT_ZH,
    HIPPORAG2_TRIPLE_ONE_SHOT_INPUT_ZH,
    HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_TRIPLE_PROMPT_ZH,
    HIPPORAG2_MINDMAP_SYSTEM, HIPPORAG2_MINDMAP_ONE_SHOT_INPUT, HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT, HIPPORAG2_MINDMAP_PROMPT,
    HIPPORAG2_MINDMAP_SYSTEM_ZH, HIPPORAG2_MINDMAP_ONE_SHOT_INPUT_ZH, HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT_ZH, HIPPORAG2_MINDMAP_PROMPT_ZH
)
from core.prompts.hipporag2_sdf_prompt import (
    HIPPORAG2_SDF_HS_ONE_SHOT_INPUT,
    HIPPORAG2_SDF_HS_ONE_SHOT_INPUT_ZH,
    HIPPORAG2_SDF_HS_ONE_SHOT_OUTPUT,
    HIPPORAG2_SDF_HS_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_SDF_HS_PROMPT,
    HIPPORAG2_SDF_HS_PROMPT_ZH,
    HIPPORAG2_SDF_HS_SYSTEM,
    HIPPORAG2_SDF_HS_SYSTEM_ZH,
)
from core.prompts.hipporag2_temporal_prompt import (
    HIPPORAG2_TEMPORAL_PROMPT_EN,
    HIPPORAG2_TEMPORAL_PROMPT_ZH,
    HIPPORAG2_TEMPORAL_SYSTEM_EN,
    HIPPORAG2_TEMPORAL_SYSTEM_ZH,
)
from core.knowledge_graph.schema import load_schema_from_yaml, normalize_relation_token
from core.knowledge_graph.sdf import hs_to_sdf_schema, parse_hs_blocks
from encapsulation.data_model.schema import Chunk, GraphData

if TYPE_CHECKING:
    from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig

logger = logging.getLogger(__name__)

class HippoRAG2Extractor(ExtractorBase):
    """
    HippoRAG2 Graph Extractor with TSV format for minimal token usage

    Features:
    - Two-stage extraction: NER first, then Triple Extraction
    - TSV format output (tab-separated values): entity\ttype
    - Always outputs entity types (LLM determines types if not specified)
    - Minimal token usage compared to JSON format
    - Optional entity type specification for targeted extraction
    """

    def __init__(self, config: "HippoRAG2ExtractorConfig"):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.entity_types = getattr(config, 'entity_types', None)  # Optional entity types to extract
        self._prompt_cache: dict[str, str] = {}
        self._kg_schema = None
        self._kg_domain = None
        self._kg_domain_schema = None
        self._schema_hint_cache: dict[str, str] = {}
        self._load_kg_schema_for_prompting()

    def _load_kg_schema_for_prompting(self) -> None:
        cfg_flag = getattr(self.config, "schema_aware_triple_extraction", None)
        cfg_path = str(getattr(self.config, "kg_schema_path", "") or "").strip()
        env_path = os.getenv("KG_SCHEMA_PATH", "").strip()
        schema_path = cfg_path or env_path
        enabled = bool(schema_path) if cfg_flag is None else bool(cfg_flag)
        if not enabled or not schema_path:
            return
        try:
            schema = load_schema_from_yaml(schema_path)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load KG schema for schema-aware prompting: %s", exc, exc_info=True)
            return
        domain_override = str(getattr(self.config, "schema_prompt_domain", "") or "").strip()
        domain = domain_override or getattr(schema, "default_domain", None) or "default"
        try:
            domain_schema = schema.for_domain(domain)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to resolve KG schema domain '%s': %s", domain, exc, exc_info=True)
            return
        self._kg_schema = schema
        self._kg_domain = str(domain)
        self._kg_domain_schema = domain_schema

    def _schema_hint_for_triples(self, *, language: str) -> str:
        schema = getattr(self, "_kg_schema", None)
        domain_schema = getattr(self, "_kg_domain_schema", None)
        if schema is None or domain_schema is None:
            return ""
        max_allowed = int(getattr(self.config, "schema_prompt_max_allowed_relations", 80) or 0)
        max_aliases = int(getattr(self.config, "schema_prompt_max_relation_aliases", 120) or 0)
        domain = getattr(self, "_kg_domain", None) or getattr(schema, "default_domain", None) or "default"
        cache_key = f"{language}:{domain}:{max_allowed}:{max_aliases}"
        cache = getattr(self, "_schema_hint_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_schema_hint_cache", cache)
        cached = cache.get(cache_key)
        if isinstance(cached, str):
            return cached

        allowed = sorted({str(item).strip() for item in (domain_schema.allowed_relations or set()) if str(item).strip()})
        if max_allowed > 0:
            allowed = allowed[:max_allowed]
        allowed_block = "\n".join(allowed) if allowed else "(none)"

        alias_map: dict[str, str] = {}
        if max_aliases != 0 and isinstance(domain_schema.relation_aliases, dict):
            items: list[tuple[str, str]] = []
            for raw_key, raw_value in domain_schema.relation_aliases.items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                canonical = normalize_relation_token(str(raw_value or ""))
                if domain_schema.allowed_relations and canonical not in domain_schema.allowed_relations:
                    continue
                items.append((key, canonical))
            items.sort(key=lambda pair: pair[0])
            if max_aliases > 0:
                items = items[:max_aliases]
            alias_map = {k: v for k, v in items}
        alias_json = json.dumps(alias_map, ensure_ascii=False, separators=(",", ":"), default=str) if alias_map else "{}"

        template = HIPPORAG2_TRIPLE_SCHEMA_HINT_ZH if language == "zh" else HIPPORAG2_TRIPLE_SCHEMA_HINT
        hint = template.format(allowed_predicates=allowed_block, predicate_aliases_json=alias_json).strip()
        cache[cache_key] = hint
        return hint

    def detect_language(self, text: str) -> str:
        """
        Detect text language (Chinese or English)

        Args:
            text: Input text to detect language

        Returns:
            'zh' for Chinese, 'en' for English
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'\s', '', text))

        if total_chars == 0:
            return 'en'

        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.1 else 'en'

    async def extract(self, chunk: Chunk) -> GraphData:
        """
        Main extraction method for HippoRAG2

        Two-stage extraction: NER first, then Triple Extraction

        Args:
            chunk: Input chunk to extract from

        Returns:
            GraphData with entities and relations
        """
        if not chunk.content:
            return GraphData()
        return await self.extract_two_stage(chunk)

    async def extract_two_stage(self, chunk: Chunk) -> GraphData:
        """
        Two-stage extraction: NER first, then Triple Extraction
        More accurate and follows HippoRAG2's original approach
        Optionally extracts mind map and saves to chunk.metadata
        """
        # Stage 1: Named Entity Recognition
        entities = await self.extract_entities(chunk.content)

        # Stage 2: Triple Extraction using extracted entities
        triples: list[tuple[str, str, str]] = []
        if entities:
            triples = await self.extract_triples(chunk.content, entities)
        else:
            self.logger.warning("No entities extracted, skipping triple extraction")

        # Stage 3: Mind Map Extraction (optional; non-fatal, but must be observable)
        if bool(getattr(self.config, "enable_mindmap_extraction", False)):
            try:
                mindmap = await self.extract_mindmap(chunk.content)
            except Exception as exc:
                self.logger.error("Error in mind map extraction: %s", exc, exc_info=True)
                chunk.metadata[MINDMAP_ERROR_KEY] = {"exception_type": exc.__class__.__name__, "message": str(exc)}
                mindmap = {}

            if mindmap:
                chunk.metadata["mindmap"] = mindmap

        # Stage 4: Business-time extraction (temporal; non-fatal, but must be observable)
        if bool(getattr(self.config, "enable_temporal_extraction", True)):
            try:
                business_time = await self.extract_business_time(chunk.content)
            except Exception as exc:
                self.logger.error("Error in temporal extraction: %s", exc, exc_info=True)
                chunk.metadata[TEMPORAL_ERROR_KEY] = {"exception_type": exc.__class__.__name__, "message": str(exc)}
                business_time = {}
            if business_time:
                chunk.metadata[BUSINESS_TIME_KEY] = business_time

        # Stage 5: SDF (process schema) extraction (non-fatal, but must be observable)
        if bool(getattr(self.config, "enable_sdf_extraction", False)):
            try:
                hs_text = await self.extract_sdf_hs(chunk.content)
                max_events = int(getattr(self.config, "sdf_max_events_per_chunk", 40) or 40)
                hs_events = parse_hs_blocks(hs_text)[: max(0, max_events)]
                doc_namespace = str(chunk.metadata.get("source_file_id") or chunk.metadata.get("file_id") or "").strip()
                sdf = hs_to_sdf_schema(
                    hs_events=hs_events,
                    owner_id=str(chunk.owner_id or chunk.metadata.get("owner_id") or "").strip() or None,
                    doc_namespace=doc_namespace or None,
                    schema_version="v0",
                    default_temporal=chunk.metadata.get(BUSINESS_TIME_KEY) if isinstance(chunk.metadata.get(BUSINESS_TIME_KEY), dict) else None,
                )
                if sdf:
                    chunk.metadata[SDF_KEY] = sdf
                if bool(getattr(self.config, "sdf_store_raw_hs", False)) and hs_text.strip():
                    chunk.metadata[SDF_HS_KEY] = hs_text
            except Exception as exc:
                self.logger.error("Error in SDF extraction: %s", exc, exc_info=True)
                chunk.metadata[SDF_ERROR_KEY] = {"exception_type": exc.__class__.__name__, "message": str(exc)}

        # Convert to GraphData format
        graph = self.build_graph_data(entities, triples)
        if chunk.metadata.get(BUSINESS_TIME_KEY):
            graph.metadata[BUSINESS_TIME_KEY] = chunk.metadata.get(BUSINESS_TIME_KEY)
        if chunk.metadata.get(TEMPORAL_ERROR_KEY):
            graph.metadata[TEMPORAL_ERROR_KEY] = chunk.metadata.get(TEMPORAL_ERROR_KEY)
        if chunk.metadata.get(MINDMAP_ERROR_KEY):
            graph.metadata[MINDMAP_ERROR_KEY] = chunk.metadata.get(MINDMAP_ERROR_KEY)
        if chunk.metadata.get(SDF_KEY):
            graph.metadata[SDF_KEY] = chunk.metadata.get(SDF_KEY)
        if chunk.metadata.get(SDF_ERROR_KEY):
            graph.metadata[SDF_ERROR_KEY] = chunk.metadata.get(SDF_ERROR_KEY)
        return graph

    async def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Stage 1: Extract named entities from text

        Always extracts entities with types in TSV format: entity\ttype
        - If entity_types is specified: only extract those types
        - If entity_types is None: LLM determines types automatically

        Returns:
            List of (entity_name, entity_type) tuples
        """
        prompt = self.build_ner_prompt(text)
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        entities = self.parse_ner_response(response)
        self.logger.info("Extracted %s entities", len(entities))
        return entities

    async def extract_triples(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """
        Stage 2: Extract triples using extracted entities

        Args:
            text: Original text
            entities: List of (entity_name, entity_type) tuples

        Returns:
            List of (subject, predicate, object) triples
        """
        entity_names = [entity[0] for entity in entities]
        prompt = self.build_triple_prompt(text, entity_names)
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        triples = self.parse_triple_response(response)
        self.logger.info("Extracted %s triples", len(triples))
        return triples

    def build_ner_prompt(self, text: str) -> str:
        """
        Build NER prompt - always outputs entity types in TSV format
        Supports both Chinese and English

        Args:
            text: Input text to extract entities from

        Returns:
            Formatted prompt string

        Note:
            - If self.entity_types is specified: uses HIPPORAG2_NER_PROMPT_WITH_TYPES
            - If self.entity_types is None: uses HIPPORAG2_NER_PROMPT (LLM auto-determines types)
            - Both formats output entity\ttype TSV format
            - Language is auto-detected (Chinese or English)
        """
        # Detect language
        language = self.detect_language(text)

        custom_template = getattr(self.config, "ner_prompt", None)
        if isinstance(custom_template, str) and custom_template.strip():
            return self._render_custom_prompt(
                custom_template,
                passage=text,
                entities="",
                entity_types=", ".join(self.entity_types) if self.entity_types else "",
                language=language,
                schema_hint="",
            )

        custom_path = getattr(self.config, "ner_prompt_path", None)
        if isinstance(custom_path, str) and custom_path.strip():
            template = self._read_prompt_file(custom_path)
            return self._render_custom_prompt(
                template,
                passage=text,
                entities="",
                entity_types=", ".join(self.entity_types) if self.entity_types else "",
                language=language,
                schema_hint="",
            )

        if self.entity_types:
            # Use entity type-specific prompt (only extract specified types)
            entity_types_str = ', '.join(self.entity_types)
            if language == 'zh':
                return HIPPORAG2_NER_PROMPT_WITH_TYPES_ZH.format(
                    system=HIPPORAG2_NER_SYSTEM_WITH_TYPES_ZH,
                    entity_types=entity_types_str,
                    example_input=HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES_ZH,
                    example_output=HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES_ZH,
                    passage=text
                )
            else:
                return HIPPORAG2_NER_PROMPT_WITH_TYPES.format(
                    system=HIPPORAG2_NER_SYSTEM_WITH_TYPES,
                    entity_types=entity_types_str,
                    example_input=HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES,
                    example_output=HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES,
                    passage=text
                )
        else:
            # Use auto-type prompt (LLM determines entity types)
            if language == 'zh':
                return HIPPORAG2_NER_PROMPT_ZH.format(
                    system=HIPPORAG2_NER_SYSTEM_ZH,
                    example_input=HIPPORAG2_NER_ONE_SHOT_INPUT_ZH,
                    example_output=HIPPORAG2_NER_ONE_SHOT_OUTPUT_ZH,
                    passage=text
                )
            else:
                return HIPPORAG2_NER_PROMPT.format(
                    system=HIPPORAG2_NER_SYSTEM,
                    example_input=HIPPORAG2_NER_ONE_SHOT_INPUT,
                    example_output=HIPPORAG2_NER_ONE_SHOT_OUTPUT,
                    passage=text
                )

    def build_triple_prompt(self, text: str, entities: List[str]) -> str:
        """
        Build triple extraction prompt
        Supports both Chinese and English
        """
        entities_str = '\n'.join(entities)

        # Detect language
        language = self.detect_language(text)

        schema_hint = self._schema_hint_for_triples(language=language)

        custom_template = getattr(self.config, "triple_prompt", None)
        if isinstance(custom_template, str) and custom_template.strip():
            return self._render_custom_prompt(
                custom_template,
                passage=text,
                entities=entities_str,
                entity_types=", ".join(self.entity_types) if self.entity_types else "",
                language=language,
                schema_hint=schema_hint,
            )

        custom_path = getattr(self.config, "triple_prompt_path", None)
        if isinstance(custom_path, str) and custom_path.strip():
            template = self._read_prompt_file(custom_path)
            return self._render_custom_prompt(
                template,
                passage=text,
                entities=entities_str,
                entity_types=", ".join(self.entity_types) if self.entity_types else "",
                language=language,
                schema_hint=schema_hint,
            )

        if language == 'zh':
            return HIPPORAG2_TRIPLE_PROMPT_ZH.format(
                system=(HIPPORAG2_TRIPLE_SYSTEM_ZH + ("\n\n" + schema_hint if schema_hint else "")).strip(),
                example_input=HIPPORAG2_TRIPLE_ONE_SHOT_INPUT_ZH,
                example_output=HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT_ZH,
                passage=text,
                entities=entities_str
            )
        else:
            return HIPPORAG2_TRIPLE_PROMPT.format(
                system=(HIPPORAG2_TRIPLE_SYSTEM + ("\n\n" + schema_hint if schema_hint else "")).strip(),
                example_input=HIPPORAG2_TRIPLE_ONE_SHOT_INPUT,
                example_output=HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT,
                passage=text,
                entities=entities_str
            )

    @staticmethod
    def _render_custom_prompt(
        template: str,
        *,
        passage: str,
        entities: str,
        entity_types: str,
        language: str,
        schema_hint: str,
    ) -> str:
        """
        Render a custom prompt template supplied via config.

        Supported placeholders (optional):
        - {passage}
        - {entities}
        - {entity_types}
        - {language}
        - {schema_hint}

        If formatting fails or placeholders are absent, fall back to appending the passage.
        """
        raw = str(template or "").strip()
        if not raw:
            return ""
        if "{passage}" in raw or "{entities}" in raw or "{entity_types}" in raw or "{language}" in raw or "{schema_hint}" in raw:
            try:
                return raw.format(
                    passage=passage,
                    entities=entities,
                    entity_types=entity_types,
                    language=language,
                    schema_hint=schema_hint,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to format custom prompt template; falling back to appending passage: %s",
                    exc,
                    exc_info=True,
                )
        return f"{raw}\n\n{passage}"

    def parse_ner_response(self, response: str) -> List[Tuple[str, str]]:
        """
        Parse NER response in TSV format

        Expected format (always with entity types):
        ### ENTITIES
        Entity1\ttype1
        Entity2\ttype2
        ...

        Args:
            response: LLM response string

        Returns:
            List of (entity_name, entity_type) tuples

        Note:
            All entities must have types in TSV format: entity\ttype
            If tab is missing, entity type defaults to 'UNKNOWN'
        """
        entities = []
        in_entities_section = False

        for line in response.strip().split('\n'):
            line = line.strip()

            if not line:
                continue

            if line.startswith('### ENTITIES'):
                in_entities_section = True
                continue

            if line.startswith('###'):
                in_entities_section = False
                continue

            if in_entities_section and line:
                # Parse entity\ttype format (required)
                if '\t' in line:
                    parts = line.split('\t')
                    entity_name = parts[0].strip()
                    entity_type = parts[1].strip() if len(parts) > 1 else 'UNKNOWN'
                    entities.append((entity_name, entity_type))
                else:
                    # Fallback: if no tab found, use UNKNOWN type
                    self.logger.warning(f"Entity without type (missing tab): {line}")
                    entities.append((line, 'UNKNOWN'))

        return entities

    def parse_triple_response(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Parse triple response in TSV format
        
        Expected format:
        ### TRIPLES
        subject\tpredicate\tobject
        ...
        """
        triples: list[tuple[str, str, str]] = []
        text = str(response or "").strip()
        if not text:
            return triples

        in_triples_section = False
        saw_triples_header = False

        # Notes:
        # - Production LLMs sometimes vary casing ("### Triples") or omit the section header entirely.
        # - Triple rows are unambiguously identified by having at least 2 tab separators (3 columns).
        # - We treat header detection as a guide, not a hard requirement, to avoid silently discarding valid triples.
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Skip code fence markers; content inside may still include TSV lines.
            if line.startswith("```"):
                continue

            normalized_header = line.lstrip("#").strip().lower()
            if normalized_header in {"triples", "triple", "relations", "relationships"}:
                in_triples_section = True
                saw_triples_header = True
                continue

            if line.startswith("###"):
                in_triples_section = False
                continue

            if saw_triples_header and not in_triples_section:
                continue

            # Accept bullet-prefixed TSV rows.
            candidate = line.lstrip("-•* ").strip()
            if candidate.count("\t") < 2:
                continue

            parts = candidate.split("\t")
            if len(parts) < 3:
                continue

            subject = parts[0].strip()
            predicate = parts[1].strip()
            obj = parts[2].strip()
            if not subject or not predicate or not obj:
                continue
            triples.append((subject, predicate, obj))

        return triples

    def build_graph_data(self, entities: List[Tuple[str, str]], triples: List[Tuple[str, str, str]]) -> GraphData:
        """
        Convert entities and triples to GraphData format

        Args:
            entities: List of (entity_name, entity_type) tuples
            triples: List of (subject, predicate, object) tuples

        Returns:
            GraphData object
        """
        # Build entity list with IDs
        entity_list = []
        entity_name_to_id = {}

        for i, (entity_name, entity_type) in enumerate(entities):
            entity_id = f"e{i+1}"
            entity_list.append({
                'id': entity_id,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'attributes': {}
            })
            entity_name_to_id[entity_name] = entity_id

        # Build relation list using entity names (as per RAG-ARC convention)
        relation_list = []
        for subject, predicate, obj in triples:
            # Use entity names directly (RAG-ARC's graph_retrieval expects names)
            relation_list.append([subject, predicate, obj])

        return GraphData(entities=entity_list, relations=relation_list, metadata={})

    async def extract_mindmap(self, text: str) -> dict:
        """
        Stage 3: Extract mind map structure from text
        
        Args:
            text: Original text
            
        Returns:
            Dictionary with mind map structure (hierarchical TSV format)
        """
        prompt = self.build_mindmap_prompt(text)
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        mindmap_data = self.parse_mindmap_response(response)
        node_count = len(mindmap_data.get("nodes", []))
        self.logger.info("Extracted mind map with %s nodes", node_count)
        return mindmap_data

    def build_mindmap_prompt(self, text: str) -> str:
        """
        Build mind map extraction prompt
        Supports both Chinese and English
        """
        # Detect language
        language = self.detect_language(text)
        
        if language == 'zh':
            return HIPPORAG2_MINDMAP_PROMPT_ZH.format(
                system=HIPPORAG2_MINDMAP_SYSTEM_ZH,
                example_input=HIPPORAG2_MINDMAP_ONE_SHOT_INPUT_ZH,
                example_output=HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT_ZH,
                passage=text
            )
        else:
            return HIPPORAG2_MINDMAP_PROMPT.format(
                system=HIPPORAG2_MINDMAP_SYSTEM,
                example_input=HIPPORAG2_MINDMAP_ONE_SHOT_INPUT,
                example_output=HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT,
                passage=text
            )

    def parse_mindmap_response(self, response: str) -> dict:
        """
        Parse mind map response in TSV format
        
        Expected format:
        ### MINDMAP
        1\tcontent1
        1.1\tcontent2
        1.1.1\tcontent3
        ...
        
        Args:
            response: LLM response string
            
        Returns:
            Dictionary with mind map structure:
            {
                'nodes': [
                    {'level': '1', 'content': 'content1'},
                    {'level': '1.1', 'content': 'content2'},
                    ...
                ]
            }
        """
        nodes = []
        in_mindmap_section = False
        
        for line in response.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('### MINDMAP'):
                in_mindmap_section = True
                continue
            
            if line.startswith('###'):
                in_mindmap_section = False
                continue
            
            if in_mindmap_section and '\t' in line:
                parts = line.split('\t', 1)  # Split only on first tab
                if len(parts) >= 2:
                    level = parts[0].strip()
                    content = parts[1].strip()
                    nodes.append({'level': level, 'content': content})
        
        return {'nodes': nodes}

    async def extract_business_time(self, text: str) -> dict:
        """
        Extract business-time fields (effective_date / valid_from / valid_to) via LLM.

        Contract:
        - Returns a JSON-serializable dict with keys subset of {effective_date, valid_from, valid_to, confidence}.
        - Returns {} when unknown or unparseable (no silent swallowing of exceptions at caller).
        """
        prompt = self.build_temporal_prompt(text)
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        payload = self._parse_json_object(str(response or ""))
        if not isinstance(payload, dict):
            return {}

        out: dict = {}
        for key in ("effective_date", "valid_from", "valid_to"):
            val = payload.get(key)
            if val is None:
                continue
            sval = str(val).strip()
            if sval:
                out[key] = sval
        conf = payload.get("confidence")
        if isinstance(conf, (int, float)):
            out["confidence"] = float(conf)
        return out

    def build_temporal_prompt(self, text: str) -> str:
        language = self.detect_language(text)

        custom_template = getattr(self.config, "temporal_prompt", None)
        if isinstance(custom_template, str) and custom_template.strip():
            return self._render_custom_prompt(
                custom_template,
                passage=text,
                entities="",
                entity_types="",
                language=language,
                schema_hint="",
            )

        custom_path = getattr(self.config, "temporal_prompt_path", None)
        if isinstance(custom_path, str) and custom_path.strip():
            template = self._read_prompt_file(custom_path)
            return self._render_custom_prompt(
                template,
                passage=text,
                entities="",
                entity_types="",
                language=language,
                schema_hint="",
            )

        if language == "zh":
            return HIPPORAG2_TEMPORAL_PROMPT_ZH.format(system=HIPPORAG2_TEMPORAL_SYSTEM_ZH, passage=text)
        return HIPPORAG2_TEMPORAL_PROMPT_EN.format(system=HIPPORAG2_TEMPORAL_SYSTEM_EN, passage=text)

    async def extract_sdf_hs(self, text: str) -> str:
        prompt = self.build_sdf_hs_prompt(text)
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        return str(response or "").strip()

    def build_sdf_hs_prompt(self, text: str) -> str:
        language = self.detect_language(text)
        custom_template = getattr(self.config, "sdf_hs_prompt", None)
        if isinstance(custom_template, str) and custom_template.strip():
            return self._render_custom_prompt(
                custom_template,
                passage=text,
                entities="",
                entity_types="",
                language=language,
                schema_hint="",
            )
        custom_path = getattr(self.config, "sdf_hs_prompt_path", None)
        if isinstance(custom_path, str) and custom_path.strip():
            template = self._read_prompt_file(custom_path)
            return self._render_custom_prompt(
                template,
                passage=text,
                entities="",
                entity_types="",
                language=language,
                schema_hint="",
            )
        if language == "zh":
            return HIPPORAG2_SDF_HS_PROMPT_ZH.format(
                system=HIPPORAG2_SDF_HS_SYSTEM_ZH,
                example_input=HIPPORAG2_SDF_HS_ONE_SHOT_INPUT_ZH,
                example_output=HIPPORAG2_SDF_HS_ONE_SHOT_OUTPUT_ZH,
                passage=text,
            )
        return HIPPORAG2_SDF_HS_PROMPT.format(
            system=HIPPORAG2_SDF_HS_SYSTEM,
            example_input=HIPPORAG2_SDF_HS_ONE_SHOT_INPUT,
            example_output=HIPPORAG2_SDF_HS_ONE_SHOT_OUTPUT,
            passage=text,
        )

    def _read_prompt_file(self, path: str) -> str:
        token = str(path or "").strip()
        if not token:
            return ""
        cached = self._prompt_cache.get(token)
        if cached is not None:
            return cached
        raw = Path(token).read_text(encoding="utf-8")
        self._prompt_cache[token] = raw
        return raw

    @staticmethod
    def _parse_json_object(raw: str) -> dict | None:
        text = str(raw or "").strip()
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
