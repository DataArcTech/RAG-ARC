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
import re
from typing import List, TYPE_CHECKING, Tuple

from core.file_management.extractor.base import ExtractorBase
from core.file_management.extractor.metadata_keys import BUSINESS_TIME_KEY, MINDMAP_ERROR_KEY, TEMPORAL_ERROR_KEY
from core.prompts.hipporag2_extractor_prompt import (
    HIPPORAG2_NER_SYSTEM, HIPPORAG2_NER_SYSTEM_WITH_TYPES,
    HIPPORAG2_NER_ONE_SHOT_INPUT, HIPPORAG2_NER_ONE_SHOT_OUTPUT,
    HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES, HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES,
    HIPPORAG2_NER_PROMPT, HIPPORAG2_NER_PROMPT_WITH_TYPES,
    HIPPORAG2_TRIPLE_SYSTEM, HIPPORAG2_TRIPLE_ONE_SHOT_INPUT, HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT, HIPPORAG2_TRIPLE_PROMPT,
    HIPPORAG2_NER_SYSTEM_ZH, HIPPORAG2_NER_SYSTEM_WITH_TYPES_ZH,
    HIPPORAG2_NER_ONE_SHOT_INPUT_ZH, HIPPORAG2_NER_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES_ZH, HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES_ZH,
    HIPPORAG2_NER_PROMPT_ZH, HIPPORAG2_NER_PROMPT_WITH_TYPES_ZH,
    HIPPORAG2_TRIPLE_SYSTEM_ZH, HIPPORAG2_TRIPLE_ONE_SHOT_INPUT_ZH, HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT_ZH, HIPPORAG2_TRIPLE_PROMPT_ZH,
    HIPPORAG2_MINDMAP_SYSTEM, HIPPORAG2_MINDMAP_ONE_SHOT_INPUT, HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT, HIPPORAG2_MINDMAP_PROMPT,
    HIPPORAG2_MINDMAP_SYSTEM_ZH, HIPPORAG2_MINDMAP_ONE_SHOT_INPUT_ZH, HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT_ZH, HIPPORAG2_MINDMAP_PROMPT_ZH
)
from core.prompts.hipporag2_temporal_prompt import HIPPORAG2_TEMPORAL_PROMPT_ZH, HIPPORAG2_TEMPORAL_SYSTEM_ZH
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
        Also extracts mind map and saves to chunk.metadata
        """
        # Stage 1: Named Entity Recognition
        entities = await self.extract_entities(chunk.content)

        # Stage 2: Triple Extraction using extracted entities
        triples: list[tuple[str, str, str]] = []
        if entities:
            triples = await self.extract_triples(chunk.content, entities)
        else:
            self.logger.warning("No entities extracted, skipping triple extraction")

        # Stage 3: Mind Map Extraction (non-fatal, but must be observable)
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

        # Convert to GraphData format
        graph = self.build_graph_data(entities, triples)
        if chunk.metadata.get(BUSINESS_TIME_KEY):
            graph.metadata[BUSINESS_TIME_KEY] = chunk.metadata.get(BUSINESS_TIME_KEY)
        if chunk.metadata.get(TEMPORAL_ERROR_KEY):
            graph.metadata[TEMPORAL_ERROR_KEY] = chunk.metadata.get(TEMPORAL_ERROR_KEY)
        if chunk.metadata.get(MINDMAP_ERROR_KEY):
            graph.metadata[MINDMAP_ERROR_KEY] = chunk.metadata.get(MINDMAP_ERROR_KEY)
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

        if language == 'zh':
            return HIPPORAG2_TRIPLE_PROMPT_ZH.format(
                system=HIPPORAG2_TRIPLE_SYSTEM_ZH,
                example_input=HIPPORAG2_TRIPLE_ONE_SHOT_INPUT_ZH,
                example_output=HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT_ZH,
                passage=text,
                entities=entities_str
            )
        else:
            return HIPPORAG2_TRIPLE_PROMPT.format(
                system=HIPPORAG2_TRIPLE_SYSTEM,
                example_input=HIPPORAG2_TRIPLE_ONE_SHOT_INPUT,
                example_output=HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT,
                passage=text,
                entities=entities_str
            )

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
        triples = []
        in_triples_section = False
        
        for line in response.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('### TRIPLES'):
                in_triples_section = True
                continue
            
            if line.startswith('###'):
                in_triples_section = False
                continue
            
            if in_triples_section and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        
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
        # For now, use the ZH prompt for both languages (the extractor is used heavily for ZH docs).
        return HIPPORAG2_TEMPORAL_PROMPT_ZH.format(system=HIPPORAG2_TEMPORAL_SYSTEM_ZH, passage=text)

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
