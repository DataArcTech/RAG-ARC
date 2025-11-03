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
import re
from typing import List, TYPE_CHECKING, Tuple

from core.file_management.extractor.base import ExtractorBase
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
    HIPPORAG2_TRIPLE_SYSTEM_ZH, HIPPORAG2_TRIPLE_ONE_SHOT_INPUT_ZH, HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT_ZH, HIPPORAG2_TRIPLE_PROMPT_ZH
)
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

        try:
            return await self.extract_two_stage(chunk)
        except Exception as e:
            self.logger.error(f"Error during HippoRAG2 extraction: {e}")
            return GraphData()

    async def extract_two_stage(self, chunk: Chunk) -> GraphData:
        """
        Two-stage extraction: NER first, then Triple Extraction
        More accurate and follows HippoRAG2's original approach
        """
        try:
            # Stage 1: Named Entity Recognition
            entities = await self.extract_entities(chunk.content)
            
            if not entities:
                self.logger.warning("No entities extracted, skipping triple extraction")
                return GraphData()
            print(entities)
            # Stage 2: Triple Extraction using extracted entities
            triples = await self.extract_triples(chunk.content, entities)
            
            # Convert to GraphData format
            graph_data = self.build_graph_data(entities, triples)
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Error in two-stage extraction: {e}")
            return GraphData()

    async def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Stage 1: Extract named entities from text

        Always extracts entities with types in TSV format: entity\ttype
        - If entity_types is specified: only extract those types
        - If entity_types is None: LLM determines types automatically

        Returns:
            List of (entity_name, entity_type) tuples
        """
        try:
            prompt = self.build_ner_prompt(text)

            response = await self.llm.achat([{"role": "user", "content": prompt}])

            entities = self.parse_ner_response(response)

            self.logger.info(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            self.logger.error(f"Error in entity extraction: {e}")
            return []

    async def extract_triples(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """
        Stage 2: Extract triples using extracted entities

        Args:
            text: Original text
            entities: List of (entity_name, entity_type) tuples

        Returns:
            List of (subject, predicate, object) triples
        """
        try:
            # Extract just entity names for the prompt
            entity_names = [entity[0] for entity in entities]
            prompt = self.build_triple_prompt(text, entity_names)
            response = await self.llm.achat([{"role": "user", "content": prompt}])

            triples = self.parse_triple_response(response)

            self.logger.info(f"Extracted {len(triples)} triples")
            return triples

        except Exception as e:
            self.logger.error(f"Error in triple extraction: {e}")
            return []

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

