"""
HippoRAG2 Graph Extractor - Optimized for minimal token usage with TSV format

This extractor follows HippoRAG2's approach:
1. Named Entity Recognition (NER) - extract entities first
2. Triple Extraction - construct RDF triples using extracted entities
3. TSV format output to minimize token usage
4. Support for both separate NER+Triple and combined OpenIE modes
"""

import logging
import re
from typing import Dict, List, TYPE_CHECKING, Tuple

from core.file_management.extractor.base import ExtractorBase
from core.prompts.hipporag2_extractor_prompt import (
    HIPPORAG2_NER_SYSTEM, HIPPORAG2_NER_ONE_SHOT_INPUT, HIPPORAG2_NER_ONE_SHOT_OUTPUT, HIPPORAG2_NER_PROMPT,
    HIPPORAG2_TRIPLE_SYSTEM, HIPPORAG2_TRIPLE_ONE_SHOT_INPUT, HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT, HIPPORAG2_TRIPLE_PROMPT,
    HIPPORAG2_OPENIE_SYSTEM, HIPPORAG2_OPENIE_ONE_SHOT_INPUT, HIPPORAG2_OPENIE_ONE_SHOT_OUTPUT, HIPPORAG2_OPENIE_PROMPT,
    HIPPORAG2_OPENIE_SYSTEM_ZH, HIPPORAG2_OPENIE_ONE_SHOT_INPUT_ZH, HIPPORAG2_OPENIE_ONE_SHOT_OUTPUT_ZH
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
    - TSV format output (tab-separated values)
    - Minimal token usage compared to JSON format
    - Bilingual support (English and Chinese)
    - Optional combined OpenIE mode (NER + Triple in one call)
    """

    def __init__(self, config: "HippoRAG2ExtractorConfig"):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.combined_mode = getattr(config, 'combined_mode', False)  # Whether to use combined OpenIE

    async def extract(self, chunk: Chunk) -> GraphData:
        """
        Main extraction method for HippoRAG2
        
        Args:
            chunk: Input chunk to extract from
            
        Returns:
            GraphData with entities and relations
        """
        if not chunk.content:
            return GraphData()

        try:
            if self.combined_mode:
                # Combined OpenIE: NER + Triple Extraction in one call
                return await self.extract_combined(chunk)
            else:
                # Two-stage: NER first, then Triple Extraction
                return await self.extract_two_stage(chunk)
        except Exception as e:
            self.logger.error(f"Error during HippoRAG2 extraction: {e}")
            return GraphData()

    async def extract_combined(self, chunk: Chunk) -> GraphData:
        """
        Combined OpenIE: Extract entities and triples in one LLM call
        More efficient but may be less accurate
        """
        try:
            # Detect language
            language = self.detect_language(chunk.content)
            
            # Build prompt
            prompt = self.build_openie_prompt(chunk.content, language)
            
            # Call LLM
            response = await self.llm.achat([{"role": "user", "content": prompt}])
            
            # Parse response
            graph_data = self.parse_openie_response(response)
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Error in combined extraction: {e}")
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
            
            # Stage 2: Triple Extraction using extracted entities
            triples = await self.extract_triples(chunk.content, entities)
            
            # Convert to GraphData format
            graph_data = self.build_graph_data(entities, triples)
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Error in two-stage extraction: {e}")
            return GraphData()

    async def extract_entities(self, text: str) -> List[str]:
        """
        Stage 1: Extract named entities from text
        
        Returns:
            List of entity names
        """
        try:
            language = self.detect_language(text)
            prompt = self.build_ner_prompt(text, language)
            
            response = await self.llm.achat([{"role": "user", "content": prompt}])
            
            entities = self.parse_ner_response(response)
            
            self.logger.info(f"Extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {e}")
            return []

    async def extract_triples(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """
        Stage 2: Extract triples using extracted entities
        
        Args:
            text: Original text
            entities: List of extracted entity names
            
        Returns:
            List of (subject, predicate, object) triples
        """
        try:
            language = self.detect_language(text)
            prompt = self.build_triple_prompt(text, entities, language)
            
            response = await self.llm.achat([{"role": "user", "content": prompt}])
            
            triples = self.parse_triple_response(response)
            
            self.logger.info(f"Extracted {len(triples)} triples")
            return triples
            
        except Exception as e:
            self.logger.error(f"Error in triple extraction: {e}")
            return []

    def detect_language(self, text: str) -> str:
        """Detect text language (Chinese or English)"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.1 else 'en'

    def build_ner_prompt(self, text: str, language: str = 'en') -> str:
        """Build NER prompt"""
        return HIPPORAG2_NER_PROMPT.format(
            system=HIPPORAG2_NER_SYSTEM,
            example_input=HIPPORAG2_NER_ONE_SHOT_INPUT,
            example_output=HIPPORAG2_NER_ONE_SHOT_OUTPUT,
            passage=text
        )

    def build_triple_prompt(self, text: str, entities: List[str], language: str = 'en') -> str:
        """Build triple extraction prompt"""
        entities_str = '\n'.join(entities)
        
        return HIPPORAG2_TRIPLE_PROMPT.format(
            system=HIPPORAG2_TRIPLE_SYSTEM,
            example_input=HIPPORAG2_TRIPLE_ONE_SHOT_INPUT,
            example_output=HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT,
            passage=text,
            entities=entities_str
        )

    def build_openie_prompt(self, text: str, language: str = 'en') -> str:
        """Build combined OpenIE prompt"""
        if language == 'zh':
            return HIPPORAG2_OPENIE_PROMPT.format(
                system=HIPPORAG2_OPENIE_SYSTEM_ZH,
                example_input=HIPPORAG2_OPENIE_ONE_SHOT_INPUT_ZH,
                example_output=HIPPORAG2_OPENIE_ONE_SHOT_OUTPUT_ZH,
                passage=text
            )
        else:
            return HIPPORAG2_OPENIE_PROMPT.format(
                system=HIPPORAG2_OPENIE_SYSTEM,
                example_input=HIPPORAG2_OPENIE_ONE_SHOT_INPUT,
                example_output=HIPPORAG2_OPENIE_ONE_SHOT_OUTPUT,
                passage=text
            )

    def parse_ner_response(self, response: str) -> List[str]:
        """
        Parse NER response in TSV format
        
        Expected format:
        ### ENTITIES
        Entity1
        Entity2
        ...
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
                entities.append(line)
        
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

    def parse_openie_response(self, response: str) -> GraphData:
        """Parse combined OpenIE response"""
        entities = self.parse_ner_response(response)
        triples = self.parse_triple_response(response)
        return self.build_graph_data(entities, triples)

    def build_graph_data(self, entities: List[str], triples: List[Tuple[str, str, str]]) -> GraphData:
        """
        Convert entities and triples to GraphData format
        
        Args:
            entities: List of entity names
            triples: List of (subject, predicate, object) tuples
            
        Returns:
            GraphData object
        """
        # Build entity list with IDs
        entity_list = []
        entity_name_to_id = {}
        
        for i, entity_name in enumerate(entities):
            entity_id = f"e{i+1}"
            entity_list.append({
                'id': entity_id,
                'entity_name': entity_name,
                'entity_type': 'Entity',  # HippoRAG2 doesn't classify entity types
                'attributes': {}
            })
            entity_name_to_id[entity_name] = entity_id
        
        # Build relation list using entity names (as per RAG-ARC convention)
        relation_list = []
        for subject, predicate, obj in triples:
            # Use entity names directly (RAG-ARC's graph_retrieval expects names)
            relation_list.append([subject, predicate, obj])
        
        return GraphData(entities=entity_list, relations=relation_list, metadata={})

