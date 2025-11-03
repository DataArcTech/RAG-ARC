"""
Multi-round extraction: in each round, the results extracted from the previous round are added to the context, continuing until the maximum number of rounds is reached or no new entities or relations can be extracted.
The multi-round extraction process is sequential, with each extraction depending on the results of the previous round.
"""

import logging
import re
from typing import Dict, List, TYPE_CHECKING

from core.file_management.extractor.base import ExtractorBase
from core.prompts.extractor_prompt import (
    EXTRACTION_PROMPT, CLEANING_PROMPT, EXTRACTION_PROMPT_EN, CLEANING_PROMPT_EN,
    EXTRACTION_PROMPT_SIMPLE, EXTRACTION_PROMPT_SIMPLE_EN
)
from encapsulation.data_model.schema import Chunk, GraphData

if TYPE_CHECKING:
    from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig

logger = logging.getLogger(__name__)


class GraphExtractor(ExtractorBase):
    """Optimized GraphExtractor, supporting multi-round extraction, cleaning, and bilingual support"""

    def __init__(self, config: "GraphExtractorConfig"):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    async def extract(self, chunk: Chunk) -> GraphData:
        """Main extraction method: automatically supports single-round or multi-round extraction based on max_rounds"""
        if not chunk.content:
            return GraphData()

        accumulated_graph = GraphData()

        # Extraction loop driven by max_rounds (1 round = single-round extraction, >1 round = multi-round extraction)
        for round_num in range(self.config.max_rounds):
            try:
                # Build prompt (supporting bilingual support)
                prompt = self.build_extraction_prompt(chunk.content, accumulated_graph)

                response = await self.llm.achat([{"role": "user", "content": prompt}])
                new_graph = self.parse_tsv_response(response)

                # If there are no new extraction results, end early
                if not new_graph.entities and not new_graph.relations:
                    break

                # Merge results
                accumulated_graph = self.merge_graph_data(accumulated_graph, new_graph)

            except Exception as e:
                self.logger.error(f"Error in round {round_num + 1}: {e}")
                break

        # Clean data if enabled
        if self.config.enable_cleaning:
            accumulated_graph = await self.clean_graph_data(accumulated_graph, chunk)

        # Convert IDs to entity names for final output
        return self.convert_final_output_to_names(accumulated_graph)

    def detect_language(self, text: str) -> str:
        """Detect text language (Chinese or English)"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'\s', '', text))

        if total_chars == 0:
            return 'zh'  # Default Chinese

        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.1 else 'en'

    def build_extraction_prompt(self, text: str, history: GraphData) -> str:
        """Build extraction prompt with user custom priority"""
        language = self.detect_language(text)

        # Use simplified prompt for single-round extraction (max_rounds == 1)
        if self.config.max_rounds == 1:
            if self.config.extraction_prompt:
                template = self.config.extraction_prompt
            else:
                if language == 'en':
                    template = EXTRACTION_PROMPT_SIMPLE_EN
                else:
                    template = EXTRACTION_PROMPT_SIMPLE

            # For single-round extraction, we don't need history
            schema_str = self.generate_schema_string(language=language)
            examples_str = self.generate_examples_string(language=language)

            return template.format(
                text=text,
                schema=schema_str,
                examples=examples_str
            )

        # Use full prompt for multi-round extraction (max_rounds > 1)
        else:
            if self.config.extraction_prompt:
                template = self.config.extraction_prompt
            else:
                if language == 'en':
                    template = EXTRACTION_PROMPT_EN
                else:
                    template = EXTRACTION_PROMPT

            schema_str = self.generate_schema_string(language=language)
            history_str = self.build_history_string(history, language=language)
            examples_str = self.generate_examples_string(language=language)

            return template.format(
                text=text,
                schema=schema_str,
                history=history_str,
                examples=examples_str
            )

    def generate_schema_string(self, language: str = 'zh') -> str:
        """Generate schema string (supporting bilingual support)"""
        parts = []

        if self.config.entity_types:
            entity_types = ", ".join(self.config.entity_types)
            if language == 'en':
                parts.append(f"**Entity Types**: {entity_types}")
            else:
                parts.append(f"**实体类型**: {entity_types}")

        if self.config.relation_types:
            relation_types = ", ".join(self.config.relation_types)
            if language == 'en':
                parts.append(f"**Relation Types**: {relation_types}")
            else:
                parts.append(f"**关系类型**: {relation_types}")

        return "\n".join(parts)

    def generate_examples_string(self, language: str = 'zh') -> str:
        """Generate example string (supporting bilingual support)"""
        parts = []

        if self.config.entity_examples:
            if language == 'en':
                parts.append("**Entity Examples**:")
            else:
                parts.append("**实体示例**:")
            parts.append("### ENTITIES")
            parts.append("id\tname\ttype\tattributes")
            for i, example in enumerate(self.config.entity_examples):
                attr_str = self.format_attributes_string(example.get('attributes', {}))
                parts.append(f"e{i+1}\t{example['name']}\t{example['type']}\t{attr_str}")

        if self.config.relation_examples:
            if parts:
                parts.append("")
            if language == 'en':
                parts.append("**Relation Examples**:")
            else:
                parts.append("**关系示例**:")
            parts.append("### RELATIONS")
            parts.append("head_id\ttype\ttail_id")
            for example in self.config.relation_examples:
                parts.append(f"{example[0]}\t{example[1]}\t{example[2]}")

        return "\n".join(parts)

    def parse_tsv_response(self, response_text: str) -> GraphData:
        """Parse TSV format response from LLM"""
        entities, relations = [], []
        current_section = None

        for line in response_text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('...'):
                continue

            if line.startswith('### ENTITIES'):
                current_section = 'entities'
                continue
            elif line.startswith('### RELATIONS'):
                current_section = 'relations'
                continue
            elif line.startswith(('id\t', 'head_id\t')):
                continue  # Skip header

            # Only process lines containing tabs
            if '\t' not in line:
                continue

            parts = [p.strip() for p in line.split('\t')]
            if len(parts) < 3:
                continue

            try:
                if current_section == 'entities':
                    attr_str = parts[3] if len(parts) > 3 else ''
                    entities.append({
                        'id': parts[0],
                        'entity_name': parts[1],
                        'entity_type': parts[2],
                        'attributes': self.parse_attributes_string(attr_str)
                    })
                elif current_section == 'relations':
                    relations.append(parts[:3])
            except Exception as e:
                logger.warning(f"Failed to parse line '{line}': {e}")

        return GraphData(entities=entities, relations=relations)

    def build_history_string(self, history: GraphData, language: str = 'zh') -> str:
        """Build history data string"""
        if history.is_empty():
            return ""

        history_parts = []

        # Build entity part
        if history.entities:
            if language == 'en':
                history_parts.extend([
                    "Previous extracted data:",
                    "### ENTITIES",
                    "id\tname\ttype\tattributes"
                ])
            else:
                history_parts.extend([
                    "Previous extracted data:",
                    "### ENTITIES",
                    "id\tname\ttype\tattributes"
                ])
            for entity in history.entities:
                entity_id = entity.get('id', '')
                entity_name = entity.get('entity_name', '')
                entity_type = entity.get('entity_type', '')
                attr_str = self.format_attributes_string(entity.get('attributes', {}))
                history_parts.append(f"{entity_id}\t{entity_name}\t{entity_type}\t{attr_str}")

        # Build relation part
        if history.relations:
            if not history.entities:
                history_parts.append("Previous extracted data:")
            history_parts.extend(["", "### RELATIONS", "head_id\ttype\ttail_id"])

            for relation in history.relations:
                if isinstance(relation, list) and len(relation) >= 3:
                    history_parts.append(f"{relation[0]}\t{relation[1]}\t{relation[2]}")

        return "\n".join(history_parts)

    def parse_attributes_string(self, attr_str: str) -> Dict:
        """Parse attribute string: key1|->|value1|#|key2|->|value2"""
        if not attr_str.strip():
            return {}

        attributes = {}
        for part in attr_str.split('|#|'):
            if '|->|' in part:
                try:
                    key, value = part.split('|->|', 1)
                    key, value = key.strip(), value.strip()
                    if key:
                        attributes[key] = value
                except ValueError:
                    continue
        return attributes

    def format_attributes_string(self, attributes: Dict) -> str:
        """Format attributes dictionary to string"""
        if not isinstance(attributes, dict):
            return ''
        return '|#|'.join(f'{k}|->|{v}' for k, v in attributes.items() if k and v is not None)

    async def clean_graph_data(self, graph_data: GraphData, chunk: Chunk) -> GraphData:
        """Clean graph data"""
        try:
            # Basic cleaning
            cleaned_entities = self.clean_entities(graph_data.entities)
            cleaned_relations = self.clean_relations(graph_data.relations, cleaned_entities)

            # LLM-assisted cleaning (optional)
            if self.config.enable_llm_cleaning:
                return await self.llm_clean(GraphData(cleaned_entities, cleaned_relations), chunk)

            return GraphData(cleaned_entities, cleaned_relations)
        except Exception as e:
            self.logger.error(f"Error during cleaning: {e}")
            return graph_data

    def clean_entities(self, entities: List[Dict]) -> List[Dict]:
        cleaned = []
        seen = set()

        for entity in entities:
            name = entity.get('entity_name', '').strip()
            entity_type = entity.get('entity_type', '').strip()

            # Deduplicate
            key = (name.lower(), entity_type.lower())
            if key in seen:
                continue
            seen.add(key)

            # Format check
            if self.is_valid_entity(name):
                cleaned.append(entity)

        return cleaned

    def clean_relations(self, relations: List[List], entities: List[Dict]) -> List[List]:
        entity_names = {e.get('entity_name', '').strip().lower() for e in entities}
        entity_ids = {str(e.get('id', '')).strip() for e in entities if e.get('id')}

        cleaned = []
        seen = set()

        for relation in relations:
            if len(relation) < 3:
                continue

            head, rel_type, tail = str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()

            head_valid = head.lower() in entity_names or head in entity_ids
            tail_valid = tail.lower() in entity_names or tail in entity_ids

            if not (head_valid and tail_valid):
                continue

            if head.lower() == tail.lower():
                continue

            key = (head.lower(), rel_type.lower(), tail.lower())
            if key in seen:
                continue
            seen.add(key)

            cleaned.append([head, rel_type, tail])

        return cleaned

    def is_valid_entity(self, name: str) -> bool:
        """Check if entity name is valid"""
        if not name or len(name.strip()) < 2:
            return False

        # Filter pure numbers
        if re.match(r'^\d+$', name) or re.match(r'^[\d\s\.,;:!?()\[\]{}""''\\-_]+$', name):
            return False

        return True

    def merge_graph_data(self, history: GraphData, new_extraction: GraphData) -> GraphData:
        """Merge history data and new extraction results"""
        merged_entities = list(history.entities)
        merged_relations = list(history.relations)

        # Merge entities
        entity_keys = {(e.get('entity_name', ''), e.get('entity_type', '')) for e in merged_entities}
        for entity in new_extraction.entities:
            key = (entity.get('entity_name', ''), entity.get('entity_type', ''))
            if key not in entity_keys:
                merged_entities.append(entity)
                entity_keys.add(key)

        # Merge relations
        relation_keys = {(str(r[0]), str(r[1]), str(r[2])) for r in merged_relations if len(r) >= 3}
        for relation in new_extraction.relations:
            if len(relation) >= 3:
                key = (str(relation[0]), str(relation[1]), str(relation[2]))
                if key not in relation_keys:
                    merged_relations.append(relation)
                    relation_keys.add(key)

        return GraphData(entities=merged_entities, relations=merged_relations)

    async def llm_clean(self, graph_data: GraphData, chunk: Chunk) -> GraphData:
        """LLM-assisted graph data cleaning"""
        try:
            language = self.detect_language(chunk.content)
            prompt = self.build_cleaning_prompt(chunk.content, graph_data, language)
            response = await self.llm.achat([{"role": "user", "content": prompt}])
            cleaned_graph = self.parse_tsv_response(response)

            if cleaned_graph.is_empty():
                self.logger.warning("LLM cleaning returned empty result, using basic cleaning result")
                return graph_data

            return cleaned_graph

        except Exception as e:
            self.logger.error(f"LLM cleaning failed: {e}")
            return graph_data

    def build_cleaning_prompt(self, text: str, graph_data: GraphData, language: str = 'zh') -> str:
        """Build cleaning prompt with user custom priority"""
        graph_data_str = self.format_graph_for_cleaning(graph_data)

        if self.config.cleaning_prompt:
            template = self.config.cleaning_prompt
        else:
            if language == 'en':
                template = CLEANING_PROMPT_EN
            else:
                template = CLEANING_PROMPT

        return template.format(
            text=text,
            graph_data=graph_data_str
        )

    def format_graph_for_cleaning(self, graph_data: GraphData) -> str:
        """Format graph data as string for cleaning prompt"""
        parts = []

        # Format entities
        if graph_data.entities:
            parts.extend([
                "### ENTITIES",
                "id\tname\ttype\tattributes"
            ])
            for entity in graph_data.entities:
                entity_id = entity.get('id', '')
                entity_name = entity.get('entity_name', '')
                entity_type = entity.get('entity_type', '')
                attr_str = self.format_attributes_string(entity.get('attributes', {}))
                parts.append(f"{entity_id}\t{entity_name}\t{entity_type}\t{attr_str}")

        # Format relations
        if graph_data.relations:
            if parts:
                parts.append("")
            parts.extend([
                "### RELATIONS",
                "head_id\ttype\ttail_id"
            ])
            for relation in graph_data.relations:
                if isinstance(relation, list) and len(relation) >= 3:
                    parts.append(f"{relation[0]}\t{relation[1]}\t{relation[2]}")

        return "\n".join(parts)


    def convert_final_output_to_names(self, graph_data: GraphData) -> GraphData:
        """Convert IDs in final output relationships to entity names"""
        # Create ID to name mapping
        id_to_name = {}
        for entity in graph_data.entities:
            entity_id = entity.get('id', '')
            entity_name = entity.get('entity_name', '')
            if entity_id and entity_name:
                id_to_name[entity_id] = entity_name

        # Convert IDs in relationships to names
        converted_relations = []
        for relation in graph_data.relations:
            if isinstance(relation, list) and len(relation) >= 3:
                head_id, rel_type, tail_id = relation[0], relation[1], relation[2]
                head_name = id_to_name.get(head_id, head_id)
                tail_name = id_to_name.get(tail_id, tail_id)
                converted_relations.append([head_name, rel_type, tail_name])
            else:
                converted_relations.append(relation)

        # Return final output format graph data
        return GraphData(
            entities=graph_data.entities,     # Entities remain unchanged
            relations=converted_relations,    # Relationships use entity names
            metadata=graph_data.metadata
        )

