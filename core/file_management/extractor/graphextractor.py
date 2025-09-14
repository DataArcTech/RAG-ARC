import logging
import re
from typing import Dict, List, Optional, Literal
from pydantic import Field

from core.file_management.extractor.base import ExtractorBase, ExtractorBaseConfig
from core.prompts.extractor_prompt import EXTRACTION_PROMPT
from core.utils.data_model import Document
from encapsulation.llm.base import LLMBase

logger = logging.getLogger(__name__)


class GraphExtractorConfig(ExtractorBaseConfig):
    """Configuration for GraphExtractor"""
    type: Literal['graph_extractor'] = 'graph_extractor'
    
    entity_types: Optional[List[str]] = Field(default=None, description="Optional predefined entity types")
    relation_types: Optional[List[str]] = Field(default=None, description="Optional predefined relation types")
    extract_prompt: str = Field(default=EXTRACTION_PROMPT)
    enable_cleaning: bool = Field(default=True, description="Whether to enable cleaning functionality")
    max_rounds: int = Field(default=3, description="Maximum number of extraction rounds", ge=1)
    
    def build(self) -> 'GraphExtractor':
        """Build GraphExtractor instance"""
        llm = self.llm_config.build()
        return GraphExtractor(config=self, llm=llm)


class GraphExtractor(ExtractorBase[GraphExtractorConfig]):
    """GraphExtractor extracts entities and relations from text using TSV format for LLM interaction."""
    
    def __init__(self, config: GraphExtractorConfig, llm: LLMBase):
        super().__init__(config)
        self.llm = llm
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.max_rounds <= 0:
            raise ValueError("max_rounds must be greater than 0")
    
    def _generate_schema_string(self) -> str:
        """Generate schema string for LLM prompt"""
        schema_parts = []
        
        if self.config.entity_types:
            entity_types = ", ".join(self.config.entity_types)
            schema_parts.append(f"**Entity Types**: {entity_types}")
        
        if self.config.relation_types:
            relation_types = ", ".join(self.config.relation_types)
            schema_parts.append(f"**Relation Types**: {relation_types}")
        
        return "\n".join(schema_parts) if schema_parts else ""

    def _parse_tsv_response(self, response_text: str) -> Dict[str, List]:
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
                        'attributes': self._parse_attributes_string(attr_str)
                    })
                elif current_section == 'relations':
                    relations.append(parts[:3])
            except Exception as e:
                logger.warning(f"Failed to parse line '{line}': {e}")
                    
        return {'entities': entities, 'relations': relations}
        
    def _parse_attributes_string(self, attr_str: str) -> Dict:
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
        
    def _format_attributes_string(self, attributes: Dict) -> str:
        """Format attributes dictionary to string"""
        if not isinstance(attributes, dict):
            return ''
        return '|#|'.join(f'{k}|->|{v}' for k, v in attributes.items() if k and v is not None)
    
    async def _aextract(self, document: Document) -> Document:
        """Complete multi-round extraction + cleaning process to extract graph structure from a single document"""
        if not document.metadata:
            document.metadata = {}
            
        if not document.content:
            document.metadata.update({'entities': [], 'relations': []})
            return document

        # Initialize history data
        history = {
            'entities': document.metadata.get('entities', []) if isinstance(document.metadata.get('entities'), list) else [],
            'relations': document.metadata.get('relations', []) if isinstance(document.metadata.get('relations'), list) else []
        }

        # Multi-round extraction
        for i in range(self.config.max_rounds):
            try:
                new_result = await self._single_round_extract(document, history)
                if not new_result.get('entities') and not new_result.get('relations'):
                    break
                history = self._merge_graph_data(history, new_result)
            except Exception as e:
                logger.error(f"Error in round {i + 1}: {e}")
                break

        # Update document metadata
        document.metadata.update(history)
        
        # Clean if enabled
        if self.config.enable_cleaning:
            document = await self._aclean(document)
        else:
            # Convert relation entity IDs to entity names
            document.metadata['relations'] = self._convert_relations_to_entity_names(
                document.metadata['relations'], document.metadata['entities']
            )
        
        return document
        
    async def _single_round_extract(self, document: Document, history: Dict[str, List]) -> Dict[str, List]:
        """Single round extraction using TSV format"""
        try:
            prompt = self.config.extract_prompt.format(
                text=document.content, 
                history=self._build_history_string(history),
                schema=self._generate_schema_string()
            )
            
            llm_response = await self.llm.achat([{"role": "user", "content": prompt}])
            response_content = self._extract_response_content(llm_response)
            
            return self._parse_tsv_response(response_content) if response_content else {'entities': [], 'relations': []}
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return {'entities': [], 'relations': []}
    
    def _build_history_string(self, history: Dict[str, List]) -> str:
        """Build TSV format history data string"""
        if not (history.get('entities') or history.get('relations')):
            return ""
        
        history_parts = []
        entities = history.get('entities', [])
        
        # Ensure entities have IDs
        self._ensure_entity_ids(entities)
        
        # Build TSV format for entities
        if entities:
            history_parts.extend([
                "Previous extracted data:",
                "### ENTITIES",
                "id\tname\ttype\tattributes"
            ])
            for ent in entities:
                ent_id = str(ent.get('id', '')).strip()
                entity_name = ent.get('entity_name') or ent.get('name') or ''
                entity_type = ent.get('entity_type') or ent.get('type') or ''
                attr_str = self._format_attributes_string(ent.get('attributes', {}))
                history_parts.append(f"{ent_id}\t{entity_name}\t{entity_type}\t{attr_str}")
        
        # Build TSV format for relations
        relations = history.get('relations', [])
        if relations:
            if not entities:
                history_parts.append("Previous extracted data:")
            history_parts.extend(["", "### RELATIONS", "head_id\ttype\ttail_id"])
            
            # Build name->ID mapping
            name_to_id = {(ent.get('entity_name') or ent.get('name', '')).strip(): str(ent.get('id', '')).strip() 
                         for ent in entities if ent.get('entity_name') or ent.get('name')}
            id_set = {str(ent.get('id', '')).strip() for ent in entities if ent.get('id')}
            
            for rel in relations:
                if isinstance(rel, list) and len(rel) >= 3:
                    head, rtype, tail = str(rel[0]).strip(), str(rel[1]).strip(), str(rel[2]).strip()
                    head = head if head in id_set else name_to_id.get(head, head)
                    tail = tail if tail in id_set else name_to_id.get(tail, tail)
                    history_parts.append(f"{head}\t{rtype}\t{tail}")
        
        return "\n".join(history_parts)
    
    def _ensure_entity_ids(self, entities: List[Dict]) -> None:
        """Ensure entities have stable IDs"""
        used_ids = {str(e.get('id', '')).strip() for e in entities if e.get('id')}
        counter = 1
        for ent in entities:
            if not ent.get('id') or not str(ent.get('id', '')).strip():
                while f"e{counter}" in used_ids:
                    counter += 1
                ent['id'] = f"e{counter}"
                used_ids.add(ent['id'])
                counter += 1
    
    def _extract_response_content(self, llm_response) -> str:
        """Extract content from LLM response"""
        if llm_response is None:
            return ""
        if isinstance(llm_response, str):
            return llm_response
        if hasattr(llm_response, 'content'):
            return str(llm_response.content)
        return str(llm_response)


    async def _aclean(self, document: Document) -> Document:
        """Asynchronously clean graph structure from a single document"""
        if not self._has_valid_triples(document):
            return document
            
        entities = document.metadata.get('entities', [])
        relations = document.metadata.get('relations', [])

        try:
            cleaned_entities = self._pre_filter_entities(entities)
            cleaned_relations = self._clean_relations(relations, cleaned_entities)
            cleaned_relations = self._convert_relations_to_entity_names(cleaned_relations, cleaned_entities)
            
            document.metadata.update({
                'entities': cleaned_entities,
                'relations': cleaned_relations
            })
        except Exception as e:
            logger.error(f"Error during cleaning process: {e}")

        return document
    
    def _has_valid_triples(self, document: Document) -> bool:
        """Check if document has valid entities or relations data"""
        return (
            document.metadata and 
            (document.metadata.get('entities') or document.metadata.get('relations'))
        )


    def _pre_filter_entities(self, entities: List[Dict]) -> List[Dict]:
        """Pre-process filter entities, remove obviously useless entities"""
        if not entities:
            return []
        
        filtered_entities = []
        for entity in entities:
            entity_name = entity.get('entity_name', '').strip()
            if not entity_name:
                continue
                
            # Filter pure numbers or pure punctuation
            if (re.match(r'^\d+$', entity_name) or 
                re.match(r'^[\d\s\.,;:!?()\[\]{}""''\-_]+$', entity_name)):
                continue
                
            filtered_entities.append(entity)
        
        return filtered_entities


    def _clean_relations(self, relations: List[List], cleaned_entities: List[Dict]) -> List[List]:
        """Clean relations, remove invalid relations (including non-existent head/tail entities, self-loop relations)"""
        if not relations:
            return []
        
        # Build valid entity sets
        valid_entity_ids = {str(entity.get('id', '')).strip() for entity in cleaned_entities if entity.get('id')}
        valid_entity_names = {entity.get('entity_name', '').strip() for entity in cleaned_entities if entity.get('entity_name')}
        
        cleaned_relations = []
        for relation in relations:
            if not isinstance(relation, list) or len(relation) < 3:
                continue
                
            head, relation_type, tail = str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()
            
            # Check entity validity and relation type
            head_valid = head in valid_entity_ids or head in valid_entity_names
            tail_valid = tail in valid_entity_ids or tail in valid_entity_names
            
            if head_valid and tail_valid and head != tail and relation_type:
                cleaned_relations.append([head, relation_type, tail])
        
        return cleaned_relations
    
    def _convert_relations_to_entity_names(self, relations: List[List], entities: List[Dict]) -> List[List]:
        """Convert entity IDs in relations to entity names"""
        if not relations or not entities:
            return relations
            
        # Build ID to name mapping
        id_to_name = {str(entity.get('id', '')).strip(): entity.get('entity_name', '').strip()
                     for entity in entities if entity.get('id') and entity.get('entity_name')}
        
        converted_relations = []
        for relation in relations:
            if isinstance(relation, list) and len(relation) >= 3:
                head, relation_type, tail = str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()
                converted_relations.append([
                    id_to_name.get(head, head),
                    relation_type,
                    id_to_name.get(tail, tail)
                ])
        return converted_relations


        
    # ==================== Data Merging Methods ====================
    
    def _merge_graph_data(self, history: Dict[str, List], new_extraction: Dict[str, List]) -> Dict[str, List]:
        """Merge history and new extraction results, deduplicate"""
        entities = list(history.get('entities', []))
        relations = self._normalize_relations(history.get('relations', []))
        
        entities, _ = self._merge_entities(entities, new_extraction.get('entities', []))
        relations = self._merge_relations(relations, new_extraction.get('relations', []))
        
        return {'entities': entities, 'relations': relations}
        
    def _normalize_relations(self, relations: List) -> List[List]:
        """Normalize relations to triple list"""
        return [list(r) for r in relations if isinstance(r, (list, tuple)) and len(r) == 3]
        
    def _merge_entities(self, existing_entities: List[Dict], new_entities: List[Dict]) -> tuple:
        """Merge entities by (entity_name, entity_type), deduplicate"""
        entity_map = {(e.get('entity_name'), e.get('entity_type')): e for e in existing_entities}
        
        for ent in new_entities:
            name = ent.get('entity_name') or ent.get('name')
            etype = ent.get('entity_type') or ent.get('type')
            if not name or not etype:
                continue
                
            key = (name, etype)
            if key not in entity_map:
                new_entity = {
                    'id': ent.get('id'),
                    'entity_name': name,
                    'entity_type': etype,
                    'attributes': ent.get('attributes', {}) if isinstance(ent.get('attributes'), dict) else {}
                }
                existing_entities.append(new_entity)
                entity_map[key] = new_entity
            else:
                self._merge_entity_attributes(entity_map[key], ent.get('attributes', {}))
                if ent.get('id') and not entity_map[key].get('id'):
                    entity_map[key]['id'] = ent.get('id')
                
        return existing_entities, entity_map
        
    def _merge_entity_attributes(self, target_entity: Dict, new_attrs: Dict):
        """Merge attributes to target entity (only add missing keys)"""
        if not isinstance(target_entity.get('attributes'), dict):
            target_entity['attributes'] = {}
        if isinstance(new_attrs, dict):
            for key, value in new_attrs.items():
                if key not in target_entity['attributes']:
                    target_entity['attributes'][key] = value
                    
    def _merge_relations(self, existing_relations: List[List], new_relations: List) -> List[List]:
        """Merge relations by (head, relation_type, tail), deduplicate"""
        rel_keys = {(str(h), str(p), str(o)) for h, p, o in existing_relations}
        
        for r in new_relations:
            if isinstance(r, (list, tuple)) and len(r) == 3:
                key = (str(r[0]), str(r[1]), str(r[2]))
                if all(key) and key not in rel_keys:
                    existing_relations.append([r[0], r[1], r[2]])
                    rel_keys.add(key)
                    
        return existing_relations