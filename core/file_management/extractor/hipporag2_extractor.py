"""
HippoRAG2 Graph Extractor (JSON output).

This extractor uses a two-stage approach:
1) NER (entities) as strict JSON (Pydantic-validated)
2) Edge extraction referencing entity IDs as strict JSON (Pydantic-validated)
"""

import logging
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TYPE_CHECKING

from core.file_management.extractor.base import ExtractorBase
from core.file_management.extractor.metadata_keys import (
    BUSINESS_TIME_KEY,
    MINDMAP_ERROR_KEY,
    SDF_ERROR_KEY,
    SDF_HS_KEY,
    SDF_KEY,
    TEMPORAL_ERROR_KEY,
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
from core.prompts.hipporag2_mindmap_prompt import (
    HIPPORAG2_MINDMAP_ONE_SHOT_INPUT,
    HIPPORAG2_MINDMAP_ONE_SHOT_INPUT_ZH,
    HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT,
    HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_MINDMAP_PROMPT,
    HIPPORAG2_MINDMAP_PROMPT_ZH,
    HIPPORAG2_MINDMAP_SYSTEM,
    HIPPORAG2_MINDMAP_SYSTEM_ZH,
)
from core.prompts.hipporag2_extractor_prompt_json import (
    HIPPORAG2_EDGE_JSON_PROMPT_EN,
    HIPPORAG2_EDGE_JSON_PROMPT_ZH,
    HIPPORAG2_EDGE_SCHEMA_HINT_EN,
    HIPPORAG2_EDGE_SCHEMA_HINT_ZH,
    HIPPORAG2_EDGE_JSON_SYSTEM_EN,
    HIPPORAG2_EDGE_JSON_SYSTEM_ZH,
    HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_EN,
    HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_ZH,
    HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_EN,
    HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_NER_JSON_PROMPT_EN,
    HIPPORAG2_NER_JSON_PROMPT_ZH,
    HIPPORAG2_NER_JSON_SYSTEM_EN,
    HIPPORAG2_NER_JSON_SYSTEM_ZH,
)
from core.knowledge_graph.schema import load_schema_from_yaml, normalize_relation_token
from core.knowledge_graph.sdf import hs_to_sdf_schema, parse_hs_blocks
from core.knowledge_graph.extraction_models import ExtractedEntities, ExtractedEdges, ExtractedEntity
from core.utils.structured_output import StructuredOutputError, parse_pydantic_json_from_llm_text
from encapsulation.data_model.schema import Chunk, GraphData

if TYPE_CHECKING:
    from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig

logger = logging.getLogger(__name__)

class HippoRAG2Extractor(ExtractorBase):
    """
    HippoRAG2 Graph Extractor (JSON-only)

    Features:
    - Two-stage extraction: Entities -> Edges
    - Strict JSON outputs validated by Pydantic (controls noise)
    - Optional entity_types filter (soft constraint for the model; still validated)
    """

    def __init__(self, config: "HippoRAG2ExtractorConfig"):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.entity_types = getattr(config, 'entity_types', None)  # Optional entity types to extract
        self._prompt_cache: dict[str, str] = {}
        self._kg_schema = None
        self._kg_domain = None
        self._kg_domain_schema = None
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

    def _schema_hint_for_edges(self, *, language: str) -> str:
        """
        Edge extraction schema hint (does NOT instruct skipping unknown relations).

        The JSON edge prompt already defines the policy for out-of-schema relations (fallback to RELATES_TO),
        so this hint only lists preferred canonical types and aliases.
        """
        schema = getattr(self, "_kg_schema", None)
        domain_schema = getattr(self, "_kg_domain_schema", None)
        if schema is None or domain_schema is None:
            return ""

        max_allowed = int(getattr(self.config, "schema_prompt_max_allowed_relations", 80) or 0)
        max_aliases = int(getattr(self.config, "schema_prompt_max_relation_aliases", 120) or 0)

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

        template = HIPPORAG2_EDGE_SCHEMA_HINT_ZH if language == "zh" else HIPPORAG2_EDGE_SCHEMA_HINT_EN
        return template.format(allowed_relations=allowed_block, relation_aliases_json=alias_json).strip()

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
        Two-stage extraction: Entities -> Edges (JSON-only)

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
        Two-stage extraction: Entities -> Edges (JSON-only)

        Mindmap/temporal/SDF extraction are independent metadata features and remain optional.
        """
        entities_model = await self.extract_entities_json(chunk.content)
        edges_model: ExtractedEdges | None = None
        edge_reference_time: str | None = None
        if entities_model.extracted_entities:
            edges_model, edge_reference_time = await self.extract_edges_json(chunk.content, entities_model)
        else:
            self.logger.warning("No entities extracted (json), skipping edge extraction")
        graph = self.build_graph_data_from_structured(
            entities_model,
            edges_model,
            edge_reference_time=edge_reference_time,
        )

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

    def _edge_reference_time(self) -> str:
        override = str(getattr(self.config, "edge_reference_time_override", "") or "").strip()
        if override:
            return override
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    async def extract_entities_json(self, text: str) -> ExtractedEntities:
        language = self.detect_language(text)
        type_constraint = ""
        if isinstance(getattr(self, "entity_types", None), list) and self.entity_types:
            # Soft constraint: the model still outputs free-form entity_type strings, but must stay JSON-valid.
            joined = ", ".join([str(t).strip() for t in self.entity_types if str(t).strip()])
            if joined:
                type_constraint = (
                    f"\nOnly extract entities of the following types: {joined}\n"
                    if language != "zh"
                    else f"\n仅抽取以下类型的实体：{joined}\n"
                )
        prompt = (HIPPORAG2_NER_JSON_PROMPT_ZH if language == "zh" else HIPPORAG2_NER_JSON_PROMPT_EN).format(
            system=((HIPPORAG2_NER_JSON_SYSTEM_ZH if language == "zh" else HIPPORAG2_NER_JSON_SYSTEM_EN) + type_constraint).strip(),
            example_input=HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_ZH if language == "zh" else HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_EN,
            example_output=HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_ZH if language == "zh" else HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_EN,
            passage=text,
        )
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        try:
            model = parse_pydantic_json_from_llm_text(response, ExtractedEntities)
        except StructuredOutputError as exc:
            raise ValueError(f"NER(JSON) structured output invalid: {exc}") from exc
        self.logger.info("Extracted %s entities (json)", len(model.extracted_entities))
        return model

    async def extract_edges_json(self, text: str, entities: ExtractedEntities) -> tuple[ExtractedEdges, str]:
        language = self.detect_language(text)
        entities_payload = [{"id": e.id, "name": e.name, "entity_type": e.entity_type} for e in entities.extracted_entities]
        reference_time = self._edge_reference_time()
        prompt = (HIPPORAG2_EDGE_JSON_PROMPT_ZH if language == "zh" else HIPPORAG2_EDGE_JSON_PROMPT_EN).format(
            system=HIPPORAG2_EDGE_JSON_SYSTEM_ZH if language == "zh" else HIPPORAG2_EDGE_JSON_SYSTEM_EN,
            edge_types=self._schema_hint_for_edges(language=language) or "",
            entities_json=json.dumps(entities_payload, ensure_ascii=False, separators=(",", ":"), default=str),
            reference_time=reference_time,
            passage=text,
        )
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        try:
            model = parse_pydantic_json_from_llm_text(response, ExtractedEdges)
        except StructuredOutputError as exc:
            raise ValueError(f"Edges(JSON) structured output invalid: {exc}") from exc
        self.logger.info("Extracted %s edges (json)", len(model.edges))
        return model, reference_time

    def build_graph_data_from_structured(
        self, entities: ExtractedEntities, edges: ExtractedEdges | None, *, edge_reference_time: str | None
    ) -> GraphData:
        entity_list: list[dict] = []
        id_to_entity: dict[int, ExtractedEntity] = {}
        for ent in sorted(entities.extracted_entities, key=lambda e: e.id):
            id_to_entity[ent.id] = ent
            entity_list.append(
                {
                    "id": f"e{ent.id}",
                    "entity_name": ent.name,
                    "entity_type": ent.entity_type,
                    "attributes": {},
                }
            )

        relations: list[list] = []
        dropped_invalid = 0
        dropped_self = 0
        if edges is not None:
            for edge in edges.edges:
                if edge.source_entity_id == edge.target_entity_id:
                    dropped_self += 1
                    continue
                src = id_to_entity.get(edge.source_entity_id)
                dst = id_to_entity.get(edge.target_entity_id)
                if src is None or dst is None:
                    dropped_invalid += 1
                    continue
                rel: list = [src.name, edge.relation_type, dst.name]
                if edge.valid_at is not None or edge.invalid_at is not None or edge.fact is not None:
                    rel.extend([edge.valid_at, edge.invalid_at, edge.fact])
                relations.append(rel)

        graph = GraphData(entities=entity_list, relations=relations, metadata={})
        graph.metadata["graph_output_format"] = "json"
        if edge_reference_time:
            graph.metadata["edge_reference_time"] = edge_reference_time
        if dropped_invalid or dropped_self:
            graph.metadata["edge_filter_stats"] = {
                "dropped_invalid_entity_ids": dropped_invalid,
                "dropped_self_loops": dropped_self,
            }
        return graph
    
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
        
        for raw_line in str(response or "").splitlines():
            line = raw_line.strip()
            
            if not line:
                continue
                
            normalized_header = line.lstrip("#").strip().lower().replace(" ", "")
            if normalized_header in {"mindmap", "mind-map"}:
                in_mindmap_section = True
                continue
            
            if line.startswith("#"):
                in_mindmap_section = False
                continue
            
            if in_mindmap_section and '\t' in line:
                candidate = line.lstrip("-•* ").strip()
                parts = candidate.split('\t', 1)  # Split only on first tab
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
